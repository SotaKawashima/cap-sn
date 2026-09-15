#!/usr/bin/env python3
"""Audit and analyze the revised final evaluation after candidate reselection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.candidate_validation_analysis import build_seed_summary
from analysis.final_evaluation_analysis import build_final_condition_summary
from analysis.final_retest_analysis import (
    FINAL_CANDIDATE_ROLE,
    REFERENCE_IDS,
    build_candidate_seed_performance,
    build_final_candidate_effects,
    build_final_candidate_results,
    build_final_decision,
    build_network_conclusions,
    build_validation_retest_comparison,
    candidate_metadata,
    load_final_retest,
)
from experiment_runtime import (
    REPO_ROOT,
    git_state,
    now_iso,
    relative_to_repo,
    sha256_file,
    validate_safe_name,
    write_json,
)
from run_stage8_final_retest import (
    DEFAULT_PROTOCOL_PATH,
    EXECUTION_PLAN_NAME,
    STAGE,
    build_specs,
    load_protocol,
)


DEFAULT_ANALYSIS_ID = "final_retest_analysis_v01"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_source(
    experiment_root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    specs: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = experiment_root / EXECUTION_PLAN_NAME
    study_path = experiment_root / "study_manifest.json"
    if not plan_path.is_file() or not study_path.is_file():
        raise FileNotFoundError(
            "revised final execution plan or study manifest is missing"
        )
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != STAGE or study.get("stage") != STAGE:
        raise ValueError("revised final source manifests have an unexpected stage")
    experiment_id = str(plan.get("experiment_id", ""))
    if not experiment_id or experiment_id != experiment_root.name:
        raise ValueError("execution-plan experiment ID does not match its directory")
    if study.get("experiment_id") != experiment_id:
        raise ValueError("study-manifest experiment ID does not match execution plan")

    expected_counts = {
        "total": len(specs),
        "completed": len(specs),
        "pending": 0,
        "other": 0,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError(
            f"revised final execution plan is incomplete: {plan.get('counts')!r}"
        )
    if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("local revised final protocol does not match execution")
    candidate_path = REPO_ROOT / protocol["candidate_source"]["path"]
    if plan.get("candidate_source", {}).get("sha256") != sha256_file(
        candidate_path
    ):
        raise ValueError("local frozen candidate source does not match execution")

    expected_by_dir = {spec.relative_run_dir: spec for spec in specs}
    if len(expected_by_dir) != len(specs):
        raise ValueError("revised final specification contains duplicate run paths")
    plan_rows = {
        str(row.get("relative_run_dir")): row for row in plan.get("runs", [])
    }
    if set(plan_rows) != set(expected_by_dir):
        raise ValueError("execution-plan run directories do not match protocol")
    if any(row.get("status") != "completed" for row in plan_rows.values()):
        raise ValueError("execution plan contains a non-completed run")
    for relative_dir, spec in expected_by_dir.items():
        row = plan_rows[relative_dir]
        expected = {
            "key": spec.key,
            "network": spec.network,
            "condition_id": spec.condition_id,
            "condition_role": spec.condition_role,
            "simulator_seed": spec.simulator_seed,
            "iterations": spec.iterations,
            "raw_level": spec.raw_level,
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError(f"execution-plan row changed: {relative_dir}")

    study_rows = {str(row.get("path")): row for row in study.get("runs", [])}
    if set(study_rows) != set(expected_by_dir):
        raise ValueError("study-manifest run directories do not match protocol")
    if any(row.get("status") != "completed" for row in study_rows.values()):
        raise ValueError("study manifest contains a non-completed run")

    manifest_paths = sorted(
        experiment_root.glob("*/*/simseed_*/manifest.json")
    )
    observed_dirs = {
        path.parent.relative_to(experiment_root).as_posix()
        for path in manifest_paths
    }
    if observed_dirs != set(expected_by_dir):
        raise ValueError("raw run manifests do not match the frozen protocol")
    executed_commit = plan.get("git", {}).get("commit")
    if not executed_commit or plan.get("git", {}).get("dirty"):
        raise ValueError("revised final execution was not tied to a clean commit")
    for path in manifest_paths:
        manifest = _read_json(path)
        if manifest.get("experiment_id") != experiment_id:
            raise ValueError(f"run experiment ID differs from execution plan: {path}")
        if manifest.get("git", {}).get("commit") != executed_commit:
            raise ValueError(f"run Git commit differs from execution plan: {path}")
        if manifest.get("git", {}).get("dirty"):
            raise ValueError(f"run used a dirty Git worktree: {path}")
    return plan, study


def _output_entry(path: Path, rows: int | None = None) -> dict[str, Any]:
    relative_path = (
        f"{path.parent.name}/{path.name}"
        if path.parent.name in {"tables", "figures"}
        else path.name
    )
    entry: dict[str, Any] = {
        "path": relative_path,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        entry["rows"] = int(rows)
    return entry


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    outputs: dict[str, dict[str, Any]],
    key: str,
) -> None:
    frame.to_csv(path, index=False)
    outputs[key] = _output_entry(path, len(frame))


def _write_parquet(
    frame: pd.DataFrame,
    path: Path,
    outputs: dict[str, dict[str, Any]],
    key: str,
) -> None:
    frame.to_parquet(path, index=False)
    outputs[key] = _output_entry(path, len(frame))


def run_analysis(args: argparse.Namespace) -> Path:
    experiment_root = args.experiment_root.resolve()
    protocol_path = args.protocol.resolve()
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    analysis_root = experiment_root / analysis_id
    if analysis_root.exists():
        raise FileExistsError(f"analysis directory already exists: {analysis_root}")
    tables_root = analysis_root / "tables"
    figures_root = analysis_root / "figures"
    tables_root.mkdir(parents=True)
    figures_root.mkdir()

    started = time.perf_counter()
    manifest_path = analysis_root / "analysis_manifest.json"
    module_path = REPO_ROOT / "analysis" / "final_retest_analysis.py"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "stage": "stage8_final_retest_analysis",
        "status": "running",
        "started_at": now_iso(),
        "source_experiment_root": relative_to_repo(experiment_root),
        "protocol": {
            "path": relative_to_repo(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "analysis_code": {
            "cli": {
                "path": relative_to_repo(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "module": {
                "path": relative_to_repo(module_path),
                "sha256": sha256_file(module_path),
            },
        },
        "git": git_state(),
        "invocation": sys.argv,
        "outputs": {},
        "failure": None,
    }
    write_json(manifest_path, manifest)

    try:
        protocol = load_protocol(protocol_path)
        specs = build_specs(protocol)
        plan, study = _validate_source(
            experiment_root, protocol_path, protocol, specs
        )
        quality = protocol["analysis_plan"]["quality_gate"]
        tolerance = float(quality["numeric_tolerance"])
        bootstrap = protocol["inference"]["bootstrap"]
        repetitions = int(bootstrap["repetitions"])
        bootstrap_seed = int(bootstrap["seed"])

        data = load_final_retest(
            str(experiment_root),
            expected_specs=specs,
            numeric_tolerance=tolerance,
        )
        metadata = candidate_metadata(protocol)
        condition_seed = build_seed_summary(data.iterations)
        condition_summary = build_final_condition_summary(
            data.iterations,
            repetitions=repetitions,
            seed=bootstrap_seed,
        )
        candidate_seed = build_candidate_seed_performance(
            data.iterations, metadata
        )
        effects_by_reference = {
            reference: build_final_candidate_effects(
                data.iterations,
                reference_id=reference,
                repetitions=repetitions,
                seed=bootstrap_seed + (index + 1) * 1000,
            )
            for index, reference in enumerate(REFERENCE_IDS)
        }
        final_results = build_final_candidate_results(
            condition_summary,
            metadata,
            effects_by_reference,
        )
        validation_retest = build_validation_retest_comparison(final_results)
        network_conclusions = build_network_conclusions(final_results)
        seed_policy = protocol["seed_policy"]
        final_seeds = [int(seed) for seed in seed_policy["revised_final_test"]]
        inspected_seeds = [
            int(seed) for seed in seed_policy["inspected_original_final_test"]
        ]
        decision = build_final_decision(
            final_results,
            network_conclusions,
            final_test_seeds=final_seeds,
            inspected_original_final_test_seeds=inspected_seeds,
        )

        outputs: dict[str, dict[str, Any]] = {}
        _write_csv(
            data.audit,
            tables_root / "data_audit.csv",
            outputs,
            "data_audit",
        )
        _write_csv(
            data.runs,
            tables_root / "run_inventory.csv",
            outputs,
            "run_inventory",
        )
        _write_parquet(
            data.iterations,
            tables_root / "iteration_metrics.parquet",
            outputs,
            "iteration_metrics",
        )
        _write_csv(
            condition_seed,
            tables_root / "condition_seed_summary.csv",
            outputs,
            "condition_seed_summary",
        )
        _write_csv(
            condition_summary,
            tables_root / "condition_summary.csv",
            outputs,
            "condition_summary",
        )
        _write_csv(
            candidate_seed,
            tables_root / "candidate_seed_performance.csv",
            outputs,
            "candidate_seed_performance",
        )
        for reference in REFERENCE_IDS:
            key = f"candidate_effects_vs_{reference}"
            _write_csv(
                effects_by_reference[reference],
                tables_root / f"{key}.csv",
                outputs,
                key,
            )
        _write_csv(
            validation_retest,
            tables_root / "validation_retest_comparison.csv",
            outputs,
            "validation_retest_comparison",
        )
        _write_csv(
            final_results,
            tables_root / "final_candidate_results.csv",
            outputs,
            "final_candidate_results",
        )
        _write_csv(
            network_conclusions,
            tables_root / "network_conclusions.csv",
            outputs,
            "network_conclusions",
        )

        decision_path = analysis_root / "decision.json"
        write_json(decision_path, decision)
        outputs["decision"] = _output_entry(decision_path)
        candidate_run_count = int(
            data.runs["condition_role"].eq(FINAL_CANDIDATE_ROLE).sum()
        )
        summary = {
            "status": "completed",
            "experiment_id": str(plan["experiment_id"]),
            "executed_git_commit": plan.get("git", {}).get("commit"),
            "run_count": int(len(data.runs)),
            "iteration_metric_count": int(len(data.iterations)),
            "valid_run_count": int(data.audit["valid"].sum()),
            "candidate_count": int(len(metadata)),
            "candidate_run_count": candidate_run_count,
            "reference_run_count": int(len(data.runs) - candidate_run_count),
            "selected_candidate_count_after_test": 0,
            "candidate_reranking_performed": False,
            "candidate_reselection_performed": False,
            "execution_plan_counts": plan["counts"],
            "study_manifest_run_count": int(len(study.get("runs", []))),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": bootstrap_seed,
            "numeric_tolerance": tolerance,
            "references": list(REFERENCE_IDS),
            "revised_final_test_seeds_used": final_seeds,
            "inspected_original_final_test_seeds_reused": False,
            "decision": decision,
        }
        summary_path = analysis_root / "analysis_summary.json"
        write_json(summary_path, summary)
        outputs["analysis_summary"] = _output_entry(summary_path)

        manifest.update(
            {
                "status": "completed",
                "completed_at": now_iso(),
                "elapsed_sec": time.perf_counter() - started,
                "source_execution": {
                    "experiment_id": plan["experiment_id"],
                    "git_commit": plan.get("git", {}).get("commit"),
                    "run_count": len(specs),
                },
                "outputs": outputs,
            }
        )
        write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": now_iso(),
                "elapsed_sec": time.perf_counter() - started,
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        write_json(manifest_path, manifest)
        raise
    return analysis_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analysis_root = run_analysis(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Completed revised final-evaluation analysis: {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
