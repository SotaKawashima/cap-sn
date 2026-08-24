#!/usr/bin/env python3
"""Audit Stage 7 validation runs and freeze the final-test shortlist."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.candidate_validation_analysis import (
    build_candidate_block_performance,
    build_candidate_effects,
    build_candidate_selection,
    build_condition_summary,
    build_exploration_validation_comparison,
    build_seed_summary,
    load_candidate_validation,
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
from run_stage7_candidate_validation import (
    DEFAULT_PROTOCOL_PATH,
    EXECUTION_PLAN_NAME,
    STAGE,
    build_specs,
    load_protocol,
)


DEFAULT_ANALYSIS_ID = "candidate_validation_analysis_v01"


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
            "Stage 7 execution plan or study manifest is missing"
        )
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != STAGE or study.get("stage") != STAGE:
        raise ValueError("Stage 7 source manifests have an unexpected stage")
    expected_counts = {
        "total": len(specs),
        "completed": len(specs),
        "pending": 0,
        "other": 0,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError(
            f"Stage 7 execution plan is incomplete: {plan.get('counts')!r}"
        )
    if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("local Stage 7 protocol does not match the execution")
    candidate_path = REPO_ROOT / protocol["candidate_source"]["path"]
    if plan.get("candidate_source", {}).get("sha256") != sha256_file(
        candidate_path
    ):
        raise ValueError("local Stage 7 candidate source does not match execution")

    expected_by_dir = {spec.relative_run_dir: spec for spec in specs}
    plan_rows = {
        str(row.get("relative_run_dir")): row for row in plan.get("runs", [])
    }
    if set(plan_rows) != set(expected_by_dir):
        raise ValueError("execution plan run directories do not match protocol")
    if any(row.get("status") != "completed" for row in plan_rows.values()):
        raise ValueError("execution plan contains a non-completed run")
    study_rows = {str(row.get("path")): row for row in study.get("runs", [])}
    if set(study_rows) != set(expected_by_dir):
        raise ValueError("study manifest run directories do not match protocol")
    if any(row.get("status") != "completed" for row in study_rows.values()):
        raise ValueError("study manifest contains a non-completed run")
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
    module_path = REPO_ROOT / "analysis" / "candidate_validation_analysis.py"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "stage": "stage7_candidate_validation_analysis",
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

        data = load_candidate_validation(
            experiment_root,
            expected_specs=specs,
            numeric_tolerance=tolerance,
        )
        condition_seed = build_seed_summary(data.iterations)
        condition_summary = build_condition_summary(
            data.iterations,
            repetitions=repetitions,
            seed=bootstrap_seed,
        )
        block_performance = build_candidate_block_performance(data.iterations)
        effects_none = build_candidate_effects(
            data.iterations,
            reference_id="none",
            repetitions=repetitions,
            seed=bootstrap_seed + 1000,
        )
        effects_legacy = build_candidate_effects(
            data.iterations,
            reference_id="legacy_balance",
            repetitions=repetitions,
            seed=bootstrap_seed + 2000,
        )
        effects_prior = build_candidate_effects(
            data.iterations,
            reference_id="prior_high",
            repetitions=repetitions,
            seed=bootstrap_seed + 3000,
        )
        exploration_validation = build_exploration_validation_comparison(
            block_performance
        )
        selection_rule = protocol["selection_rule"]
        none_rule = selection_rule["eligibility"]["none_reference"]
        legacy_rule = selection_rule["eligibility"]["legacy_balance_reference"]
        if none_rule["minimum_positive_seed_blocks"] != legacy_rule[
            "minimum_positive_seed_blocks"
        ]:
            raise ValueError(
                "Stage 7 implementation requires the same block count for both "
                "eligibility references"
            )
        ranking, clusters, selected, decision = build_candidate_selection(
            block_performance,
            effects_none,
            effects_legacy,
            effects_prior,
            maximum_cluster_distance=float(
                selection_rule["region_grouping"][
                    "maximum_complete_linkage_distance"
                ]
            ),
            maximum_regions_per_network=int(
                selection_rule["shortlist"]["maximum_regions_per_network"]
            ),
            minimum_positive_seed_blocks=int(
                none_rule["minimum_positive_seed_blocks"]
            ),
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
            block_performance,
            tables_root / "candidate_block_performance.csv",
            outputs,
            "candidate_block_performance",
        )
        _write_csv(
            effects_none,
            tables_root / "candidate_effects_vs_none.csv",
            outputs,
            "candidate_effects_vs_none",
        )
        _write_csv(
            effects_legacy,
            tables_root / "candidate_effects_vs_legacy_balance.csv",
            outputs,
            "candidate_effects_vs_legacy_balance",
        )
        _write_csv(
            effects_prior,
            tables_root / "candidate_effects_vs_prior_high.csv",
            outputs,
            "candidate_effects_vs_prior_high",
        )
        _write_csv(
            exploration_validation,
            tables_root / "exploration_validation_comparison.csv",
            outputs,
            "exploration_validation_comparison",
        )
        _write_csv(
            clusters,
            tables_root / "candidate_clusters.csv",
            outputs,
            "candidate_clusters",
        )
        _write_csv(
            ranking,
            tables_root / "candidate_ranking.csv",
            outputs,
            "candidate_ranking",
        )
        _write_csv(
            selected,
            tables_root / "selected_candidates.csv",
            outputs,
            "selected_candidates",
        )

        decision_path = analysis_root / "decision.json"
        write_json(decision_path, decision)
        outputs["decision"] = _output_entry(decision_path)
        summary = {
            "status": "completed",
            "experiment_id": str(plan["experiment_id"]),
            "executed_git_commit": plan.get("git", {}).get("commit"),
            "run_count": int(len(data.runs)),
            "iteration_metric_count": int(len(data.iterations)),
            "valid_run_count": int(data.audit["valid"].sum()),
            "candidate_count": int(
                data.runs["condition_role"].eq("stage6_candidate").groupby(
                    data.runs["network"]
                ).sum().sum()
                / len(protocol["design"]["simulator_seeds"])
            ),
            "candidate_block_count": int(len(block_performance)),
            "qualified_candidate_count": int(ranking["qualified"].sum()),
            "selected_candidate_count": int(len(selected)),
            "execution_plan_counts": plan["counts"],
            "study_manifest_run_count": int(len(study.get("runs", []))),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": bootstrap_seed,
            "numeric_tolerance": tolerance,
            "reserved_final_test_seeds_used": False,
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
    print(f"Completed Stage 7 candidate validation analysis: {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
