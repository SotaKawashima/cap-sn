#!/usr/bin/env python3
"""Audit a completed Stage 6 reoptimization and build its candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.reoptimization_analysis import (
    build_best_so_far,
    build_candidate_pool,
    build_convergence_summary,
    build_method_summary,
    build_run_final_summary,
    build_stage6_decision,
    build_timing_summary,
    build_unique_candidate_pool,
    load_reoptimization,
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
from run_stage6_reoptimization import (
    DEFAULT_PROTOCOL_PATH,
    STAGE,
    build_run_specs,
    load_protocol,
)


DEFAULT_ANALYSIS_ID = "reoptimization_analysis_v01"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_source(
    experiment_root: Path,
    protocol_path: Path,
    specs: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = experiment_root / "reoptimization_execution_plan.json"
    study_path = experiment_root / "study_manifest.json"
    if not plan_path.is_file() or not study_path.is_file():
        raise FileNotFoundError(
            "Stage 6 execution plan or study manifest is missing"
        )
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != STAGE:
        raise ValueError("execution plan has an unexpected stage")
    expected_counts = {
        "total": len(specs),
        "completed": len(specs),
        "pending": 0,
        "other": 0,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError(
            f"Stage 6 execution plan is incomplete: {plan.get('counts')!r}"
        )
    if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("local Stage 6 protocol does not match the execution")

    expected_by_dir = {spec.relative_run_dir: spec for spec in specs}
    plan_rows = {
        str(row.get("relative_run_dir")): row for row in plan.get("runs", [])
    }
    if set(plan_rows) != set(expected_by_dir):
        raise ValueError("execution plan run directories do not match the protocol")
    if any(row.get("status") != "completed" for row in plan_rows.values()):
        raise ValueError("execution plan contains a non-completed run")

    study_rows = {str(row.get("path")): row for row in study.get("runs", [])}
    if set(study_rows) != set(expected_by_dir):
        raise ValueError("study manifest run directories do not match the protocol")
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
    module_path = REPO_ROOT / "analysis" / "reoptimization_analysis.py"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "stage": "stage6_reoptimization_analysis",
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
        specs = build_run_specs(protocol)
        plan, study = _validate_source(
            experiment_root,
            protocol_path,
            specs,
        )
        tolerance = float(
            protocol["analysis_plan"]["quality_gate"]["numeric_tolerance"]
        )
        data = load_reoptimization(
            experiment_root,
            expected_specs=specs,
            numeric_tolerance=tolerance,
        )
        best_so_far = build_best_so_far(data.trials)
        convergence = build_convergence_summary(best_so_far)
        run_final = build_run_final_summary(data.trials, data.runs)
        method_summary = build_method_summary(run_final)
        timing_summary = build_timing_summary(data.runs)
        candidate_pool = build_candidate_pool(run_final)
        unique_candidate_pool = build_unique_candidate_pool(candidate_pool)
        decision = build_stage6_decision(
            run_final,
            candidate_pool,
            unique_candidate_pool,
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
            data.trials,
            tables_root / "trial_inventory.parquet",
            outputs,
            "trial_inventory",
        )
        _write_parquet(
            data.iterations,
            tables_root / "iteration_metrics.parquet",
            outputs,
            "iteration_metrics",
        )
        _write_parquet(
            best_so_far,
            tables_root / "best_so_far.parquet",
            outputs,
            "best_so_far",
        )
        _write_csv(
            convergence,
            tables_root / "convergence_summary.csv",
            outputs,
            "convergence_summary",
        )
        _write_csv(
            run_final,
            tables_root / "run_final_summary.csv",
            outputs,
            "run_final_summary",
        )
        _write_csv(
            method_summary,
            tables_root / "method_summary.csv",
            outputs,
            "method_summary",
        )
        _write_csv(
            timing_summary,
            tables_root / "timing_summary.csv",
            outputs,
            "timing_summary",
        )
        _write_csv(
            candidate_pool,
            tables_root / "candidate_pool.csv",
            outputs,
            "candidate_pool",
        )
        _write_csv(
            unique_candidate_pool,
            tables_root / "unique_candidate_pool.csv",
            outputs,
            "unique_candidate_pool",
        )

        decision_path = analysis_root / "decision.json"
        write_json(decision_path, decision)
        outputs["decision"] = _output_entry(decision_path)

        summary = {
            "status": "completed",
            "experiment_id": str(plan["experiment_id"]),
            "executed_git_commit": plan.get("git", {}).get("commit"),
            "run_count": int(len(data.runs)),
            "trial_count": int(len(data.trials)),
            "iteration_metric_count": int(len(data.iterations)),
            "valid_run_count": int(data.audit["valid"].sum()),
            "network_count": int(data.runs["network"].nunique()),
            "method_count": int(data.runs["method"].nunique()),
            "optimizer_replicates_per_network_method": int(
                data.runs.groupby(["network", "method"])["run_key"].count().min()
            ),
            "candidate_count": int(len(candidate_pool)),
            "unique_candidate_count": int(len(unique_candidate_pool)),
            "execution_plan_counts": plan["counts"],
            "study_manifest_run_count": int(len(study.get("runs", []))),
            "numeric_tolerance": tolerance,
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
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Completed Stage 6 reoptimization analysis: {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
