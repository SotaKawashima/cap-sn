#!/usr/bin/env python3
"""Audit and analyze the Stage 4 fixed confirmation experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.fixed_confirmation_analysis import (
    build_all_vs_none,
    build_comparison_table,
    build_condition_summary,
    build_factorial_tables,
    build_information_summary,
    build_seed_summary,
    build_stage4_decision,
    build_time_series_summary,
    load_fixed_confirmation,
    replace_fixed_confirmation_condition,
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
from run_stage4_fixed_confirmation import (
    DEFAULT_PROTOCOL_PATH,
    build_specs,
    load_protocol,
)


DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT
    / "experiments"
    / "summer_2026"
    / "stage4_fixed_confirmation"
    / "20260817_112313_fixed_confirmation_v01"
)
DEFAULT_ANALYSIS_ID = "fixed_confirmation_analysis_v01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    parser.add_argument("--condition-override-id", default=None)
    parser.add_argument("--condition-override-root", type=Path, default=None)
    parser.add_argument("--condition-override-protocol", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_source(
    experiment_root: Path,
    protocol_path: Path,
    expected_run_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = experiment_root / "fixed_confirmation_execution_plan.json"
    study_path = experiment_root / "study_manifest.json"
    if not plan_path.is_file() or not study_path.is_file():
        raise FileNotFoundError("Stage 4 execution plan or study manifest is missing")
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != "stage4_fixed_confirmation":
        raise ValueError("execution plan has an unexpected stage")
    expected_counts = {
        "total": expected_run_count,
        "completed": expected_run_count,
        "pending": 0,
        "other": 0,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError(
            f"execution plan is incomplete: {plan.get('counts')!r}"
        )
    if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("local protocol does not match the executed protocol")
    if len(plan.get("runs", [])) != expected_run_count:
        raise ValueError("execution plan run count does not match the protocol")
    return plan, study


def _validate_override_source(
    experiment_root: Path,
    protocol_path: Path,
    expected_specs: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = experiment_root / "fixed_confirmation_execution_plan.json"
    study_path = experiment_root / "study_manifest.json"
    if not plan_path.is_file() or not study_path.is_file():
        raise FileNotFoundError(
            "condition override execution plan or study manifest is missing"
        )
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != "stage4_fixed_confirmation":
        raise ValueError(
            "condition override is not a Stage 4 fixed confirmation execution"
        )
    if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("condition override protocol does not match its execution")
    statuses = {
        str(row.get("relative_run_dir")): str(row.get("status"))
        for row in plan.get("runs", [])
    }
    missing = [
        spec.relative_run_dir
        for spec in expected_specs
        if statuses.get(spec.relative_run_dir) != "completed"
    ]
    if missing:
        raise ValueError(
            "condition override does not contain every completed target run: "
            f"{missing[:5]}"
        )
    return plan, study


def _table_entry(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": f"{path.parent.name}/{path.name}",
        "rows": int(rows),
        "sha256": sha256_file(path),
    }


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    outputs: dict[str, dict[str, Any]],
    key: str,
) -> None:
    frame.to_csv(path, index=False)
    outputs[key] = _table_entry(path, len(frame))


def _write_parquet(
    frame: pd.DataFrame,
    path: Path,
    outputs: dict[str, dict[str, Any]],
    key: str,
) -> None:
    frame.to_parquet(path, index=False)
    outputs[key] = _table_entry(path, len(frame))


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    protocol_path = args.protocol.resolve()
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    analysis_root = experiment_root / analysis_id
    if analysis_root.exists():
        raise FileExistsError(
            f"analysis directory already exists: {analysis_root}"
        )
    tables_root = analysis_root / "tables"
    figures_root = analysis_root / "figures"
    tables_root.mkdir(parents=True)
    figures_root.mkdir()

    started_at = now_iso()
    started = time.perf_counter()
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "stage": "stage4_fixed_confirmation_analysis",
        "status": "running",
        "started_at": started_at,
        "source_experiment_root": relative_to_repo(experiment_root),
        "protocol": {
            "path": relative_to_repo(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "condition_override": None,
        "analysis_code": {
            "cli": {
                "path": relative_to_repo(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "module": {
                "path": "analysis/fixed_confirmation_analysis.py",
                "sha256": sha256_file(
                    REPO_ROOT / "analysis" / "fixed_confirmation_analysis.py"
                ),
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
            experiment_root, protocol_path, len(specs)
        )
        inference = protocol["inference"]["bootstrap"]
        repetitions = int(inference["repetitions"])
        bootstrap_seed = int(inference["seed"])

        data = load_fixed_confirmation(
            experiment_root,
            expected_specs=specs,
        )
        override_values = (
            args.condition_override_id,
            args.condition_override_root,
            args.condition_override_protocol,
        )
        if any(value is not None for value in override_values):
            if not all(value is not None for value in override_values):
                raise ValueError(
                    "condition override ID, root, and protocol must be specified together"
                )
            override_id = validate_safe_name(
                args.condition_override_id, "condition_override_id"
            )
            override_root = args.condition_override_root.resolve()
            override_protocol_path = args.condition_override_protocol.resolve()
            override_protocol = load_protocol(override_protocol_path)
            override_specs = [
                spec
                for spec in build_specs(override_protocol)
                if spec.condition_id == override_id
            ]
            if not override_specs:
                raise ValueError(
                    f"condition override is not present in protocol: {override_id}"
                )
            override_plan, override_study = _validate_override_source(
                override_root,
                override_protocol_path,
                override_specs,
            )
            correction = load_fixed_confirmation(
                override_root,
                expected_specs=override_specs,
            )
            data = replace_fixed_confirmation_condition(
                data,
                correction,
                condition_id=override_id,
            )
            manifest["condition_override"] = {
                "condition_id": override_id,
                "experiment_root": relative_to_repo(override_root),
                "protocol": {
                    "path": relative_to_repo(override_protocol_path),
                    "sha256": sha256_file(override_protocol_path),
                },
                "run_count": len(override_specs),
                "execution_plan_sha256": sha256_file(
                    override_root / "fixed_confirmation_execution_plan.json"
                ),
                "study_manifest_sha256": sha256_file(
                    override_root / "study_manifest.json"
                ),
                "execution_plan_experiment_id": override_plan.get("experiment_id"),
                "study_manifest_run_count": len(override_study.get("runs", [])),
            }
        condition_seed = build_seed_summary(data.iterations)
        condition_summary = build_condition_summary(
            data.iterations,
            repetitions=repetitions,
            seed=bootstrap_seed,
        )
        primary = build_comparison_table(
            data.iterations,
            comparisons=protocol["primary_comparisons"],
            family="pre_specified_primary",
            repetitions=repetitions,
            seed=bootstrap_seed + 1000,
        )
        all_vs_none = build_all_vs_none(
            data.iterations,
            protocol,
            repetitions=repetitions,
            seed=bootstrap_seed + 2000,
        )
        factorial, interaction = build_factorial_tables(
            data.iterations,
            protocol,
            repetitions=repetitions,
            seed=bootstrap_seed + 3000,
        )
        information = build_information_summary(data.info_long)
        time_series = build_time_series_summary(data.pop_events)
        decision = build_stage4_decision(
            condition_summary,
            primary,
            all_vs_none,
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
            primary,
            tables_root / "primary_comparisons.csv",
            outputs,
            "primary_comparisons",
        )
        _write_csv(
            all_vs_none,
            tables_root / "all_enabled_vs_none.csv",
            outputs,
            "all_enabled_vs_none",
        )
        _write_csv(
            factorial,
            tables_root / "factorial_contrasts.csv",
            outputs,
            "factorial_contrasts",
        )
        _write_csv(
            interaction,
            tables_root / "factorial_interaction.csv",
            outputs,
            "factorial_interaction",
        )
        _write_csv(
            data.info_runs,
            tables_root / "information_run_summary.csv",
            outputs,
            "information_run_summary",
        )
        _write_csv(
            information,
            tables_root / "information_condition_summary.csv",
            outputs,
            "information_condition_summary",
        )
        _write_parquet(
            time_series,
            tables_root / "cumulative_time_series.parquet",
            outputs,
            "cumulative_time_series",
        )

        decision_path = analysis_root / "decision.json"
        write_json(decision_path, decision)
        outputs["decision"] = {
            "path": decision_path.name,
            "sha256": sha256_file(decision_path),
        }
        summary = {
            "status": "completed",
            "experiment_id": str(plan["experiment_id"]),
            "executed_git_commit": plan.get("git", {}).get("commit"),
            "run_count": int(len(data.runs)),
            "iteration_count": int(len(data.iterations)),
            "valid_run_count": int(data.audit["valid"].sum()),
            "network_count": int(data.runs["network"].nunique()),
            "condition_count_per_network": int(
                data.runs.groupby("network")["condition_id"].nunique().min()
            ),
            "seed_blocks_per_condition": int(
                data.runs.groupby(["network", "condition_id"])[
                    "simulator_seed"
                ].nunique().min()
            ),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": bootstrap_seed,
            "execution_plan_counts": plan["counts"],
            "study_manifest_run_count": len(study.get("runs", [])),
            "decision": decision,
        }
        summary_path = analysis_root / "analysis_summary.json"
        write_json(summary_path, summary)
        outputs["analysis_summary"] = {
            "path": summary_path.name,
            "sha256": sha256_file(summary_path),
        }

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
                "inference": {
                    "method": inference["method"],
                    "repetitions": repetitions,
                    "seed": bootstrap_seed,
                    "confidence_level": inference["confidence_level"],
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

    print(f"Completed Stage 4 fixed confirmation analysis: {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
