#!/usr/bin/env python3
"""Combine Stage 7 data with simple_max and freeze an amended shortlist."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from analysis.candidate_reselection_analysis import (
    ALL_REFERENCES,
    build_candidate_block_performance,
    build_candidate_selection,
    combine_validation_data,
)
from analysis.candidate_validation_analysis import (
    build_candidate_effects,
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
from run_stage7_candidate_reselection import (
    DEFAULT_PROTOCOL_PATH,
    EXECUTION_PLAN_NAME,
    STAGE,
    build_specs,
    load_protocol,
)
from run_stage7_candidate_validation import (
    EXECUTION_PLAN_NAME as SOURCE_EXECUTION_PLAN_NAME,
)
from run_stage7_candidate_validation import (
    STAGE as SOURCE_STAGE,
)
from run_stage7_candidate_validation import (
    build_specs as build_source_specs,
)
from run_stage7_candidate_validation import (
    load_protocol as load_source_protocol,
)


DEFAULT_ANALYSIS_ID = "candidate_reselection_analysis_v01"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_execution(
    experiment_root: Path,
    *,
    execution_plan_name: str,
    expected_stage: str,
    expected_specs: Sequence[Any],
    protocol_path: Path,
    candidate_source_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = experiment_root / execution_plan_name
    study_path = experiment_root / "study_manifest.json"
    if not plan_path.is_file() or not study_path.is_file():
        raise FileNotFoundError(
            f"execution plan or study manifest is missing: {experiment_root}"
        )
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != expected_stage or study.get("stage") != expected_stage:
        raise ValueError(f"unexpected stage in {experiment_root}")
    expected_counts = {
        "total": len(expected_specs),
        "completed": len(expected_specs),
        "pending": 0,
        "other": 0,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError(
            f"incomplete execution plan at {experiment_root}: "
            f"{plan.get('counts')!r}"
        )
    if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError(f"protocol hash mismatch at {experiment_root}")
    if candidate_source_path is not None:
        observed_hash = plan.get("candidate_source", {}).get("sha256")
        if observed_hash != sha256_file(candidate_source_path):
            raise ValueError(
                f"candidate source hash mismatch at {experiment_root}"
            )

    expected_dirs = {spec.relative_run_dir for spec in expected_specs}
    plan_rows = {
        str(row.get("relative_run_dir")): row for row in plan.get("runs", [])
    }
    study_rows = {str(row.get("path")): row for row in study.get("runs", [])}
    if set(plan_rows) != expected_dirs or set(study_rows) != expected_dirs:
        raise ValueError(f"run directories do not match at {experiment_root}")
    if any(row.get("status") != "completed" for row in plan_rows.values()):
        raise ValueError(f"execution plan has incomplete runs: {experiment_root}")
    if any(row.get("status") != "completed" for row in study_rows.values()):
        raise ValueError(f"study manifest has incomplete runs: {experiment_root}")
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
    supplemental_root = args.experiment_root.resolve()
    protocol_path = args.protocol.resolve()
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    analysis_root = supplemental_root / analysis_id
    if analysis_root.exists():
        raise FileExistsError(f"analysis directory already exists: {analysis_root}")
    tables_root = analysis_root / "tables"
    figures_root = analysis_root / "figures"
    tables_root.mkdir(parents=True)
    figures_root.mkdir()

    started = time.perf_counter()
    manifest_path = analysis_root / "analysis_manifest.json"
    module_path = REPO_ROOT / "analysis" / "candidate_reselection_analysis.py"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "stage": "stage7_candidate_reselection_analysis",
        "status": "running",
        "started_at": now_iso(),
        "supplemental_experiment_root": relative_to_repo(supplemental_root),
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
        source_config = protocol["source_validation"]
        source_root = (REPO_ROOT / source_config["experiment_root"]).resolve()
        source_protocol_path = (
            REPO_ROOT / source_config["protocol"]["path"]
        ).resolve()
        source_candidate_path = (
            REPO_ROOT / source_config["candidate_source"]["path"]
        ).resolve()
        if sha256_file(source_protocol_path) != source_config["protocol"]["sha256"]:
            raise ValueError("source Stage 7 protocol hash changed")
        if (
            sha256_file(source_candidate_path)
            != source_config["candidate_source"]["sha256"]
        ):
            raise ValueError("source Stage 7 candidate hash changed")

        source_protocol = load_source_protocol(source_protocol_path)
        source_specs = build_source_specs(source_protocol)
        supplemental_specs = build_specs(protocol)
        source_plan, source_study = _validate_execution(
            source_root,
            execution_plan_name=SOURCE_EXECUTION_PLAN_NAME,
            expected_stage=SOURCE_STAGE,
            expected_specs=source_specs,
            protocol_path=source_protocol_path,
            candidate_source_path=source_candidate_path,
        )
        supplemental_plan, supplemental_study = _validate_execution(
            supplemental_root,
            execution_plan_name=EXECUTION_PLAN_NAME,
            expected_stage=STAGE,
            expected_specs=supplemental_specs,
            protocol_path=protocol_path,
        )

        quality = protocol["analysis_plan"]["quality_gate"]
        tolerance = float(quality["numeric_tolerance"])
        source_data = load_candidate_validation(
            source_root,
            expected_specs=source_specs,
            numeric_tolerance=tolerance,
            expected_stage=SOURCE_STAGE,
        )
        supplemental_data = load_candidate_validation(
            supplemental_root,
            expected_specs=supplemental_specs,
            numeric_tolerance=tolerance,
            expected_stage=STAGE,
        )
        combined = combine_validation_data(
            source_data,
            supplemental_data,
            expected_source_runs=int(quality["required_source_runs"]),
            expected_supplemental_runs=int(
                quality["required_supplemental_runs"]
            ),
            expected_iterations_per_run=int(
                quality["required_iterations_per_run"]
            ),
        )
        if len(combined.runs) != int(quality["required_combined_runs"]):
            raise ValueError("combined run count does not match the protocol")

        bootstrap = protocol["inference"]["bootstrap"]
        repetitions = int(bootstrap["repetitions"])
        bootstrap_seed = int(bootstrap["seed"])
        condition_seed = build_seed_summary(combined.iterations)
        condition_summary = build_condition_summary(
            combined.iterations,
            repetitions=repetitions,
            seed=bootstrap_seed,
        )
        block_performance = build_candidate_block_performance(
            combined.iterations
        )
        effects_by_reference: dict[str, pd.DataFrame] = {}
        for offset, reference in enumerate(ALL_REFERENCES, start=1):
            effects_by_reference[reference] = build_candidate_effects(
                combined.iterations,
                reference_id=reference,
                repetitions=repetitions,
                seed=bootstrap_seed + offset * 1000,
            )
        exploration_validation = build_exploration_validation_comparison(
            block_performance
        )
        selection_rule = protocol["selection_rule"]
        ranking, clusters, selected, decision = build_candidate_selection(
            block_performance,
            effects_by_reference,
            maximum_cluster_distance=float(
                selection_rule["region_grouping"][
                    "maximum_complete_linkage_distance"
                ]
            ),
            target_candidates_per_network=int(
                selection_rule["shortlist"]["target_candidates_per_network"]
            ),
            minimum_positive_seed_blocks=int(
                selection_rule["eligibility"][
                    "minimum_positive_seed_blocks_for_each_required_reference"
                ]
            ),
            reserved_revised_final_test_seeds=protocol["seed_policy"][
                "reserved_revised_final_test"
            ],
        )
        reserved = list(protocol["seed_policy"]["reserved_revised_final_test"])
        if decision["reserved_revised_final_test_seeds"] != reserved:
            raise ValueError("analysis and protocol reserved seeds do not match")

        outputs: dict[str, dict[str, Any]] = {}
        _write_csv(
            source_data.audit,
            tables_root / "source_data_audit.csv",
            outputs,
            "source_data_audit",
        )
        _write_csv(
            supplemental_data.audit,
            tables_root / "supplemental_data_audit.csv",
            outputs,
            "supplemental_data_audit",
        )
        _write_csv(
            combined.audit,
            tables_root / "combined_data_audit.csv",
            outputs,
            "combined_data_audit",
        )
        _write_csv(
            combined.runs,
            tables_root / "combined_run_inventory.csv",
            outputs,
            "combined_run_inventory",
        )
        _write_parquet(
            combined.iterations,
            tables_root / "combined_iteration_metrics.parquet",
            outputs,
            "combined_iteration_metrics",
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
        for reference, frame in effects_by_reference.items():
            _write_csv(
                frame,
                tables_root / f"candidate_effects_vs_{reference}.csv",
                outputs,
                f"candidate_effects_vs_{reference}",
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
            "source_experiment_id": source_plan["experiment_id"],
            "supplemental_experiment_id": supplemental_plan["experiment_id"],
            "source_run_count": len(source_data.runs),
            "supplemental_run_count": len(supplemental_data.runs),
            "combined_run_count": len(combined.runs),
            "combined_iteration_count": len(combined.iterations),
            "candidate_count": int(
                ranking[["network", "condition_id"]].drop_duplicates().shape[0]
            ),
            "qualified_candidate_count": int(ranking["qualified"].sum()),
            "selected_candidate_count": len(selected),
            "source_study_manifest_run_count": len(source_study.get("runs", [])),
            "supplemental_study_manifest_run_count": len(
                supplemental_study.get("runs", [])
            ),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": bootstrap_seed,
            "numeric_tolerance": tolerance,
            "revised_final_test_seeds_used": False,
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
                "source_experiments": {
                    "original_stage7": {
                        "path": relative_to_repo(source_root),
                        "experiment_id": source_plan["experiment_id"],
                        "git_commit": source_plan.get("git", {}).get("commit"),
                        "run_count": len(source_specs),
                    },
                    "simple_max_supplement": {
                        "path": relative_to_repo(supplemental_root),
                        "experiment_id": supplemental_plan["experiment_id"],
                        "git_commit": supplemental_plan.get("git", {}).get(
                            "commit"
                        ),
                        "run_count": len(supplemental_specs),
                    },
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
    print(f"Completed amended candidate reselection: {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
