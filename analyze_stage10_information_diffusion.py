"""Run the formal Stage 10 information-diffusion analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.information_diffusion_analysis import (
    build_behavior_time_series,
    build_correlation_tables,
    build_decision,
    build_direction_consistency,
    build_exploratory_fixed_differences,
    build_fixed_condition_summary,
    build_fixed_run_summary,
    build_information_time_series,
    build_primary_paired_effects,
    build_ratio_diagnostics,
    combine_fixed_raw,
    load_existing_information_data,
    load_selected_fixed_raw,
    load_stage10_raw,
)
from experiment_runtime import (
    REPO_ROOT,
    create_unique_run_directory,
    git_state,
    now_iso,
    sha256_file,
    software_versions,
    validate_safe_name,
    write_json,
)
from run_stage10_information_diffusion import (
    DEFAULT_PROTOCOL_PATH,
    STAGE,
    load_protocol,
)


DEFAULT_ANALYSIS_ID = "information_diffusion_analysis_v01"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Stage 10 information diffusion without reselection."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    return parser.parse_args(argv)


def _write_table(
    table: pd.DataFrame,
    path: Path,
    *,
    parquet: bool = False,
) -> dict[str, Any]:
    if parquet:
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)
    return {
        "path": path.relative_to(path.parent.parent).as_posix(),
        "rows": int(len(table)),
        "sha256": sha256_file(path),
    }


def run_analysis(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    experiment_root = args.experiment_root.resolve()
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    analysis_root = experiment_root / analysis_id
    create_unique_run_directory(analysis_root)
    tables_root = analysis_root / "tables"
    tables_root.mkdir()
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "stage": f"{STAGE}_analysis",
        "status": "running",
        "started_at": now_iso(),
        "source_experiment_root": experiment_root.as_posix(),
        "protocol": {
            "path": protocol_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "analysis_code": {
            "cli": {
                "path": "analyze_stage10_information_diffusion.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "module": {
                "path": "analysis/information_diffusion_analysis.py",
                "sha256": sha256_file(
                    REPO_ROOT / "analysis" / "information_diffusion_analysis.py"
                ),
            },
        },
        "git": git_state(),
        "software": software_versions(),
        "outputs": {},
        "failure": None,
    }
    write_json(manifest_path, manifest)
    try:
        existing = load_existing_information_data(protocol)
        stage4_raw = load_selected_fixed_raw(protocol, existing)
        stage10_raw = load_stage10_raw(
            experiment_root,
            protocol,
            expected_protocol_sha256=sha256_file(protocol_path),
        )
        fixed = combine_fixed_raw(stage4_raw, stage10_raw)

        correlations = build_correlation_tables(existing)
        primary_indicators = protocol["analysis_plan"][
            "primary_information_indicators"
        ]
        bootstrap = protocol["analysis_plan"]["bootstrap"]
        fixed_run_summary = build_fixed_run_summary(fixed.iterations)
        fixed_condition_summary = build_fixed_condition_summary(fixed_run_summary)
        primary_effects = build_primary_paired_effects(
            fixed.iterations,
            primary_indicators=primary_indicators,
            repetitions=int(bootstrap["repetitions"]),
            seed=int(bootstrap["seed"]),
        )
        exploratory_effects = build_exploratory_fixed_differences(
            fixed_run_summary,
            primary_indicators=primary_indicators,
        )
        behavior_time_series = build_behavior_time_series(fixed.pop_events)
        information_time_series = build_information_time_series(fixed.info_events)
        direction_consistency = build_direction_consistency(
            correlations, primary_indicators
        )

        stage2_random = existing.stage2_trials[
            existing.stage2_trials["method"] == "random"
        ].copy()
        stage2_random["subset"] = "random_search_primary"
        stage2_all = existing.stage2_trials.copy()
        stage2_all["subset"] = "all_methods_sensitivity"
        stage2_ratio = pd.concat(
            [
                build_ratio_diagnostics(
                    stage2_random,
                    source="stage2_saved_trials",
                    grouping_columns=["subset", "network"],
                ),
                build_ratio_diagnostics(
                    stage2_all,
                    source="stage2_saved_trials",
                    grouping_columns=["subset", "network"],
                ),
            ],
            ignore_index=True,
        )
        fixed_ratio = build_ratio_diagnostics(
            fixed.iterations,
            source="fixed_paired_conditions",
            grouping_columns=["network", "condition_group"],
        )
        ratio_diagnostics = pd.concat(
            [stage2_ratio, fixed_ratio], ignore_index=True, sort=False
        )
        data_audit = pd.concat(
            [existing.audit, stage4_raw.audit, stage10_raw.audit],
            ignore_index=True,
            sort=False,
        )
        decision = build_decision(
            primary_effects,
            direction_consistency,
            stage10_raw.runs,
        )

        outputs: dict[str, dict[str, Any]] = {}
        csv_tables = {
            "data_audit": data_audit,
            "stage10_run_inventory": stage10_raw.runs,
            "fixed_run_summary": fixed_run_summary,
            "fixed_condition_summary": fixed_condition_summary,
            "design_information_correlations": correlations[
                "design_information_correlations"
            ],
            "outcome_information_correlations": correlations[
                "outcome_information_correlations"
            ],
            "partial_correlations": correlations["partial_correlations"],
            "direction_consistency": direction_consistency,
            "ratio_diagnostics": ratio_diagnostics,
            "primary_paired_effects": primary_effects,
            "exploratory_fixed_differences": exploratory_effects,
        }
        for name, table in csv_tables.items():
            outputs[name] = _write_table(table, tables_root / f"{name}.csv")
        parquet_tables = {
            "fixed_iteration_metrics": fixed.iterations,
            "behavior_time_series": behavior_time_series,
            "information_time_series": information_time_series,
        }
        for name, table in parquet_tables.items():
            outputs[name] = _write_table(
                table,
                tables_root / f"{name}.parquet",
                parquet=True,
            )

        decision_path = analysis_root / "decision.json"
        write_json(decision_path, decision)
        outputs["decision"] = {
            "path": "decision.json",
            "sha256": sha256_file(decision_path),
        }
        summary = {
            "status": "completed",
            "stage2_trial_count": int(len(existing.stage2_trials)),
            "stage4_formal_run_count": int(len(existing.stage4_run_inventory)),
            "selected_stage4_run_count": int(len(stage4_raw.runs)),
            "stage10_run_count": int(len(stage10_raw.runs)),
            "combined_fixed_run_count": int(len(fixed.runs)),
            "combined_fixed_iteration_count": int(len(fixed.iterations)),
            "primary_indicator_count": int(len(primary_indicators)),
            "primary_paired_effect_rows": int(len(primary_effects)),
            "candidate_selection_changed": False,
            "causal_claim_supported": False,
        }
        summary_path = analysis_root / "analysis_summary.json"
        write_json(summary_path, summary)
        outputs["analysis_summary"] = {
            "path": "analysis_summary.json",
            "sha256": sha256_file(summary_path),
        }
        manifest["outputs"] = outputs
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        manifest["completed_at"] = now_iso()
        manifest["elapsed_sec"] = time.perf_counter() - started
        write_json(manifest_path, manifest)
    return analysis_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = run_analysis(args)
    except (FileExistsError, FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Completed Stage 10 information-diffusion analysis: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
