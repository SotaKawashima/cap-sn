"""Create reproducible analysis tables for a completed Stage 3 pilot phase."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.pilot_analysis import (
    assess_optimization_budget,
    build_best_so_far,
    build_condition_precision,
    build_paired_differences,
    build_prefix_estimates,
    build_rank_stability,
    build_resource_summary,
    load_optimization_pilot,
    load_precision_pilot,
    project_future_resources,
    recommend_iteration_count,
)
from experiment_runtime import (
    REPO_ROOT,
    create_unique_run_directory,
    git_state,
    now_iso,
    sha256_file,
    validate_safe_name,
    write_json,
)
from run_stage3_pilot import (
    DEFAULT_PROTOCOL_PATH,
    build_optimization_specs,
    build_precision_specs,
    load_protocol,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(frame: pd.DataFrame, output_dir: Path, name: str) -> str:
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path.name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed Stage 3 pilot.")
    parser.add_argument(
        "--phase",
        choices=["precision", "optimization_budget"],
        required=True,
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--analysis-id", default="analysis_v01")
    parser.add_argument("--delta-min", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--optimization-evaluations", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    return parser.parse_args(argv)


def analyze(args: argparse.Namespace) -> Path:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    experiment_root = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        raise ValueError(f"pilot experiment root does not exist: {experiment_root}")
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    output_dir = experiment_root / analysis_id
    create_unique_run_directory(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage3_pilot",
        "phase": args.phase,
        "analysis_id": analysis_id,
        "created_at": now_iso(),
        "status": "running",
        "git": git_state(),
        "protocol": {
            "path": protocol_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "pilot_experiment_root": experiment_root.as_posix(),
        "delta_min": args.delta_min,
        "outputs": {},
        "decision": None,
        "failure": None,
    }
    write_json(output_dir / "analysis_manifest.json", manifest)

    try:
        if args.phase == "precision":
            phase = protocol["precision_phase"]
            specs = build_precision_specs(protocol)
            data = load_precision_pilot(experiment_root, expected_specs=specs)
            prefixes = [int(value) for value in phase["prefix_iterations"]]
            repetitions = int(
                args.bootstrap_repetitions
                if args.bootstrap_repetitions is not None
                else phase["bootstrap"]["repetitions"]
            )
            if repetitions <= 0:
                raise ValueError("bootstrap repetitions must be positive")

            prefix_estimates = build_prefix_estimates(
                data.iterations, prefixes=prefixes
            )
            condition_precision = build_condition_precision(
                data.iterations,
                prefixes=prefixes,
                repetitions=repetitions,
                seed=int(phase["bootstrap"]["seed"]),
            )
            paired_differences = build_paired_differences(
                data.iterations,
                prefixes=prefixes,
                comparisons=phase["primary_comparisons"],
                repetitions=repetitions,
                seed=int(phase["bootstrap"]["seed"]),
            )
            rank_detail, rank_summary = build_rank_stability(
                data.iterations, prefixes=prefixes
            )
            resources = build_resource_summary(data.runs)
            decision_rule = phase["decision_rule"]
            decision = recommend_iteration_count(
                paired_differences,
                rank_summary,
                prefixes=prefixes,
                delta_min=args.delta_min,
                ci_fraction=float(
                    decision_rule["ci_half_width_fraction_of_delta_min"]
                ),
                minimum_median_rank_spearman=float(
                    decision_rule["minimum_median_rank_spearman"]
                ),
                minimum_best_condition_agreement=float(
                    decision_rule["minimum_best_condition_agreement"]
                ),
            )

            outputs = {
                "run_inventory": _write_csv(data.runs, tables_dir, "run_inventory"),
                "prefix_estimates": _write_csv(
                    prefix_estimates, tables_dir, "prefix_estimates"
                ),
                "condition_precision": _write_csv(
                    condition_precision, tables_dir, "condition_precision"
                ),
                "paired_differences": _write_csv(
                    paired_differences, tables_dir, "paired_differences"
                ),
                "rank_stability_detail": _write_csv(
                    rank_detail, tables_dir, "rank_stability_detail"
                ),
                "rank_stability_summary": _write_csv(
                    rank_summary, tables_dir, "rank_stability_summary"
                ),
                "resource_summary": _write_csv(
                    resources, tables_dir, "resource_summary"
                ),
            }
            iteration_path = tables_dir / "iteration_metrics.parquet"
            data.iterations.to_parquet(iteration_path, index=False)
            outputs["iteration_metrics"] = iteration_path.name

            recommended = decision.get("recommended_iterations")
            if recommended is not None:
                optimization_evaluations = args.optimization_evaluations
                if (
                    optimization_evaluations is not None
                    and optimization_evaluations <= 0
                ):
                    raise ValueError("optimization evaluations must be positive")
                projection = project_future_resources(
                    resources,
                    iterations=int(recommended),
                    optimization_evaluations=optimization_evaluations,
                )
                outputs["future_resource_projection"] = _write_csv(
                    projection, tables_dir, "future_resource_projection"
                )
            manifest["analysis_settings"] = {
                "prefixes": prefixes,
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": int(phase["bootstrap"]["seed"]),
                "optimization_evaluations": args.optimization_evaluations,
            }
        else:
            if args.iterations is None:
                raise ValueError("optimization analysis requires --iterations")
            phase = protocol["optimization_budget_phase"]
            specs = build_optimization_specs(protocol, iterations=args.iterations)
            data = load_optimization_pilot(experiment_root, expected_specs=specs)
            checkpoints = [int(value) for value in phase["checkpoints"]]
            best_so_far, checkpoint_table, improvement_summary = build_best_so_far(
                data.trials, checkpoints=checkpoints
            )
            decision = assess_optimization_budget(
                checkpoint_table,
                checkpoints=checkpoints,
                delta_min=args.delta_min,
                late_improvement_fraction=float(
                    phase["decision_rule"][
                        "late_improvement_fraction_of_delta_min"
                    ]
                ),
            )
            outputs = {
                "run_inventory": _write_csv(data.runs, tables_dir, "run_inventory"),
                "best_so_far": _write_csv(best_so_far, tables_dir, "best_so_far"),
                "checkpoint_values": _write_csv(
                    checkpoint_table, tables_dir, "checkpoint_values"
                ),
                "improvement_summary": _write_csv(
                    improvement_summary, tables_dir, "improvement_summary"
                ),
            }
            manifest["analysis_settings"] = {
                "iterations": int(args.iterations),
                "checkpoints": checkpoints,
            }

        decision_path = output_dir / "decision.json"
        write_json(decision_path, _json_value(decision))
        manifest["outputs"] = {
            key: f"tables/{filename}" for key, filename in outputs.items()
        }
        manifest["outputs"]["decision"] = decision_path.name
        manifest["decision"] = decision
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        manifest["updated_at"] = now_iso()
        write_json(output_dir / "analysis_manifest.json", _json_value(manifest))
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_dir = analyze(args)
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Completed Stage 3 pilot analysis: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
