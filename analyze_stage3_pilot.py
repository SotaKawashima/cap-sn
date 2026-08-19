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
    assess_optimization_budget_by_tolerance,
    build_best_so_far,
    build_condition_precision,
    build_paired_differences,
    build_prefix_estimates,
    build_rank_stability,
    build_resource_summary,
    build_selection_regret,
    load_optimization_pilot,
    load_precision_analysis_snapshot,
    load_precision_pilot,
    project_future_resources,
    recommend_iteration_count,
    recommend_iteration_count_by_selection_regret,
    replace_precision_condition,
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_precision_override_source(
    experiment_root: Path,
    protocol_path: Path,
    expected_specs: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = experiment_root / "pilot_execution_plan.json"
    study_path = experiment_root / "study_manifest.json"
    if not plan_path.is_file() or not study_path.is_file():
        raise FileNotFoundError(
            "precision condition override execution plan or study manifest "
            "is missing"
        )
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != "stage3_pilot" or plan.get("phase") != "precision":
        raise ValueError("condition override is not a Stage 3 precision execution")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed Stage 3 pilot.")
    parser.add_argument(
        "--phase",
        choices=["precision", "optimization_budget"],
        required=True,
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--source-analysis-root",
        type=Path,
        default=None,
        help=(
            "Completed precision analysis containing iteration_metrics.parquet "
            "and run_inventory.csv; use when raw run directories are elsewhere."
        ),
    )
    parser.add_argument(
        "--condition-override-root",
        type=Path,
        default=None,
        help="Completed fixed-condition runs used to replace one precision condition.",
    )
    parser.add_argument("--condition-override-id", default=None)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--analysis-id", default="analysis_v01")
    parser.add_argument("--delta-min", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--optimization-evaluations", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    return parser.parse_args(argv)


def _resource_projection_settings(
    protocol: dict[str, Any],
    *,
    recommended_iterations: int,
    requested_optimization_evaluations: int | None,
) -> tuple[int | None, dict[str, int], dict[str, Any] | None]:
    """Resolve future-experiment scale from the finalized protocol amendment."""

    plan = protocol.get("finalized_experiment_plan")
    if plan is None:
        if (
            requested_optimization_evaluations is not None
            and requested_optimization_evaluations <= 0
        ):
            raise ValueError("optimization evaluations must be positive")
        return requested_optimization_evaluations, {}, None

    selected_iterations = int(plan["iterations_per_evaluation"])
    if selected_iterations != int(recommended_iterations):
        raise ValueError(
            "finalized experiment-plan iterations do not match the "
            "precision recommendation"
        )
    selected_evaluations = int(plan["optimization_evaluations"])
    if selected_evaluations <= 0:
        raise ValueError("finalized optimization evaluations must be positive")
    if (
        requested_optimization_evaluations is not None
        and int(requested_optimization_evaluations) != selected_evaluations
    ):
        raise ValueError(
            "requested optimization evaluations do not match the finalized "
            "experiment plan"
        )

    stage4 = plan["stage4_fixed_confirmation"]
    stage6 = plan["stage6_optimization"]
    expected_networks = list(protocol["precision_phase"]["networks"])
    if list(stage4["networks"]) != expected_networks:
        raise ValueError(
            "finalized Stage 4 networks must match the precision networks"
        )
    if list(stage6["networks"]) != expected_networks:
        raise ValueError(
            "finalized Stage 6 networks must match the precision networks"
        )
    if stage4["raw_level"] != "all":
        raise ValueError(
            "finalized Stage 4 resource projection requires raw_level=all"
        )
    if stage6["raw_level"] != "pop":
        raise ValueError(
            "finalized Stage 6 resource projection requires raw_level=pop"
        )
    conditions = stage4["conditions"]
    condition_ids = [str(condition["id"]) for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("finalized Stage 4 condition IDs must be unique")
    fixed_seeds = [int(seed) for seed in stage4["simulator_seeds"]]
    if len(fixed_seeds) != len(set(fixed_seeds)):
        raise ValueError("finalized Stage 4 simulator seeds must be unique")

    stage6_simulator_seeds = {int(stage6["simulator_seed"])}
    stage6_simulator_seeds.update(
        int(seed) for seed in stage6["reserve_simulator_seeds"]
    )
    future_seed_groups = {
        "stage4": set(fixed_seeds),
        "stage6": stage6_simulator_seeds,
        "candidate_validation": {
            int(seed)
            for seed in plan["candidate_validation"]["simulator_seeds"]
        },
        "final_test": {
            int(seed) for seed in plan["final_test"]["simulator_seeds"]
        },
    }
    used_seeds = {
        int(seed) for seed in protocol["precision_phase"]["simulator_seeds"]
    }
    used_seeds.add(int(protocol["optimization_budget_phase"]["simulator_seed"]))
    for name, seeds in future_seed_groups.items():
        if used_seeds.intersection(seeds):
            raise ValueError(
                f"finalized simulator seed group overlaps an earlier role: {name}"
            )
        used_seeds.update(seeds)

    methods = [str(method) for method in stage6["methods"]]
    optimizer_seeds = stage6["optimizer_seeds"]
    if set(methods) != set(optimizer_seeds):
        raise ValueError(
            "finalized Stage 6 methods and optimizer seed groups must match"
        )
    replicate_counts = {len(optimizer_seeds[method]) for method in methods}
    if len(replicate_counts) != 1:
        raise ValueError(
            "finalized Stage 6 methods must use the same replicate count"
        )
    optimization_replicates = replicate_counts.pop()
    if optimization_replicates <= 0:
        raise ValueError("finalized optimizer replicate count must be positive")
    replicate_ids = [int(value) for value in stage6["optimizer_replicates"]]
    if (
        len(replicate_ids) != optimization_replicates
        or len(replicate_ids) != len(set(replicate_ids))
    ):
        raise ValueError(
            "finalized optimizer replicate IDs must be unique and match seed counts"
        )
    flattened_optimizer_seeds = [
        int(seed)
        for method in methods
        for seed in optimizer_seeds[method]
    ]
    if len(flattened_optimizer_seeds) != len(set(flattened_optimizer_seeds)):
        raise ValueError("finalized optimizer seeds must be unique")

    settings = {
        "fixed_conditions": len(conditions),
        "fixed_seed_blocks": len(fixed_seeds),
        "optimization_methods": len(methods),
        "optimization_replicates": optimization_replicates,
    }
    metadata = {
        "iterations_per_evaluation": selected_iterations,
        "optimization_evaluations": selected_evaluations,
        **settings,
        "stage4_raw_level": stage4["raw_level"],
        "stage6_raw_level": stage6["raw_level"],
        "stage6_simulator_seed_blocks": 1,
        "candidate_validation_seed_blocks": len(
            future_seed_groups["candidate_validation"]
        ),
        "final_test_seed_blocks": len(future_seed_groups["final_test"]),
    }
    return selected_evaluations, settings, metadata


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
        "source_analysis": None,
        "condition_override": None,
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
            if args.source_analysis_root is None:
                data = load_precision_pilot(experiment_root, expected_specs=specs)
            else:
                source_analysis_root = args.source_analysis_root.resolve()
                data = load_precision_analysis_snapshot(
                    source_analysis_root,
                    expected_specs=specs,
                )
                source_manifest = source_analysis_root / "analysis_manifest.json"
                source_iterations = (
                    source_analysis_root / "tables" / "iteration_metrics.parquet"
                )
                source_runs = source_analysis_root / "tables" / "run_inventory.csv"
                manifest["source_analysis"] = {
                    "path": source_analysis_root.as_posix(),
                    "analysis_manifest_sha256": sha256_file(source_manifest),
                    "iteration_metrics_sha256": sha256_file(source_iterations),
                    "run_inventory_sha256": sha256_file(source_runs),
                }
            override_requested = (
                args.condition_override_root is not None
                or args.condition_override_id is not None
            )
            if override_requested:
                if (
                    args.condition_override_root is None
                    or args.condition_override_id is None
                ):
                    raise ValueError(
                        "--condition-override-root and --condition-override-id "
                        "must be specified together"
                    )
                if args.source_analysis_root is None:
                    raise ValueError(
                        "condition override requires --source-analysis-root"
                    )
                override_id = validate_safe_name(
                    args.condition_override_id, "condition_override_id"
                )
                override_specs = [
                    spec for spec in specs if spec.condition_id == override_id
                ]
                if not override_specs:
                    raise ValueError(
                        f"condition override is not present in protocol: {override_id}"
                    )
                override_root = args.condition_override_root.resolve()
                override_plan, override_study = _validate_precision_override_source(
                    override_root,
                    protocol_path,
                    override_specs,
                )
                correction = load_precision_pilot(
                    override_root, expected_specs=override_specs
                )
                data = replace_precision_condition(
                    data, correction, condition_id=override_id
                )
                study_manifest = override_root / "study_manifest.json"
                manifest["condition_override"] = {
                    "condition_id": override_id,
                    "experiment_root": override_root.as_posix(),
                    "run_count": len(override_specs),
                    "execution_plan_sha256": sha256_file(
                        override_root / "pilot_execution_plan.json"
                    ),
                    "study_manifest_sha256": sha256_file(study_manifest),
                    "execution_plan_experiment_id": override_plan.get(
                        "experiment_id"
                    ),
                    "study_manifest_run_count": len(
                        override_study.get("runs", [])
                    ),
                }
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
            decision_method = decision_rule.get(
                "method", "legacy_delta_min_precision_and_rank"
            )
            selection_regret_detail = None
            selection_regret_summary = None
            if (
                decision_method
                == "leave_one_simulator_seed_block_out_selection_regret"
            ):
                tolerance = float(decision_rule["selection_regret_tolerance"])
                selection_regret_detail, selection_regret_summary = (
                    build_selection_regret(
                        data.iterations,
                        prefixes=prefixes,
                        tolerance=tolerance,
                    )
                )
                decision = recommend_iteration_count_by_selection_regret(
                    selection_regret_summary,
                    prefixes=prefixes,
                    tolerance=tolerance,
                    minimum_block_pass_rate_per_network=float(
                        decision_rule["minimum_block_pass_rate_per_network"]
                    ),
                )
                selected_iterations = decision_rule.get("selected_iterations")
                if (
                    selected_iterations is not None
                    and decision.get("recommended_iterations")
                    != int(selected_iterations)
                ):
                    raise ValueError(
                        "protocol-selected iterations do not match the "
                        "selection-regret recommendation"
                    )
            elif decision_method == "legacy_delta_min_precision_and_rank":
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
            else:
                raise ValueError(
                    f"unsupported precision decision method: {decision_method}"
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
            if selection_regret_detail is not None:
                outputs["selection_regret_detail"] = _write_csv(
                    selection_regret_detail,
                    tables_dir,
                    "selection_regret_detail",
                )
                outputs["selection_regret_summary"] = _write_csv(
                    selection_regret_summary,
                    tables_dir,
                    "selection_regret_summary",
                )
            iteration_path = tables_dir / "iteration_metrics.parquet"
            data.iterations.to_parquet(iteration_path, index=False)
            outputs["iteration_metrics"] = iteration_path.name

            recommended = decision.get("recommended_iterations")
            projection_metadata = None
            if recommended is not None:
                (
                    optimization_evaluations,
                    projection_settings,
                    projection_metadata,
                ) = _resource_projection_settings(
                    protocol,
                    recommended_iterations=int(recommended),
                    requested_optimization_evaluations=(
                        args.optimization_evaluations
                    ),
                )
                projection = project_future_resources(
                    resources,
                    iterations=int(recommended),
                    optimization_evaluations=optimization_evaluations,
                    **projection_settings,
                )
                outputs["future_resource_projection"] = _write_csv(
                    projection, tables_dir, "future_resource_projection"
                )
            manifest["analysis_settings"] = {
                "prefixes": prefixes,
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": int(phase["bootstrap"]["seed"]),
                "optimization_evaluations": (
                    projection_metadata["optimization_evaluations"]
                    if projection_metadata is not None
                    else args.optimization_evaluations
                ),
                "resource_projection": projection_metadata,
                "decision_method": decision_method,
            }
        else:
            if (
                args.source_analysis_root is not None
                or args.condition_override_root is not None
                or args.condition_override_id is not None
            ):
                raise ValueError(
                    "source and condition overrides are only supported for "
                    "precision analysis"
                )
            if args.iterations is None:
                raise ValueError("optimization analysis requires --iterations")
            phase = protocol["optimization_budget_phase"]
            specs = build_optimization_specs(protocol, iterations=args.iterations)
            data = load_optimization_pilot(experiment_root, expected_specs=specs)
            checkpoints = [int(value) for value in phase["checkpoints"]]
            best_so_far, checkpoint_table, improvement_summary = build_best_so_far(
                data.trials, checkpoints=checkpoints
            )
            decision_rule = phase["decision_rule"]
            decision_method = decision_rule.get(
                "method", "legacy_fraction_of_delta_min"
            )
            if (
                decision_method
                == "best_so_far_improvement_to_final_checkpoint"
            ):
                decision = assess_optimization_budget_by_tolerance(
                    checkpoint_table,
                    checkpoints=checkpoints,
                    improvement_tolerance=float(
                        decision_rule["absolute_improvement_tolerance"]
                    ),
                    # Accept v2 only for reproducibility of its dry-run design.
                    minimum_passing_replicates_per_network_method=int(
                        decision_rule.get(
                            "minimum_passing_replicates_per_network_method",
                            decision_rule.get(
                                "minimum_passing_replicates_per_method"
                            ),
                        )
                    ),
                )
            elif decision_method == "legacy_fraction_of_delta_min":
                decision = assess_optimization_budget(
                    checkpoint_table,
                    checkpoints=checkpoints,
                    delta_min=args.delta_min,
                    late_improvement_fraction=float(
                        decision_rule[
                            "late_improvement_fraction_of_delta_min"
                        ]
                    ),
                )
            else:
                raise ValueError(
                    f"unsupported optimization decision method: {decision_method}"
                )
            selected_evaluations = decision_rule.get("selected_evaluations")
            if (
                selected_evaluations is not None
                and decision.get("recommended_evaluations")
                != int(selected_evaluations)
            ):
                raise ValueError(
                    "protocol-selected evaluations do not match the "
                    "optimization-budget recommendation"
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
                "decision_method": decision_method,
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
