"""Run one fixed intervention or no-intervention simulation condition."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from analysis.optimization_metrics import (
    MetricValidationError,
    compute_selfish_metrics_from_arrow,
)
from experiment_runtime import (
    AGENT_CONFIG,
    MANIFEST_SCHEMA_VERSION,
    NETWORKS,
    STRATEGY_TEMPLATE,
    SUMMER_EXPERIMENT_ROOT,
    ExperimentConfigurationError,
    SimulationExecutionError,
    append_study_manifest,
    build_simulator_command,
    command_for_manifest,
    config_manifest_entry,
    create_unique_run_directory,
    git_state,
    make_experiment_id,
    now_iso,
    parameter_condition_id,
    relative_to_run,
    remove_unrequested_raw,
    resolve_output_root,
    run_simulator,
    software_versions,
    validate_experiment_id,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_safe_name,
    write_intervention_opinion_csv,
    write_json,
    write_runtime_config,
    write_strategy_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed simulation condition with reproducible outputs."
    )
    parser.add_argument("--stage", default="stage1_metric_validation")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--purpose", default="fixed_condition")
    parser.add_argument("--network", choices=sorted(NETWORKS), required=True)
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--certainty", type=float, default=None)
    parser.add_argument("--effectiveness", type=float, default=None)
    parser.add_argument("--no-intervention", action="store_true")
    parser.add_argument("--simulator-seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--raw-level", choices=["pop", "info_pop", "all"], default="all"
    )
    parser.add_argument(
        "--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT
    )
    return parser.parse_args(argv)


def resolve_condition(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_intervention:
        if args.certainty is not None or args.effectiveness is not None:
            raise ExperimentConfigurationError(
                "certainty/effectiveness must be omitted for no-intervention runs"
            )
        condition_id = args.condition_id or "none"
        if condition_id != "none":
            raise ExperimentConfigurationError(
                "no-intervention runs must use condition_id 'none'"
            )
        return {
            "enabled": False,
            "condition_id": "none",
            "proposed_parameters": None,
        }

    if args.certainty is None or args.effectiveness is None:
        raise ExperimentConfigurationError(
            "intervention runs require both --certainty and --effectiveness"
        )
    generated_condition_id = parameter_condition_id(
        args.certainty, args.effectiveness
    )
    condition_id = args.condition_id or generated_condition_id
    validate_safe_name(condition_id, "condition_id")
    return {
        "enabled": True,
        "condition_id": condition_id,
        "proposed_parameters": {
            "certainty": float(args.certainty),
            "effectiveness": float(args.effectiveness),
        },
    }


def run_fixed_condition(args: argparse.Namespace) -> Path:
    stage = validate_safe_name(args.stage, "stage")
    simulator_seed = validate_nonnegative_integer(
        args.simulator_seed, "simulator_seed"
    )
    iterations = validate_positive_integer(args.iterations, "iterations")
    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id(args.purpose)
    )
    network = NETWORKS[args.network]
    condition = resolve_condition(args)
    output_root = resolve_output_root(args.output_root)
    experiment_root = output_root / stage / experiment_id
    run_dir = (
        experiment_root
        / network.id
        / condition["condition_id"]
        / f"simseed_{simulator_seed}"
    )
    create_unique_run_directory(run_dir)

    manifest_path = run_dir / "manifest.json"
    runtime_path = write_runtime_config(
        run_dir / "runtime.toml",
        simulator_seed=simulator_seed,
        iteration_count=iterations,
    )

    applied_parameters: dict[str, float] | None = None
    opinion_path: Path | None = None
    if condition["enabled"]:
        opinion_path = run_dir / "inhibition_opinion.csv"
        applied_parameters = write_intervention_opinion_csv(
            opinion_path,
            certainty=condition["proposed_parameters"]["certainty"],
            effectiveness=condition["proposed_parameters"]["effectiveness"],
        )

    strategy_path = write_strategy_config(
        run_dir / "strategy.toml",
        intervention_opinion_csv=opinion_path,
    )
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    identifier = "simulation"
    simulator_command = build_simulator_command(
        identifier=identifier,
        output_dir=run_dir,
        runtime_path=runtime_path,
        network_path=network.config_path,
        agent_path=AGENT_CONFIG,
        strategy_path=strategy_path,
        intervention_enabled=condition["enabled"],
    )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "stage": stage,
        "run_type": "fixed_condition",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "git": git_state(),
        "software": software_versions(),
        "simulator_binary": config_manifest_entry(
            Path(simulator_command[0])
        ),
        "invocation_command": command_for_manifest(),
        "simulator_command": simulator_command,
        "network": {
            "id": network.id,
            **config_manifest_entry(network.config_path),
            "network_seed": None,
            "num_agents": network.num_agents,
        },
        "agent": config_manifest_entry(AGENT_CONFIG),
        "runtime": {
            "simulator_seed": simulator_seed,
            "iteration_count": iterations,
            **config_manifest_entry(runtime_path),
        },
        "strategy": config_manifest_entry(strategy_path),
        "intervention": {
            **condition,
            "applied_parameters": applied_parameters,
            "opinion_csv": (
                None if opinion_path is None else config_manifest_entry(opinion_path)
            ),
        },
        "objective": {
            "name": "cumulative_selfish_fraction",
            "definition_version": "cumulative_selfish_fraction_v1",
            "value": None,
        },
        "metrics": None,
        "outputs": {},
        "timing_sec": {
            "simulation": None,
            "metric_calculation": None,
            "total": None,
        },
        "failure": None,
    }
    write_json(manifest_path, manifest)
    append_study_manifest(
        experiment_root,
        experiment_id=experiment_id,
        stage=stage,
        run_dir=run_dir,
        status="running",
    )

    total_start = time.perf_counter()
    failure_stage = "simulation"
    try:
        simulation = run_simulator(
            identifier=identifier,
            output_dir=run_dir,
            runtime_path=runtime_path,
            network_path=network.config_path,
            strategy_path=strategy_path,
            intervention_enabled=condition["enabled"],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        manifest["simulator_command"] = simulation.command
        manifest["timing_sec"]["simulation"] = simulation.elapsed_sec

        failure_stage = "metric_calculation"
        metric_start = time.perf_counter()
        metrics = compute_selfish_metrics_from_arrow(
            simulation.arrow_paths["pop"],
            num_agents=network.num_agents,
            expected_iterations=iterations,
        )
        metric_elapsed = time.perf_counter() - metric_start
        metrics_path = run_dir / "metrics.csv"
        metrics_summary_path = run_dir / "metrics_summary.json"
        metrics.per_iteration.to_csv(metrics_path, index=False)
        write_json(metrics_summary_path, metrics.summary)

        retained = remove_unrequested_raw(simulation.arrow_paths, args.raw_level)
        manifest["objective"]["value"] = metrics.objective_value
        manifest["metrics"] = metrics.summary
        manifest["outputs"] = {
            "metrics_csv": relative_to_run(metrics_path, run_dir),
            "metrics_summary_json": relative_to_run(metrics_summary_path, run_dir),
            "stdout_log": relative_to_run(stdout_path, run_dir),
            "stderr_log": relative_to_run(stderr_path, run_dir),
            **{
                f"{kind}_arrow": relative_to_run(path, run_dir)
                for kind, path in retained.items()
            },
        }
        manifest["timing_sec"]["metric_calculation"] = metric_elapsed
        manifest["status"] = "completed"
    except (SimulationExecutionError, MetricValidationError, OSError) as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "stage": getattr(exc, "stage", failure_stage),
            "exit_code": getattr(exc, "exit_code", None),
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        manifest["updated_at"] = now_iso()
        manifest["timing_sec"]["total"] = time.perf_counter() - total_start
        write_json(manifest_path, manifest)
        append_study_manifest(
            experiment_root,
            experiment_id=experiment_id,
            stage=stage,
            run_dir=run_dir,
            status=manifest["status"],
        )

    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = run_fixed_condition(args)
    except (
        ExperimentConfigurationError,
        SimulationExecutionError,
        MetricValidationError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Completed fixed-condition run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
