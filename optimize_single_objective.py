"""Run the protocol-defined single-objective optimization experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from botorch.exceptions.warnings import OptimizationWarning
from optuna.exceptions import ExperimentalWarning
from optuna.integration import BoTorchSampler

from analysis.optimization_metrics import (
    MetricValidationError,
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
    compute_selfish_metrics_from_arrow,
)
from experiment_runtime import (
    AGENT_CONFIG,
    MANIFEST_SCHEMA_VERSION,
    NETWORKS,
    RUST_BINARY,
    STRATEGY_TEMPLATE,
    SUMMER_EXPERIMENT_ROOT,
    ExperimentConfigurationError,
    SimulationExecutionError,
    append_study_manifest,
    command_for_manifest,
    config_manifest_entry,
    create_unique_run_directory,
    git_state,
    make_experiment_id,
    now_iso,
    relative_to_run,
    remove_unrequested_raw,
    resolve_output_root,
    run_simulator,
    sha256_file,
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


METHODS = ("bo_gp", "cma_es", "random_search")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the summer protocol's single-objective optimization."
    )
    parser.add_argument("--stage", default="stage1_metric_validation")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--purpose", default="single_objective_optimization")
    parser.add_argument("--network", choices=sorted(NETWORKS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--optimizer-replicate", type=int, required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--simulator-seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument(
        "--raw-level", choices=["pop", "info_pop", "all"], default="pop"
    )
    parser.add_argument(
        "--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT
    )
    return parser.parse_args(argv)


def create_sampler(method: str, *, seed: int, startup_trials: int):
    if method == "bo_gp":
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        warnings.filterwarnings("ignore", category=OptimizationWarning)
        return BoTorchSampler(n_startup_trials=startup_trials, seed=seed)
    if method == "cma_es":
        return optuna.samplers.CmaEsSampler(seed=seed)
    if method == "random_search":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ExperimentConfigurationError(f"unsupported optimization method: {method}")


def _serializable_trial_row(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "trial": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "proposed_certainty": trial.params.get("certainty"),
        "proposed_effectiveness": trial.params.get("effectiveness"),
        "applied_certainty": trial.user_attrs.get("applied_certainty"),
        "applied_effectiveness": trial.user_attrs.get("applied_effectiveness"),
        "cumulative_selfish_fraction": trial.user_attrs.get(
            "cumulative_selfish_fraction"
        ),
        "peak_new_selfish_ratio": trial.user_attrs.get("peak_new_selfish_ratio"),
        "mean_new_selfish_rate_per_step": trial.user_attrs.get(
            "mean_new_selfish_rate_per_step"
        ),
        "n_iterations": trial.user_attrs.get("n_iterations"),
        "n_zero_selfish_iterations": trial.user_attrs.get(
            "n_zero_selfish_iterations"
        ),
        "simulation_sec": trial.user_attrs.get("simulation_sec"),
        "metric_calculation_sec": trial.user_attrs.get("metric_calculation_sec"),
        "trial_total_sec": trial.user_attrs.get("trial_total_sec"),
        "simulator_command": trial.user_attrs.get("simulator_command"),
        "raw_dir": trial.user_attrs.get("raw_dir"),
        "failure_stage": trial.user_attrs.get("failure_stage"),
        "failure_type": trial.user_attrs.get("failure_type"),
        "failure_message": trial.user_attrs.get("failure_message"),
    }


def _trial_failure_attrs(trial: optuna.Trial, exc: BaseException, stage: str) -> None:
    trial.set_user_attr("failure_stage", getattr(exc, "stage", stage))
    trial.set_user_attr("failure_type", type(exc).__name__)
    trial.set_user_attr("failure_message", str(exc))
    exit_code = getattr(exc, "exit_code", None)
    if exit_code is not None:
        trial.set_user_attr("failure_exit_code", exit_code)


def run_optimization(args: argparse.Namespace) -> Path:
    stage = validate_safe_name(args.stage, "stage")
    optimizer_replicate = validate_positive_integer(
        args.optimizer_replicate, "optimizer_replicate"
    )
    optimizer_seed = validate_nonnegative_integer(
        args.optimizer_seed, "optimizer_seed"
    )
    simulator_seed = validate_nonnegative_integer(
        args.simulator_seed, "simulator_seed"
    )
    iterations = validate_positive_integer(args.iterations, "iterations")
    n_trials = validate_positive_integer(args.trials, "trials")
    startup_trials = validate_positive_integer(args.startup_trials, "startup_trials")
    startup_trials = min(startup_trials, n_trials)
    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id(args.purpose)
    )
    network = NETWORKS[args.network]
    output_root = resolve_output_root(args.output_root)
    experiment_root = output_root / stage / experiment_id
    run_dir = (
        experiment_root
        / network.id
        / args.method
        / f"optseed_{optimizer_replicate}"
    )
    create_unique_run_directory(run_dir)
    raw_dir = run_dir / "raw"
    log_dir = run_dir / "logs"
    raw_dir.mkdir()
    log_dir.mkdir()

    manifest_path = run_dir / "manifest.json"
    runtime_path = write_runtime_config(
        run_dir / "runtime.toml",
        simulator_seed=simulator_seed,
        iteration_count=iterations,
    )
    database_path = run_dir / "study.db"
    storage_url = f"sqlite:///{database_path.resolve().as_posix()}"

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "stage": stage,
        "run_type": "single_objective_optimization",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "git": git_state(),
        "software": software_versions(),
        "simulator_binary": config_manifest_entry(RUST_BINARY),
        "invocation_command": command_for_manifest(),
        "network": {
            "id": network.id,
            **config_manifest_entry(network.config_path),
            "network_seed": None,
            "num_agents": network.num_agents,
        },
        "agent": config_manifest_entry(AGENT_CONFIG),
        "strategy_template": config_manifest_entry(STRATEGY_TEMPLATE),
        "runtime": {
            "simulator_seed": simulator_seed,
            "iteration_count": iterations,
            **config_manifest_entry(runtime_path),
        },
        "intervention": {
            "enabled": True,
            "parameter_bounds": {
                "certainty": [0.5, 1.0],
                "effectiveness": [0.5, 1.0],
            },
            "application_precision_decimal_places": 4,
        },
        "optimization": {
            "method": args.method,
            "optimizer_replicate": optimizer_replicate,
            "optimizer_seed": optimizer_seed,
            "n_trials_requested": n_trials,
            "startup_trials": startup_trials if args.method == "bo_gp" else None,
            "raw_level": args.raw_level,
        },
        "objective": {
            "name": OBJECTIVE_NAME,
            "definition_version": OBJECTIVE_DEFINITION_VERSION,
            "direction": "minimize",
        },
        "outputs": {
            "study_db": "study.db",
            "trials_csv": "trials.csv",
            "summary_json": "summary.json",
            "raw_dir": "raw",
            "logs_dir": "logs",
        },
        "counts": {
            "complete": 0,
            "failed": 0,
            "pruned": 0,
        },
        "timing_sec": {
            "optimization_total": None,
            "simulation_total": None,
            "metric_calculation_total": None,
            "optimizer_overhead": None,
        },
        "best": None,
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

    sampler = create_sampler(
        args.method, seed=optimizer_seed, startup_trials=startup_trials
    )
    study = optuna.create_study(
        study_name=(
            f"{experiment_id}_{network.id}_{args.method}_"
            f"optseed_{optimizer_replicate}"
        ),
        direction="minimize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=False,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        stage_name = "configuration"
        trial_dir = raw_dir / f"trial_{trial.number:04d}"
        create_unique_run_directory(trial_dir)
        stdout_path = log_dir / f"trial_{trial.number:04d}_stdout.log"
        stderr_path = log_dir / f"trial_{trial.number:04d}_stderr.log"
        trial.set_user_attr("raw_dir", relative_to_run(trial_dir, run_dir))

        proposed_certainty = trial.suggest_float("certainty", 0.5, 1.0)
        proposed_effectiveness = trial.suggest_float("effectiveness", 0.5, 1.0)
        opinion_path = trial_dir / "inhibition_opinion.csv"
        strategy_path = trial_dir / "strategy.toml"

        try:
            applied = write_intervention_opinion_csv(
                opinion_path,
                certainty=proposed_certainty,
                effectiveness=proposed_effectiveness,
            )
            write_strategy_config(
                strategy_path,
                intervention_opinion_csv=opinion_path,
            )
            trial.set_user_attr("applied_certainty", applied["certainty"])
            trial.set_user_attr("applied_effectiveness", applied["effectiveness"])
            trial.set_user_attr("opinion_csv_sha256", sha256_file(opinion_path))
            trial.set_user_attr("strategy_sha256", sha256_file(strategy_path))

            stage_name = "simulation"
            simulation = run_simulator(
                identifier=f"trial_{trial.number:04d}",
                output_dir=trial_dir,
                runtime_path=runtime_path,
                network_path=network.config_path,
                strategy_path=strategy_path,
                intervention_enabled=True,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            trial.set_user_attr(
                "simulator_command", json.dumps(simulation.command, ensure_ascii=False)
            )
            trial.set_user_attr("simulation_sec", simulation.elapsed_sec)

            stage_name = "metric_calculation"
            metric_start = time.perf_counter()
            metrics = compute_selfish_metrics_from_arrow(
                simulation.arrow_paths["pop"],
                num_agents=network.num_agents,
                expected_iterations=iterations,
            )
            metric_elapsed = time.perf_counter() - metric_start
            trial.set_user_attr("metric_calculation_sec", metric_elapsed)
            for key, value in metrics.summary.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    trial.set_user_attr(key, value)

            metrics_path = trial_dir / "metrics.csv"
            metrics_summary_path = trial_dir / "metrics_summary.json"
            metrics.per_iteration.to_csv(metrics_path, index=False)
            write_json(metrics_summary_path, metrics.summary)
            retained = remove_unrequested_raw(simulation.arrow_paths, args.raw_level)
            trial.set_user_attr(
                "retained_raw",
                json.dumps(
                    {
                        kind: relative_to_run(path, run_dir)
                        for kind, path in retained.items()
                    },
                    sort_keys=True,
                ),
            )
            return metrics.objective_value
        except (SimulationExecutionError, MetricValidationError, OSError) as exc:
            _trial_failure_attrs(trial, exc, stage_name)
            raise
        finally:
            trial.set_user_attr(
                "trial_total_sec", time.perf_counter() - trial_start
            )

    optimization_start = time.perf_counter()
    fatal_error: BaseException | None = None
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=n_trials,
            catch=(SimulationExecutionError, MetricValidationError, OSError),
        )
    except BaseException as exc:
        fatal_error = exc
    optimization_elapsed = time.perf_counter() - optimization_start

    rows = [_serializable_trial_row(trial) for trial in study.trials]
    trials_frame = pd.DataFrame(rows)
    trials_path = run_dir / "trials.csv"
    trials_frame.to_csv(trials_path, index=False)

    complete_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    failed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL
    ]
    pruned_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
    ]
    simulation_total = sum(
        float(trial.user_attrs.get("simulation_sec", 0.0)) for trial in study.trials
    )
    metric_total = sum(
        float(trial.user_attrs.get("metric_calculation_sec", 0.0))
        for trial in study.trials
    )
    objective_total = sum(
        float(trial.user_attrs.get("trial_total_sec", 0.0)) for trial in study.trials
    )

    best: dict[str, Any] | None = None
    if complete_trials:
        best_trial = study.best_trial
        best = {
            "trial": best_trial.number,
            "value": best_trial.value,
            "proposed_parameters": best_trial.params,
            "applied_parameters": {
                "certainty": best_trial.user_attrs.get("applied_certainty"),
                "effectiveness": best_trial.user_attrs.get("applied_effectiveness"),
            },
            "metrics": {
                key: best_trial.user_attrs.get(key)
                for key in (
                    "cumulative_selfish_fraction",
                    "peak_new_selfish_ratio",
                    "mean_new_selfish_rate_per_step",
                    "mean_first_selfish_step",
                    "mean_last_selfish_step",
                    "mean_t50_selfish_step",
                    "mean_t90_selfish_step",
                )
            },
        }

    summary = {
        "experiment_id": experiment_id,
        "network": network.id,
        "method": args.method,
        "optimizer_replicate": optimizer_replicate,
        "optimizer_seed": optimizer_seed,
        "simulator_seed": simulator_seed,
        "iteration_count": iterations,
        "objective_name": OBJECTIVE_NAME,
        "objective_definition_version": OBJECTIVE_DEFINITION_VERSION,
        "n_trials_requested": n_trials,
        "n_trials_recorded": len(study.trials),
        "n_trials_complete": len(complete_trials),
        "n_trials_failed": len(failed_trials),
        "n_trials_pruned": len(pruned_trials),
        "best": best,
        "timing_sec": {
            "optimization_total": optimization_elapsed,
            "simulation_total": simulation_total,
            "metric_calculation_total": metric_total,
            "optimizer_overhead": max(0.0, optimization_elapsed - objective_total),
        },
    }
    write_json(run_dir / "summary.json", summary)

    manifest["counts"] = {
        "complete": len(complete_trials),
        "failed": len(failed_trials),
        "pruned": len(pruned_trials),
    }
    manifest["timing_sec"] = summary["timing_sec"]
    manifest["best"] = best
    manifest["updated_at"] = now_iso()
    if fatal_error is not None:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "stage": "optimization",
            "type": type(fatal_error).__name__,
            "message": str(fatal_error),
        }
    elif not complete_trials:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "stage": "optimization",
            "type": "NoCompleteTrials",
            "message": "no optimization trial completed successfully",
        }
    elif (
        len(complete_trials) != n_trials
        or failed_trials
        or pruned_trials
    ):
        manifest["status"] = "failed"
        manifest["failure"] = {
            "stage": "optimization",
            "type": "IncompleteOptimizationBudget",
            "message": (
                "the requested optimization budget was not completed without "
                "failed or pruned trials"
            ),
        }
    else:
        manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    append_study_manifest(
        experiment_root,
        experiment_id=experiment_id,
        stage=stage,
        run_dir=run_dir,
        status=manifest["status"],
    )

    if fatal_error is not None:
        raise fatal_error
    if not complete_trials:
        raise RuntimeError("optimization produced no complete trials")
    if manifest["status"] != "completed":
        raise RuntimeError("optimization did not complete the requested trial budget")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = run_optimization(args)
    except (
        ExperimentConfigurationError,
        SimulationExecutionError,
        MetricValidationError,
        FileExistsError,
        RuntimeError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Completed single-objective optimization: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
