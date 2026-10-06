"""Sequential Optuna optimization using complete five-block evaluations."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from experiment_runtime import ExperimentConfigurationError, now_iso, sha256_file, write_json
from multiseed_protocol import STAGE, OptimizationSpec, block_spec, digest_json, read_json
from multiseed_runtime import load_checkpoint, run_block, save_checkpoint
from optimize_single_objective import create_sampler


def trial_block_specs(spec: OptimizationSpec, trial: Any) -> list[Any]:
    coordinates = trial.user_attrs["applied_parameters"]
    result = []
    for seed in spec.simulator_seeds:
        item = block_spec(
            phase="optimization", network=spec.network,
            condition_id=f"trial_{trial.number:04d}", simulator_seed=seed,
            certainty=coordinates["certainty"], effectiveness=coordinates["effectiveness"],
        )
        result.append(replace(item, relative_run_dir=(
            f"{spec.relative_run_dir}/raw/trial_{trial.number:04d}/simseed_{seed}"
        )))
    return result


def _trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    return [{
        "trial": trial.number, "state": trial.state.name, "value": trial.value,
        "proposed_certainty": trial.params.get("certainty"),
        "proposed_effectiveness": trial.params.get("effectiveness"),
        "applied_certainty": trial.user_attrs.get("applied_parameters", {}).get("certainty"),
        "applied_effectiveness": trial.user_attrs.get("applied_parameters", {}).get("effectiveness"),
        "block_count": len(trial.user_attrs.get("block_values", [])),
        "simulation_sec": trial.user_attrs.get("simulation_sec"),
        "metric_calculation_sec": trial.user_attrs.get("metric_calculation_sec"),
    } for trial in study.trials]


def run_optimization(
    experiment_root: Path, spec: OptimizationSpec, *, context: dict[str, Any],
    resume: bool, lock_fd: int, max_evaluations: int = 50,
) -> Path:
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= spec.trials:
        raise ExperimentConfigurationError("max evaluations must be between 1 and 50")
    root = experiment_root / spec.relative_run_dir
    expected = {"context_sha256": digest_json(context), "spec_sha256": digest_json(asdict(spec))}
    elapsed_before = 0.0
    if root.exists():
        if not resume:
            raise FileExistsError(f"optimization exists: {root}")
        previous = read_json(root / "manifest.json")
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ExperimentConfigurationError("optimization settings changed")
        elapsed_before = float(previous.get("timing_sec", {}).get("total", 0))
        if not (root / "checkpoint.json").exists():
            if (root / "raw").exists() or (root / "trials.csv").exists():
                raise ExperimentConfigurationError("optimizer checkpoint is missing; refusing to resample")
            state = None
        else:
            state = load_checkpoint(root)
    else:
        root.mkdir(parents=True, exist_ok=False)
        state = None
    manifest = {
        "schema_version": 1, "stage": STAGE, "experiment_id": experiment_root.name,
        "run_type": "multiseed_optimization", "status": "running", "spec": asdict(spec),
        **expected, "updated_at": now_iso(), "counts": {}, "failure": None,
        "checkpoint_format": "atomic_pickled_in_memory_optuna_study_and_sampler",
    }
    write_json(root / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        if state is None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="minimize", study_name=f"{experiment_root.name}_{spec.key}",
                sampler=create_sampler(spec.method, seed=spec.optimizer_seed, startup_trials=spec.startup_trials),
            )
            state = {**expected, "study": study, "active_trial": None}
            save_checkpoint(root, state)
        if any(state.get(key) != value for key, value in expected.items()):
            raise ExperimentConfigurationError("checkpoint provenance does not match optimization")
        study = state["study"]
        trials = study.trials
        if any(trial.state not in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING} for trial in trials):
            raise ExperimentConfigurationError("checkpoint contains failed or pruned evaluations")
        running = [trial for trial in trials if trial.state == optuna.trial.TrialState.RUNNING]
        active = state["active_trial"]
        if len(running) != int(active is not None) or (active is not None and running[0].number != active.number):
            raise ExperimentConfigurationError("checkpoint active evaluation is inconsistent")
        if len(trials) > spec.trials:
            raise ExperimentConfigurationError("checkpoint exceeds the evaluation budget")

        while sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials) < max_evaluations:
            active = state["active_trial"]
            if active is None:
                active = study.ask()
                certainty = active.suggest_float("certainty", 0.5, 1.0)
                effectiveness = active.suggest_float("effectiveness", 0.5, 1.0)
                active.set_user_attr("applied_parameters", {
                    "certainty": round(certainty, 4), "effectiveness": round(effectiveness, 4),
                })
                state["active_trial"] = active
                save_checkpoint(root, state)
            values = []
            timings = []
            for index, item in enumerate(trial_block_specs(spec, active), start=1):
                print(f"{spec.key} evaluation {active.number + 1}/{spec.trials}, block {index}/5", flush=True)
                directory = experiment_root / item.relative_run_dir
                value = run_block(directory, item, context=context, experiment_id=experiment_root.name,
                                  resume=True, lock_fd=lock_fd)
                values.append(value)
                timings.append(read_json(directory / "manifest.json")["timing_sec"])
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ExperimentConfigurationError("a complete evaluation requires five finite blocks")
            value = float(sum(values) / 5)
            active.set_user_attr("block_values", values)
            active.set_user_attr("simulation_sec", sum(item["simulation"] for item in timings))
            active.set_user_attr("metric_calculation_sec", sum(item["metric_calculation"] for item in timings))
            study.tell(active, value)
            state["active_trial"] = None
            save_checkpoint(root, state)
            print(f"{spec.key} evaluation {active.number + 1}: mean Jcum={value:.9f}", flush=True)
        rows = _trial_rows(study)
        pd.DataFrame(rows).to_csv(root / "trials.csv", index=False)
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        manifest["status"] = "completed" if len(completed) == spec.trials else "partial"
        if completed:
            best = study.best_trial
            manifest["best"] = {"trial": best.number, "value": best.value,
                                **best.user_attrs["applied_parameters"]}
        write_json(root / "summary.json", {
            "status": manifest["status"], "spec": asdict(spec), "best": manifest.get("best"),
            "counts": {"complete": len(completed), "failed": 0, "pruned": 0},
        })
        manifest["output_hashes"] = {
            name: sha256_file(root / name) for name in ("trials.csv", "summary.json")
        }
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if state is not None:
            manifest["counts"] = {
                "complete": sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in state["study"].trials),
                "failed": sum(trial.state == optuna.trial.TrialState.FAIL for trial in state["study"].trials),
                "pruned": sum(trial.state == optuna.trial.TrialState.PRUNED for trial in state["study"].trials),
            }
        manifest["timing_sec"] = {"total": elapsed_before + time.perf_counter() - started}
        manifest["updated_at"] = now_iso()
        write_json(root / "manifest.json", manifest)
    return root
