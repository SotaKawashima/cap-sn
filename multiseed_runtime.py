"""Auditable block execution and atomic optimizer checkpoints."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import math
import pickle
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from analysis.candidate_validation_analysis import _manifest_errors, _metric_errors
from analysis.optimization_metrics import OBJECTIVE_NAME, compute_selfish_metrics_from_arrow
from experiment_runtime import (
    AGENT_CONFIG, NETWORKS, RUST_BINARY, REPO_ROOT,
    ExperimentConfigurationError, config_manifest_entry, git_state, now_iso,
    read_intervention_opinion_csv, remove_unrequested_raw, run_simulator,
    sha256_file, software_versions, write_intervention_opinion_csv, write_json,
    write_runtime_config, write_strategy_config,
)
from multiseed_protocol import STAGE, digest_json, read_json, verify_input_files
from run_stage7_candidate_validation import CandidateValidationRunSpec


@contextmanager
def execution_lock(root: Path) -> Iterator[int]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".execution.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExperimentConfigurationError("this experiment still has an active process") from exc
        # The Rust child inherits this descriptor, so an orphaned child also holds the lock.
        yield handle.fileno()


def initialize_experiment(
    root: Path, *, protocol_path: Path, input_files: list[dict[str, str]], resume: bool,
) -> dict[str, Any]:
    verify_input_files(input_files)
    code_paths = [
        REPO_ROOT / name for name in (
            "run_multiseed_reoptimization.py", "optimize_multiseed_objective.py",
            "multiseed_protocol.py", "multiseed_runtime.py", "experiment_runtime.py",
            "optimize_single_objective.py", "analysis/optimization_metrics.py",
        )
    ]
    software = software_versions()
    for package in ("numpy", "scipy", "cmaes", "gpytorch", "optuna-integration"):
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = None
    context = {
        "protocol": config_manifest_entry(protocol_path),
        "input_files": input_files,
        "execution_code": [config_manifest_entry(path) for path in code_paths],
        "simulator_binary": config_manifest_entry(RUST_BINARY),
        "software": software,
    }
    manifest_path = root / "execution_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("experiment exists; use --resume")
        manifest = read_json(manifest_path)
        if manifest["context"] != context:
            raise ExperimentConfigurationError("protocol, inputs, code, binary, or software changed since execution")
        return manifest["context"]
    if any(path.name != ".execution.lock" for path in root.iterdir()):
        raise ExperimentConfigurationError("existing experiment has no valid execution manifest")
    write_json(manifest_path, {
        "schema_version": 1, "stage": STAGE, "experiment_id": root.name,
        "created_at": now_iso(), "git": git_state(), "context": context,
        "context_sha256": digest_json(context),
    })
    return context


def verify_block(
    directory: Path, spec: CandidateValidationRunSpec, context: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("context_sha256") != digest_json(context) or manifest.get("spec_sha256") != digest_json(asdict(spec)):
        raise ExperimentConfigurationError(f"block settings changed: {directory}")
    errors = _manifest_errors(manifest, spec, tolerance=1e-12, expected_stage=STAGE)
    if manifest.get("network", {}).get("num_agents") != NETWORKS[spec.network].num_agents:
        errors.append("network population size changed")
    if errors:
        raise ExperimentConfigurationError("; ".join(errors))
    hashes = manifest.get("output_hashes", {})
    if not {"pop.arrow", "metrics.csv", "metrics_summary.json", "runtime.toml", "strategy.toml"}.issubset(hashes):
        raise ExperimentConfigurationError(f"block hash inventory is incomplete: {directory}")
    if spec.intervention_enabled and "inhibition_opinion.csv" not in hashes:
        raise ExperimentConfigurationError("generated opinion hash is missing")
    for name, expected in hashes.items():
        path = directory / name
        if Path(name).name != name or not path.is_file() or sha256_file(path) != expected:
            raise ExperimentConfigurationError(f"block output hash mismatch: {directory / name}")
    if any((directory / name).exists() for name in ("info.arrow", "agent.arrow")):
        raise ExperimentConfigurationError("raw-level pop policy violated")
    metrics = compute_selfish_metrics_from_arrow(
        directory / "pop.arrow", num_agents=NETWORKS[spec.network].num_agents,
        expected_iterations=spec.iterations,
    )
    errors = _metric_errors(pd.read_csv(directory / "metrics.csv"), metrics.per_iteration,
                            expected_iterations=spec.iterations, tolerance=1e-12)
    summary = read_json(directory / "metrics_summary.json")
    for observed in (manifest["objective"]["value"], summary[OBJECTIVE_NAME]):
        if not math.isclose(observed, metrics.objective_value, rel_tol=0, abs_tol=1e-12):
            errors.append("saved objective differs from raw")
    if errors:
        raise ExperimentConfigurationError("; ".join(errors))
    return manifest, metrics


def run_block(
    directory: Path, spec: CandidateValidationRunSpec, *, context: dict[str, Any],
    experiment_id: str, resume: bool, lock_fd: int,
) -> float:
    if directory.exists():
        if not resume:
            raise FileExistsError(f"block exists: {directory}")
        manifest = read_json(directory / "manifest.json")
        if manifest.get("context_sha256") != digest_json(context) or manifest.get("spec_sha256") != digest_json(asdict(spec)):
            raise ExperimentConfigurationError("cannot resume a block with different settings")
        if manifest["status"] == "completed":
            return float(verify_block(directory, spec, context)[1].objective_value)
        if manifest["status"] not in {"running", "failed", "interrupted"}:
            raise ExperimentConfigurationError(f"unsupported block status: {manifest['status']}")
        attempts = directory.parent / "_failed_attempts"
        attempts.mkdir(exist_ok=True)
        index = 1
        while (attempts / f"{directory.name}_attempt_{index:04d}").exists():
            index += 1
        directory.rename(attempts / f"{directory.name}_attempt_{index:04d}")
    directory.mkdir(parents=True, exist_ok=False)
    network = NETWORKS[spec.network]
    manifest: dict[str, Any] = {
        "schema_version": 1, "stage": STAGE, "experiment_id": experiment_id,
        "run_type": "fixed_condition", "status": "running", "created_at": now_iso(),
        "context_sha256": digest_json(context), "spec_sha256": digest_json(asdict(spec)),
        "network": {"id": network.id, "num_agents": network.num_agents,
                    **config_manifest_entry(network.config_path)},
        "agent": config_manifest_entry(AGENT_CONFIG),
        "runtime": {"simulator_seed": spec.simulator_seed, "iteration_count": spec.iterations},
        "intervention": {"enabled": spec.intervention_enabled, "condition_id": spec.condition_id,
                         "applied_parameters": None,
                         "opinion_mode": "generated_from_design_variables" if spec.intervention_enabled else None},
        "objective": {"name": OBJECTIVE_NAME, "definition_version": "cumulative_selfish_fraction_v1", "value": None},
        "outputs": {}, "output_hashes": {}, "timing_sec": {}, "failure": None,
    }
    write_json(directory / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        runtime = write_runtime_config(directory / "runtime.toml", simulator_seed=spec.simulator_seed,
                                       iteration_count=spec.iterations)
        opinion = None
        if spec.intervention_enabled:
            opinion = directory / "inhibition_opinion.csv"
            applied = write_intervention_opinion_csv(opinion, certainty=spec.certainty, effectiveness=spec.effectiveness)
            manifest["intervention"].update({
                "applied_parameters": applied, "opinion_csv": config_manifest_entry(opinion),
                "opinion_values": read_intervention_opinion_csv(opinion)[0],
            })
        strategy = write_strategy_config(directory / "strategy.toml", intervention_opinion_csv=opinion)
        write_json(directory / "manifest.json", manifest)
        simulation = run_simulator(
            identifier="simulation", output_dir=directory, runtime_path=runtime,
            network_path=network.config_path, strategy_path=strategy,
            intervention_enabled=spec.intervention_enabled,
            stdout_path=directory / "stdout.log", stderr_path=directory / "stderr.log",
            pass_fds=(lock_fd,),
        )
        manifest["simulator_command"] = simulation.command
        manifest["timing_sec"]["simulation"] = simulation.elapsed_sec
        metric_start = time.perf_counter()
        metrics = compute_selfish_metrics_from_arrow(
            simulation.arrow_paths["pop"], num_agents=network.num_agents,
            expected_iterations=spec.iterations,
        )
        manifest["timing_sec"]["metric_calculation"] = time.perf_counter() - metric_start
        metrics.per_iteration.to_csv(directory / "metrics.csv", index=False)
        write_json(directory / "metrics_summary.json", metrics.summary)
        remove_unrequested_raw(simulation.arrow_paths, "pop")
        names = ["pop.arrow", "metrics.csv", "metrics_summary.json", "runtime.toml", "strategy.toml"]
        if opinion is not None:
            names.append(opinion.name)
        manifest["output_hashes"] = {name: sha256_file(directory / name) for name in names}
        manifest["outputs"] = {"pop_arrow": "pop.arrow", "metrics_csv": "metrics.csv",
                               "metrics_summary_json": "metrics_summary.json"}
        manifest["objective"]["value"] = metrics.objective_value
        manifest["metrics"] = metrics.summary
        manifest["status"] = "completed"
        return float(metrics.objective_value)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["timing_sec"]["total"] = time.perf_counter() - started
        manifest["updated_at"] = now_iso()
        write_json(directory / "manifest.json", manifest)


def save_checkpoint(root: Path, state: dict[str, Any]) -> None:
    data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(data).hexdigest()
    folder = root / "checkpoints"
    folder.mkdir(exist_ok=True)
    target = folder / f"{digest}.pickle"
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    # Publish the pointer last; a crash cannot mix sampler and study states.
    write_json(root / "checkpoint.json", {"sha256": digest, "path": f"checkpoints/{digest}.pickle"})


def load_checkpoint(root: Path) -> dict[str, Any]:
    pointer = read_json(root / "checkpoint.json")
    path = (root / pointer["path"]).resolve()
    if not path.is_relative_to((root / "checkpoints").resolve()) or sha256_file(path) != pointer["sha256"]:
        raise ExperimentConfigurationError("optimizer checkpoint is invalid")
    # Only load checkpoints produced by this experiment, never external pickle files.
    state = pickle.loads(path.read_bytes())
    if not isinstance(state, dict) or not {"study", "active_trial", "context_sha256", "spec_sha256"}.issubset(state):
        raise ExperimentConfigurationError("optimizer checkpoint structure is invalid")
    return state
