"""Shared runtime and provenance helpers for summer 2026 experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import toml


REPO_ROOT = Path(__file__).resolve().parent
ENV_ROOT = REPO_ROOT / "v2" / "test_2"
SUMMER_EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "summer_2026"
RUST_BINARY = REPO_ROOT / "target" / "release" / (
    "v2.exe" if os.name == "nt" else "v2"
)
AGENT_CONFIG = ENV_ROOT / "agent" / "agent-type6.toml"
STRATEGY_TEMPLATE = ENV_ROOT / "strategy" / "strategy-config.toml"

MANIFEST_SCHEMA_VERSION = 1
TOKYO = ZoneInfo("Asia/Tokyo")
EXPERIMENT_ID_PATTERN = re.compile(
    r"^\d{8}_\d{6}_[a-z0-9]+(?:_[a-z0-9]+)*_v\d{2}$"
)
SAFE_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
INTERVENTION_OPINION_COLUMNS = (
    "phi_b0",
    "phi_b1",
    "phi_u",
    "phi_a0",
    "phi_a1",
    "psi0_b0",
    "psi0_b1",
    "psi0_u",
    "psi1_b0",
    "psi1_b1",
    "psi1_u",
    "b0_b0",
    "b0_b1",
    "b0_u",
    "b1_b0",
    "b1_b1",
    "b1_u",
)


@dataclass(frozen=True)
class NetworkSpec:
    id: str
    config_path: Path
    num_agents: int


NETWORKS: dict[str, NetworkSpec] = {
    "ba1000": NetworkSpec(
        id="ba1000",
        config_path=ENV_ROOT / "network" / "network-ba1000.toml",
        num_agents=1000,
    ),
    "facebook": NetworkSpec(
        id="facebook",
        config_path=ENV_ROOT / "network" / "network-facebook.toml",
        num_agents=4039,
    ),
    "wiki_vote": NetworkSpec(
        id="wiki_vote",
        config_path=ENV_ROOT / "network" / "network-wiki-vote.toml",
        num_agents=7115,
    ),
}


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment request violates the fixed protocol."""


class SimulationExecutionError(RuntimeError):
    """Raised when the simulator exits unsuccessfully or misses raw outputs."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.exit_code = exit_code


@dataclass(frozen=True)
class SimulationRunResult:
    command: list[str]
    elapsed_sec: float
    stdout_path: Path
    stderr_path: Path
    arrow_paths: dict[str, Path]


def now_iso() -> str:
    return datetime.now(TOKYO).isoformat(timespec="seconds")


def make_experiment_id(purpose: str, *, version: int = 1) -> str:
    purpose = purpose.strip().lower().replace("-", "_")
    if not SAFE_NAME_PATTERN.fullmatch(purpose):
        raise ExperimentConfigurationError(
            "purpose must use lowercase ASCII letters, digits, and underscores"
        )
    if version <= 0 or version > 99:
        raise ExperimentConfigurationError("version must be between 1 and 99")
    timestamp = datetime.now(TOKYO).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{purpose}_v{version:02d}"


def validate_experiment_id(experiment_id: str) -> str:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ExperimentConfigurationError(
            "experiment_id must match YYYYMMDD_HHMMSS_<purpose>_vNN"
        )
    return experiment_id


def validate_safe_name(value: str, field_name: str) -> str:
    if not SAFE_NAME_PATTERN.fullmatch(value):
        raise ExperimentConfigurationError(
            f"{field_name} must use lowercase ASCII letters, digits, and underscores"
        )
    return value


def validate_positive_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentConfigurationError(f"{field_name} must be a positive integer")
    return value


def validate_nonnegative_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentConfigurationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def resolve_output_root(path: str | Path) -> Path:
    output_root = Path(path)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    return output_root.resolve()


def create_unique_run_directory(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(
            f"run directory already exists and will not be overwritten: {path}"
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def relative_to_run(path: str | Path, run_dir: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(run_dir).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def config_manifest_entry(path: str | Path) -> dict[str, str]:
    path = Path(path).resolve()
    return {
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
    }


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def software_versions() -> dict[str, str | None]:
    packages = ["pandas", "pyarrow", "optuna", "botorch", "torch", "toml"]
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        versions["rustc"] = subprocess.run(
            ["rustc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        versions["rustc"] = None
    try:
        versions["simulator_package"] = str(
            toml.load(REPO_ROOT / "v2" / "Cargo.toml")["package"]["version"]
        )
    except (OSError, KeyError, toml.TomlDecodeError):
        versions["simulator_package"] = None
    return versions


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_runtime_config(
    path: str | Path, *, simulator_seed: int, iteration_count: int
) -> Path:
    simulator_seed = validate_nonnegative_integer(simulator_seed, "simulator_seed")
    iteration_count = validate_positive_integer(iteration_count, "iteration_count")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        toml.dump(
            {"seed_state": simulator_seed, "iteration_count": iteration_count},
            handle,
        )
    return path


def _validate_parameter(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExperimentConfigurationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.5 <= value <= 1.0:
        raise ExperimentConfigurationError(f"{name} must be in [0.5, 1.0]")
    return value


def parameter_condition_id(certainty: float, effectiveness: float) -> str:
    certainty = _validate_parameter(certainty, "certainty")
    effectiveness = _validate_parameter(effectiveness, "effectiveness")

    def encode(value: float) -> str:
        return f"{round(value, 4):.4f}".replace(".", "p")

    return f"c{encode(certainty)}_e{encode(effectiveness)}"


def read_intervention_opinion_csv(
    path: str | Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate one complete intervention opinion and extract its design values."""

    path = Path(path).resolve()
    if not path.is_file():
        raise ExperimentConfigurationError(
            f"intervention opinion CSV does not exist: {path}"
        )
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ExperimentConfigurationError(
            f"failed to read intervention opinion CSV: {path}"
        ) from exc
    if len(frame) != 1:
        raise ExperimentConfigurationError(
            "intervention opinion CSV must contain exactly one data row"
        )
    if tuple(frame.columns) != INTERVENTION_OPINION_COLUMNS:
        raise ExperimentConfigurationError(
            "intervention opinion CSV columns do not match the simulator schema"
        )

    values: dict[str, float] = {}
    for column in INTERVENTION_OPINION_COLUMNS:
        try:
            value = float(frame.iloc[0][column])
        except (TypeError, ValueError) as exc:
            raise ExperimentConfigurationError(
                f"intervention opinion value is not numeric: {column}"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ExperimentConfigurationError(
                f"intervention opinion value must be in [0, 1]: {column}"
            )
        values[column] = value

    normalized_groups = (
        ("phi_b0", "phi_b1", "phi_u"),
        ("phi_a0", "phi_a1"),
        ("psi0_b0", "psi0_b1", "psi0_u"),
        ("psi1_b0", "psi1_b1", "psi1_u"),
        ("b0_b0", "b0_b1", "b0_u"),
        ("b1_b0", "b1_b1", "b1_u"),
    )
    for columns in normalized_groups:
        total = sum(values[column] for column in columns)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ExperimentConfigurationError(
                "intervention opinion components must sum to 1: "
                + ", ".join(columns)
            )
    if not math.isclose(
        values["psi1_b0"], values["b1_b0"], rel_tol=0.0, abs_tol=1e-9
    ):
        raise ExperimentConfigurationError(
            "psi1_b0 and b1_b0 must encode the same effectiveness"
        )

    applied = {
        "certainty": _validate_parameter(values["phi_b1"], "certainty"),
        "effectiveness": _validate_parameter(
            values["psi1_b0"], "effectiveness"
        ),
    }
    return values, applied


def copy_intervention_opinion_csv(
    source: str | Path, destination: str | Path
) -> tuple[dict[str, float], dict[str, float]]:
    """Copy a validated fixed opinion without regenerating any of its fields."""

    source = Path(source).resolve()
    values, applied = read_intervention_opinion_csv(source)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise OSError("copied intervention opinion CSV failed SHA-256 verification")
    return values, applied


def write_intervention_opinion_csv(
    path: str | Path, *, certainty: float, effectiveness: float
) -> dict[str, float]:
    proposed_certainty = _validate_parameter(certainty, "certainty")
    proposed_effectiveness = _validate_parameter(effectiveness, "effectiveness")
    applied_certainty = round(proposed_certainty, 4)
    applied_effectiveness = round(proposed_effectiveness, 4)
    certainty_uncertainty = round(1.0 - applied_certainty, 4)
    effectiveness_uncertainty = round(1.0 - applied_effectiveness, 4)

    row = {
        "phi_b0": 0.0,
        "phi_b1": applied_certainty,
        "phi_u": certainty_uncertainty,
        "phi_a0": 0.5,
        "phi_a1": 0.5,
        "psi0_b0": 0.5,
        "psi0_b1": 0.0,
        "psi0_u": 0.5,
        "psi1_b0": applied_effectiveness,
        "psi1_b1": 0.0,
        "psi1_u": effectiveness_uncertainty,
        "b0_b0": 0.5,
        "b0_b1": 0.0,
        "b0_u": 0.5,
        "b1_b0": applied_effectiveness,
        "b1_b1": 0.0,
        "b1_u": effectiveness_uncertainty,
    }

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    return {
        "certainty": applied_certainty,
        "effectiveness": applied_effectiveness,
    }


def _resolve_template_path(value: str, template_path: Path) -> str:
    return (template_path.parent / value).resolve().as_posix()


def write_strategy_config(
    path: str | Path,
    *,
    intervention_opinion_csv: str | Path | None,
    template_path: str | Path = STRATEGY_TEMPLATE,
) -> Path:
    template_path = Path(template_path).resolve()
    config = toml.load(template_path)

    if "informing" in config:
        config["informing"] = _resolve_template_path(config["informing"], template_path)

    if "information" not in config:
        raise ExperimentConfigurationError("strategy template lacks [information]")
    for key, value in config["information"].items():
        if key == "inhibition" and intervention_opinion_csv is not None:
            config["information"][key] = Path(
                intervention_opinion_csv
            ).resolve().as_posix()
        else:
            config["information"][key] = _resolve_template_path(value, template_path)

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        toml.dump(config, handle)
    return path


def build_simulator_command(
    *,
    identifier: str,
    output_dir: str | Path,
    runtime_path: str | Path,
    network_path: str | Path,
    agent_path: str | Path,
    strategy_path: str | Path,
    intervention_enabled: bool,
) -> list[str]:
    command = [
        str(RUST_BINARY.resolve()),
        identifier,
        str(Path(output_dir).resolve()),
        "--runtime",
        str(Path(runtime_path).resolve()),
        "--network",
        str(Path(network_path).resolve()),
        "--agent",
        str(Path(agent_path).resolve()),
        "--strategy",
        str(Path(strategy_path).resolve()),
    ]
    if intervention_enabled:
        command.append("-e")
    command.extend(["-d", "0"])
    return command


def _normalise_arrow_outputs(
    output_dir: Path, identifier: str
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for kind in ("pop", "info", "agent"):
        source = output_dir / f"{identifier}_{kind}.arrow"
        destination = output_dir / f"{kind}.arrow"
        if not source.exists():
            missing.append(source.name)
            continue
        if destination.exists():
            raise SimulationExecutionError(
                f"normalized raw output already exists: {destination}",
                stage="raw_output",
            )
        source.replace(destination)
        paths[kind] = destination

    if missing:
        raise SimulationExecutionError(
            f"simulator completed but raw outputs are missing: {missing}",
            stage="raw_output",
        )
    return paths


def run_simulator(
    *,
    identifier: str,
    output_dir: str | Path,
    runtime_path: str | Path,
    network_path: str | Path,
    strategy_path: str | Path,
    intervention_enabled: bool,
    stdout_path: str | Path,
    stderr_path: str | Path,
) -> SimulationRunResult:
    if not RUST_BINARY.exists():
        raise SimulationExecutionError(
            f"Rust binary not found: {RUST_BINARY}", stage="preflight"
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(stdout_path).resolve()
    stderr_path = Path(stderr_path).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_simulator_command(
        identifier=identifier,
        output_dir=output_dir,
        runtime_path=runtime_path,
        network_path=network_path,
        agent_path=AGENT_CONFIG,
        strategy_path=strategy_path,
        intervention_enabled=intervention_enabled,
    )

    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    elapsed = time.perf_counter() - start

    if completed.returncode != 0:
        raise SimulationExecutionError(
            f"simulator exited with code {completed.returncode}",
            stage="simulation",
            exit_code=completed.returncode,
        )

    arrow_paths = _normalise_arrow_outputs(output_dir, identifier)
    return SimulationRunResult(
        command=command,
        elapsed_sec=elapsed,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        arrow_paths=arrow_paths,
    )


def remove_unrequested_raw(
    arrow_paths: dict[str, Path], raw_level: str
) -> dict[str, Path]:
    keep_by_level = {
        "pop": {"pop"},
        "info_pop": {"pop", "info"},
        "all": {"pop", "info", "agent"},
    }
    if raw_level not in keep_by_level:
        raise ExperimentConfigurationError(
            "raw_level must be one of: pop, info_pop, all"
        )
    keep = keep_by_level[raw_level]
    retained: dict[str, Path] = {}
    for kind, path in arrow_paths.items():
        if kind in keep:
            retained[kind] = path
        else:
            path.unlink(missing_ok=True)
    return retained


def append_study_manifest(
    experiment_root: str | Path,
    *,
    experiment_id: str,
    stage: str,
    run_dir: str | Path,
    status: str,
) -> None:
    experiment_root = Path(experiment_root).resolve()
    manifest_path = experiment_root / "study_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "stage": stage,
            "created_at": now_iso(),
            "runs": [],
        }

    relative_run = Path(run_dir).resolve().relative_to(experiment_root).as_posix()
    existing = [entry for entry in manifest["runs"] if entry["path"] == relative_run]
    if existing:
        existing[0]["status"] = status
        existing[0]["updated_at"] = now_iso()
    else:
        manifest["runs"].append(
            {"path": relative_run, "status": status, "updated_at": now_iso()}
        )
    write_json(manifest_path, manifest)


def command_for_manifest() -> list[str]:
    return [sys.executable, *sys.argv]
