"""Run the frozen Stage 10 information-diffusion confirmation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from analysis.optimization_metrics import (
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
)
from experiment_runtime import (
    NETWORKS,
    REPO_ROOT,
    RUST_BINARY,
    SUMMER_EXPERIMENT_ROOT,
    ExperimentConfigurationError,
    create_unique_run_directory,
    git_state,
    make_experiment_id,
    now_iso,
    resolve_output_root,
    sha256_file,
    validate_experiment_id,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_safe_name,
    write_json,
)


DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT / "experiment_protocols" / "stage10_information_diffusion_v1.json"
)
STAGE = "stage10_information_diffusion"
EXECUTION_PLAN_NAME = "information_diffusion_execution_plan.json"
EXPECTED_NETWORKS = {"ba1000", "facebook", "wiki_vote"}


@dataclass(frozen=True)
class InformationDiffusionRunSpec:
    key: str
    relative_run_dir: str
    network: str
    condition_id: str
    condition_role: str
    reporting_role: str
    selection_mode: str
    qualified: bool
    certainty: float
    effectiveness: float
    simulator_seed: int
    iterations: int
    raw_level: str


def _repo_path(value: str | Path, field_name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ExperimentConfigurationError(
            f"{field_name} must be inside the repository"
        ) from exc
    return path


def _validate_hash(path: Path, expected: Any, field_name: str) -> str:
    value = str(expected).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExperimentConfigurationError(f"{field_name} must be a SHA-256 hash")
    if not path.is_file():
        raise ExperimentConfigurationError(f"input file is missing: {path}")
    if sha256_file(path) != value:
        raise ExperimentConfigurationError(f"{field_name} does not match {path}")
    return value


def _design_parameter(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ExperimentConfigurationError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigurationError(
            f"{field_name} must be numeric"
        ) from exc
    if not math.isfinite(result) or not 0.5 <= result <= 1.0:
        raise ExperimentConfigurationError(f"{field_name} must be in [0.5, 1.0]")
    return result


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ExperimentConfigurationError(f"{field_name} must be boolean")


def _load_primary_candidates(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    source = protocol.get("additional_experiment", {}).get("candidate_source", {})
    path = _repo_path(source.get("path", ""), "candidate_source.path")
    _validate_hash(path, source.get("sha256"), "candidate_source.sha256")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["selection_order"]) == 1]
    if len(selected) != int(source.get("expected_rows", -1)):
        raise ExperimentConfigurationError(
            "Stage 10 primary-candidate count does not match the protocol"
        )
    networks = [str(row["network"]) for row in selected]
    if set(networks) != EXPECTED_NETWORKS or len(networks) != len(set(networks)):
        raise ExperimentConfigurationError(
            "Stage 10 requires one primary candidate for each network"
        )
    return selected


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ExperimentConfigurationError("unsupported Stage 10 protocol schema")
    if protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError(f"protocol stage must be {STAGE}")
    objective = protocol.get("objective", {})
    if objective.get("name") != OBJECTIVE_NAME or objective.get(
        "definition_version"
    ) != OBJECTIVE_DEFINITION_VERSION:
        raise ExperimentConfigurationError("Stage 10 objective definition is stale")

    for index, source in enumerate(protocol.get("decision_sources", [])):
        if not isinstance(source, dict):
            raise ExperimentConfigurationError(
                f"decision_sources[{index}] must be an object"
            )
        path = _repo_path(source.get("path", ""), f"decision_sources[{index}].path")
        required = bool(source.get("required_for_execution", False))
        if required or path.is_file():
            _validate_hash(
                path,
                source.get("sha256"),
                f"decision_sources[{index}].sha256",
            )

    design = protocol.get("additional_experiment")
    if not isinstance(design, dict):
        raise ExperimentConfigurationError(
            "protocol is missing the Stage 10 additional experiment"
        )
    networks = [str(value) for value in design.get("networks", [])]
    if set(networks) != EXPECTED_NETWORKS or len(networks) != len(set(networks)):
        raise ExperimentConfigurationError("Stage 10 network list is invalid")
    seeds = [
        validate_nonnegative_integer(value, "simulator_seed")
        for value in design.get("simulator_seeds", [])
    ]
    if len(seeds) != 5 or len(seeds) != len(set(seeds)):
        raise ExperimentConfigurationError(
            "Stage 10 requires five unique simulator seeds"
        )
    iterations = validate_positive_integer(
        design.get("iterations_per_seed_block"), "iterations_per_seed_block"
    )
    if design.get("raw_level") != "info_pop":
        raise ExperimentConfigurationError("Stage 10 raw_level must be info_pop")
    candidates = _load_primary_candidates(protocol)
    if {row["network"] for row in candidates} != set(networks):
        raise ExperimentConfigurationError(
            "candidate networks do not match the Stage 10 design"
        )
    expected_runs = len(candidates) * len(seeds)
    if expected_runs != int(design.get("expected_run_count", -1)):
        raise ExperimentConfigurationError(
            "Stage 10 expected run count is inconsistent"
        )
    if expected_runs * iterations != int(
        design.get("expected_simulation_iterations", -1)
    ):
        raise ExperimentConfigurationError(
            "Stage 10 expected simulation iterations are inconsistent"
        )


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    _validate_protocol(protocol)
    return protocol


def build_specs(protocol: dict[str, Any]) -> list[InformationDiffusionRunSpec]:
    _validate_protocol(protocol)
    design = protocol["additional_experiment"]
    specs: list[InformationDiffusionRunSpec] = []
    for row in _load_primary_candidates(protocol):
        network = str(row["network"])
        condition_id = validate_safe_name(str(row["condition_id"]), "condition_id")
        certainty = _design_parameter(row["certainty"], "certainty")
        effectiveness = _design_parameter(row["effectiveness"], "effectiveness")
        reporting_role = validate_safe_name(
            str(row["reporting_role"]), "reporting_role"
        )
        selection_mode = validate_safe_name(
            str(row["selection_mode"]), "selection_mode"
        )
        qualified = _parse_bool(row["qualified"], "qualified")
        for seed_value in design["simulator_seeds"]:
            seed = int(seed_value)
            specs.append(
                InformationDiffusionRunSpec(
                    key=f"{network}:{condition_id}:simseed{seed}",
                    relative_run_dir=f"{network}/{condition_id}/simseed_{seed}",
                    network=network,
                    condition_id=condition_id,
                    condition_role="frozen_primary_candidate",
                    reporting_role=reporting_role,
                    selection_mode=selection_mode,
                    qualified=qualified,
                    certainty=certainty,
                    effectiveness=effectiveness,
                    simulator_seed=seed,
                    iterations=int(design["iterations_per_seed_block"]),
                    raw_level=str(design["raw_level"]),
                )
            )
    return specs


def command_for_spec(
    spec: InformationDiffusionRunSpec,
    *,
    experiment_id: str,
    output_root: str | Path = SUMMER_EXPERIMENT_ROOT,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "run_fixed_condition.py"),
        "--stage",
        STAGE,
        "--experiment-id",
        experiment_id,
        "--purpose",
        STAGE,
        "--network",
        spec.network,
        "--condition-id",
        spec.condition_id,
        "--certainty",
        str(spec.certainty),
        "--effectiveness",
        str(spec.effectiveness),
        "--simulator-seed",
        str(spec.simulator_seed),
        "--iterations",
        str(spec.iterations),
        "--raw-level",
        spec.raw_level,
        "--output-root",
        str(resolve_output_root(output_root)),
    ]


def _read_status(run_dir: Path) -> str:
    if not run_dir.exists():
        return "pending"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return "invalid_existing_directory"
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            return str(json.load(handle).get("status", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "invalid_manifest"


def _write_execution_plan(
    *,
    experiment_root: Path,
    experiment_id: str,
    protocol_path: Path,
    specs: Sequence[InformationDiffusionRunSpec],
    output_root: Path,
    initial_git_state: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                **asdict(spec),
                "status": _read_status(experiment_root / spec.relative_run_dir),
                "command": command_for_spec(
                    spec,
                    experiment_id=experiment_id,
                    output_root=output_root,
                ),
            }
        )
    write_json(
        experiment_root / EXECUTION_PLAN_NAME,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "stage": STAGE,
            "updated_at": now_iso(),
            "protocol": {
                "path": protocol_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "git": initial_git_state,
            "counts": {
                "total": len(rows),
                "completed": sum(row["status"] == "completed" for row in rows),
                "pending": sum(row["status"] == "pending" for row in rows),
                "other": sum(
                    row["status"] not in {"completed", "pending"} for row in rows
                ),
            },
            "runs": rows,
        },
    )


def _filter_specs(
    specs: Sequence[InformationDiffusionRunSpec],
    *,
    networks: Sequence[str] | None,
    simulator_seeds: Sequence[int] | None,
) -> list[InformationDiffusionRunSpec]:
    known_networks = {spec.network for spec in specs}
    known_seeds = {spec.simulator_seed for spec in specs}
    requested_networks = set(networks or known_networks)
    requested_seeds = set(simulator_seeds or known_seeds)
    if requested_networks - known_networks:
        raise ExperimentConfigurationError("requested network is not in protocol")
    if requested_seeds - known_seeds:
        raise ExperimentConfigurationError(
            "requested simulator seed is not in protocol"
        )
    return [
        spec
        for spec in specs
        if spec.network in requested_networks
        and spec.simulator_seed in requested_seeds
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Stage 10 information-diffusion protocol."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--networks", nargs="+", choices=sorted(EXPECTED_NETWORKS))
    parser.add_argument("--simulator-seeds", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_information_diffusion(args: argparse.Namespace) -> Path | None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    all_specs = build_specs(protocol)
    selected_specs = _filter_specs(
        all_specs,
        networks=args.networks,
        simulator_seeds=args.simulator_seeds,
    )
    if not selected_specs:
        raise ExperimentConfigurationError("the Stage 10 selection is empty")
    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id("information_diffusion")
    )
    output_root = resolve_output_root(args.output_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "stage": STAGE,
                    "full_run_count": len(all_specs),
                    "selected_run_count": len(selected_specs),
                    "commands": [
                        command_for_spec(
                            spec,
                            experiment_id=experiment_id,
                            output_root=output_root,
                        )
                        for spec in selected_specs
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return None

    if not RUST_BINARY.is_file():
        raise ExperimentConfigurationError(
            f"release simulator binary is missing: {RUST_BINARY}"
        )
    current_git_state = git_state()
    if current_git_state["dirty"] and not args.allow_dirty:
        raise ExperimentConfigurationError(
            "Stage 10 execution requires a clean Git worktree; commit the "
            "execution code first"
        )
    experiment_root = output_root / STAGE / experiment_id
    if experiment_root.exists():
        if not args.resume:
            raise FileExistsError(
                "Stage 10 experiment already exists; use --resume or a new ID"
            )
        plan_path = experiment_root / EXECUTION_PLAN_NAME
        if not plan_path.is_file():
            raise ExperimentConfigurationError(
                "existing Stage 10 experiment has no execution plan"
            )
        with plan_path.open(encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("stage") != STAGE:
            raise ExperimentConfigurationError(
                "existing experiment stage does not match Stage 10"
            )
        if previous_plan.get("protocol", {}).get("sha256") != sha256_file(
            protocol_path
        ):
            raise ExperimentConfigurationError(
                "protocol changed after the Stage 10 experiment was created"
            )
        initial_git_state = previous_plan.get("git", {})
        if initial_git_state.get("commit") != current_git_state.get("commit"):
            raise ExperimentConfigurationError(
                "Git commit changed after the Stage 10 experiment was created"
            )
    else:
        create_unique_run_directory(experiment_root)
        initial_git_state = current_git_state

    _write_execution_plan(
        experiment_root=experiment_root,
        experiment_id=experiment_id,
        protocol_path=protocol_path,
        specs=all_specs,
        output_root=output_root,
        initial_git_state=initial_git_state,
    )
    for index, spec in enumerate(selected_specs, start=1):
        status = _read_status(experiment_root / spec.relative_run_dir)
        if status == "completed":
            print(f"[{index}/{len(selected_specs)}] skip completed: {spec.key}")
            continue
        if status != "pending":
            raise RuntimeError(
                f"cannot resume {spec.key} from existing status {status!r}"
            )
        print(f"[{index}/{len(selected_specs)}] run: {spec.key}", flush=True)
        completed = subprocess.run(
            command_for_spec(
                spec,
                experiment_id=experiment_id,
                output_root=output_root,
            ),
            cwd=REPO_ROOT,
            check=False,
        )
        _write_execution_plan(
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            protocol_path=protocol_path,
            specs=all_specs,
            output_root=output_root,
            initial_git_state=initial_git_state,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Stage 10 child process failed for {spec.key} with code "
                f"{completed.returncode}"
            )
    return experiment_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_root = run_information_diffusion(args)
    except (
        ExperimentConfigurationError,
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if experiment_root is not None:
        print(f"Completed selected Stage 10 runs: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
