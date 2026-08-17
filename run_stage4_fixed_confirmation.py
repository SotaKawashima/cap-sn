"""Run the protocol-defined Stage 4 fixed confirmation experiment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

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
    REPO_ROOT / "experiment_protocols" / "stage4_fixed_confirmation_v1.json"
)
STAGE = "stage4_fixed_confirmation"
EXECUTION_PLAN_NAME = "fixed_confirmation_execution_plan.json"


@dataclass(frozen=True)
class FixedConfirmationRunSpec:
    key: str
    relative_run_dir: str
    network: str
    condition_id: str
    condition_role: str
    intervention_enabled: bool
    certainty: float | None
    effectiveness: float | None
    simulator_seed: int
    iterations: int
    raw_level: str


def _validate_parameter(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentConfigurationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.5 <= value <= 1.0:
        raise ExperimentConfigurationError(f"{name} must be in [0.5, 1.0]")
    return value


def _comparison_condition_ids(protocol: dict[str, Any]) -> set[str]:
    condition_ids: set[str] = set()
    for comparison in protocol.get("primary_comparisons", []):
        condition_ids.update(
            [str(comparison["reference"]), str(comparison["candidate"])]
        )
    factorial = protocol.get("factorial_contrasts", {})
    for group_name in ("effectiveness_low_to_high", "certainty_low_to_high"):
        for comparison in factorial.get(group_name, []):
            condition_ids.update(
                [str(comparison["reference"]), str(comparison["candidate"])]
            )
    interaction = factorial.get("corner_interaction", {})
    for key in (
        "effectiveness_contrast_at_low_certainty",
        "effectiveness_contrast_at_high_certainty",
    ):
        comparison = interaction.get(key)
        if comparison:
            condition_ids.update(
                [str(comparison["reference"]), str(comparison["candidate"])]
            )
    return condition_ids


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ExperimentConfigurationError("unsupported Stage 4 protocol schema")
    if protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError(
            "protocol stage must be stage4_fixed_confirmation"
        )

    design = protocol.get("design")
    if not isinstance(design, dict):
        raise ExperimentConfigurationError("protocol is missing the Stage 4 design")

    networks = design.get("networks", [])
    if not networks or len(networks) != len(set(networks)):
        raise ExperimentConfigurationError(
            "Stage 4 networks must be a non-empty unique list"
        )
    unsupported_networks = set(networks) - set(NETWORKS)
    if unsupported_networks:
        raise ExperimentConfigurationError(
            f"unsupported networks: {sorted(unsupported_networks)}"
        )

    seeds = [
        validate_nonnegative_integer(seed, "simulator_seed")
        for seed in design.get("simulator_seeds", [])
    ]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ExperimentConfigurationError(
            "Stage 4 simulator seeds must be a non-empty unique list"
        )

    validate_positive_integer(
        design.get("iterations_per_seed_block"), "iterations_per_seed_block"
    )
    if design.get("raw_level") not in {"pop", "info_pop", "all"}:
        raise ExperimentConfigurationError(
            "Stage 4 raw_level must be one of: pop, info_pop, all"
        )

    conditions = design.get("conditions", [])
    if not conditions:
        raise ExperimentConfigurationError("Stage 4 conditions must not be empty")
    condition_ids = [
        validate_safe_name(str(condition.get("id", "")), "condition_id")
        for condition in conditions
    ]
    if len(condition_ids) != len(set(condition_ids)):
        raise ExperimentConfigurationError("Stage 4 condition IDs must be unique")

    parameter_pairs: set[tuple[float, float]] = set()
    disabled_ids: list[str] = []
    for condition in conditions:
        condition_id = str(condition["id"])
        enabled = condition.get("enabled")
        if not isinstance(enabled, bool):
            raise ExperimentConfigurationError(
                f"condition {condition_id} must have a boolean enabled value"
            )
        validate_safe_name(str(condition.get("role", "")), "condition_role")
        if not enabled:
            disabled_ids.append(condition_id)
            if condition.get("certainty") is not None or condition.get(
                "effectiveness"
            ) is not None:
                raise ExperimentConfigurationError(
                    "the no-intervention condition must have null parameters"
                )
            continue
        pair = (
            _validate_parameter(
                condition.get("certainty"), f"{condition_id}.certainty"
            ),
            _validate_parameter(
                condition.get("effectiveness"), f"{condition_id}.effectiveness"
            ),
        )
        if pair in parameter_pairs:
            raise ExperimentConfigurationError(
                f"duplicate Stage 4 intervention parameters: {pair}"
            )
        parameter_pairs.add(pair)
    if disabled_ids != ["none"]:
        raise ExperimentConfigurationError(
            "Stage 4 must contain exactly one disabled condition named 'none'"
        )

    referenced_ids = _comparison_condition_ids(protocol)
    unknown_references = referenced_ids - set(condition_ids)
    if unknown_references:
        raise ExperimentConfigurationError(
            "comparisons reference unknown conditions: "
            f"{sorted(unknown_references)}"
        )

    expected_conditions = design.get("expected_condition_count_per_network")
    if expected_conditions != len(conditions):
        raise ExperimentConfigurationError(
            "expected_condition_count_per_network does not match the design"
        )
    expected_runs = len(networks) * len(conditions) * len(seeds)
    if design.get("expected_run_count") != expected_runs:
        raise ExperimentConfigurationError(
            "expected_run_count does not match networks x conditions x seeds"
        )
    expected_iterations = expected_runs * design["iterations_per_seed_block"]
    if design.get("expected_simulation_iterations") != expected_iterations:
        raise ExperimentConfigurationError(
            "expected_simulation_iterations does not match the design"
        )

    future_groups = protocol.get("seed_policy", {}).get(
        "reserved_future_groups", {}
    )
    stage4_seeds = set(seeds)
    seen_future: set[int] = set()
    for group_name, values in future_groups.items():
        group = {
            validate_nonnegative_integer(value, f"{group_name}_simulator_seed")
            for value in values
        }
        if stage4_seeds & group:
            raise ExperimentConfigurationError(
                f"Stage 4 seeds overlap with reserved group {group_name}"
            )
        if seen_future & group:
            raise ExperimentConfigurationError(
                f"reserved seed groups overlap at {group_name}"
            )
        seen_future.update(group)


def load_protocol(
    path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    _validate_protocol(protocol)
    return protocol


def build_specs(protocol: dict[str, Any]) -> list[FixedConfirmationRunSpec]:
    _validate_protocol(protocol)
    design = protocol["design"]
    iterations = int(design["iterations_per_seed_block"])
    raw_level = str(design["raw_level"])
    specs: list[FixedConfirmationRunSpec] = []

    for network in design["networks"]:
        for condition in design["conditions"]:
            enabled = bool(condition["enabled"])
            certainty = float(condition["certainty"]) if enabled else None
            effectiveness = (
                float(condition["effectiveness"]) if enabled else None
            )
            for seed_value in design["simulator_seeds"]:
                seed = int(seed_value)
                condition_id = str(condition["id"])
                specs.append(
                    FixedConfirmationRunSpec(
                        key=f"{network}:{condition_id}:simseed{seed}",
                        relative_run_dir=(
                            f"{network}/{condition_id}/simseed_{seed}"
                        ),
                        network=str(network),
                        condition_id=condition_id,
                        condition_role=str(condition["role"]),
                        intervention_enabled=enabled,
                        certainty=certainty,
                        effectiveness=effectiveness,
                        simulator_seed=seed,
                        iterations=iterations,
                        raw_level=raw_level,
                    )
                )
    return specs


def command_for_spec(
    spec: FixedConfirmationRunSpec,
    *,
    experiment_id: str,
    output_root: str | Path = SUMMER_EXPERIMENT_ROOT,
) -> list[str]:
    resolved_output_root = resolve_output_root(output_root)
    command = [
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
        "--simulator-seed",
        str(spec.simulator_seed),
        "--iterations",
        str(spec.iterations),
        "--raw-level",
        spec.raw_level,
        "--output-root",
        str(resolved_output_root),
    ]
    if spec.intervention_enabled:
        command.extend(
            [
                "--certainty",
                str(spec.certainty),
                "--effectiveness",
                str(spec.effectiveness),
            ]
        )
    else:
        command.append("--no-intervention")
    return command


def _read_status(run_dir: Path) -> str:
    if not run_dir.exists():
        return "pending"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return "invalid_existing_directory"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return str(json.load(handle).get("status", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "invalid_manifest"


def _write_execution_plan(
    *,
    experiment_root: Path,
    experiment_id: str,
    protocol_path: Path,
    specs: Sequence[FixedConfirmationRunSpec],
    output_root: Path,
    initial_git_state: dict[str, Any],
) -> None:
    rows = []
    for spec in specs:
        run_dir = experiment_root / spec.relative_run_dir
        rows.append(
            {
                **asdict(spec),
                "status": _read_status(run_dir),
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
    specs: Sequence[FixedConfirmationRunSpec],
    *,
    networks: Sequence[str] | None,
    conditions: Sequence[str] | None,
    simulator_seeds: Sequence[int] | None,
) -> list[FixedConfirmationRunSpec]:
    known_networks = {spec.network for spec in specs}
    known_conditions = {spec.condition_id for spec in specs}
    known_seeds = {spec.simulator_seed for spec in specs}
    requested_networks = set(networks or known_networks)
    requested_conditions = set(conditions or known_conditions)
    requested_seeds = set(simulator_seeds or known_seeds)

    if requested_networks - known_networks:
        raise ExperimentConfigurationError("requested network is not in protocol")
    if requested_conditions - known_conditions:
        raise ExperimentConfigurationError("requested condition is not in protocol")
    if requested_seeds - known_seeds:
        raise ExperimentConfigurationError(
            "requested simulator seed is not in protocol"
        )

    return [
        spec
        for spec in specs
        if spec.network in requested_networks
        and spec.condition_id in requested_conditions
        and spec.simulator_seed in requested_seeds
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 4 fixed confirmation protocol."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--networks", nargs="+", choices=sorted(NETWORKS))
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--simulator-seeds", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_fixed_confirmation(args: argparse.Namespace) -> Path | None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    all_specs = build_specs(protocol)
    selected_specs = _filter_specs(
        all_specs,
        networks=args.networks,
        conditions=args.conditions,
        simulator_seeds=args.simulator_seeds,
    )
    if not selected_specs:
        raise ExperimentConfigurationError("the Stage 4 selection is empty")

    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id("fixed_confirmation")
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
            "Stage 4 execution requires a clean Git worktree; "
            "commit the execution code first"
        )

    experiment_root = output_root / STAGE / experiment_id
    if experiment_root.exists():
        if not args.resume:
            raise FileExistsError(
                "Stage 4 experiment already exists; use --resume or a new "
                "experiment ID"
            )
        plan_path = experiment_root / EXECUTION_PLAN_NAME
        if not plan_path.is_file():
            raise ExperimentConfigurationError(
                "existing Stage 4 experiment has no execution plan"
            )
        with plan_path.open("r", encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("stage") != STAGE:
            raise ExperimentConfigurationError(
                "existing experiment stage does not match Stage 4"
            )
        if previous_plan.get("protocol", {}).get("sha256") != sha256_file(
            protocol_path
        ):
            raise ExperimentConfigurationError(
                "protocol changed after this Stage 4 experiment was created"
            )
        initial_git_state = previous_plan.get("git", {})
        initial_commit = initial_git_state.get("commit")
        if initial_commit != current_git_state.get("commit"):
            raise ExperimentConfigurationError(
                "Git commit changed after this Stage 4 experiment was created"
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
        run_dir = experiment_root / spec.relative_run_dir
        status = _read_status(run_dir)
        if status == "completed":
            print(f"[{index}/{len(selected_specs)}] skip completed: {spec.key}")
            continue
        if status != "pending":
            raise RuntimeError(
                f"cannot resume {spec.key} from existing status '{status}'"
            )
        command = command_for_spec(
            spec,
            experiment_id=experiment_id,
            output_root=output_root,
        )
        print(f"[{index}/{len(selected_specs)}] run: {spec.key}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
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
                f"Stage 4 child process failed for {spec.key} with code "
                f"{completed.returncode}"
            )

    return experiment_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_root = run_fixed_confirmation(args)
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
        print(f"Completed selected Stage 4 runs: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
