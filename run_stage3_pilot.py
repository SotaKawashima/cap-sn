"""Orchestrate the two protocol-defined Stage 3 pilot phases."""

from __future__ import annotations

import argparse
import json
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
    validate_positive_integer,
    write_json,
)


DEFAULT_PROTOCOL_PATH = REPO_ROOT / "experiment_protocols" / "stage3_pilot_v3.json"
STAGE = "stage3_pilot"
PHASES = ("precision", "optimization_budget")


@dataclass(frozen=True)
class PilotRunSpec:
    key: str
    phase: str
    run_type: str
    relative_run_dir: str
    network: str
    condition_id: str | None = None
    intervention_enabled: bool | None = None
    certainty: float | None = None
    effectiveness: float | None = None
    simulator_seed: int | None = None
    iterations: int | None = None
    method: str | None = None
    optimizer_replicate: int | None = None
    optimizer_seed: int | None = None
    trials: int | None = None
    startup_trials: int | None = None
    raw_level: str = "pop"


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("schema_version") != 1:
        raise ExperimentConfigurationError("unsupported Stage 3 protocol schema")
    if protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError("protocol stage must be stage3_pilot")
    if "precision_phase" not in protocol or "optimization_budget_phase" not in protocol:
        raise ExperimentConfigurationError("protocol is missing a pilot phase")
    return protocol


def _condition_parameters(
    condition: dict[str, Any], network: str
) -> tuple[float | None, float | None]:
    if not condition["enabled"]:
        return None, None
    if "parameters_by_network" in condition:
        parameters = condition["parameters_by_network"].get(network)
        if parameters is None:
            raise ExperimentConfigurationError(
                f"condition {condition['id']} has no parameters for {network}"
            )
        return float(parameters["certainty"]), float(parameters["effectiveness"])
    return float(condition["certainty"]), float(condition["effectiveness"])


def build_precision_specs(protocol: dict[str, Any]) -> list[PilotRunSpec]:
    phase = protocol["precision_phase"]
    iterations = validate_positive_integer(
        phase["max_iterations_per_block"], "max_iterations_per_block"
    )
    conditions = phase["conditions"]
    condition_ids = [condition["id"] for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ExperimentConfigurationError("precision condition IDs must be unique")

    specs: list[PilotRunSpec] = []
    for network in phase["networks"]:
        if network not in NETWORKS:
            raise ExperimentConfigurationError(f"unsupported network: {network}")
        for condition in conditions:
            certainty, effectiveness = _condition_parameters(condition, network)
            for seed in phase["simulator_seeds"]:
                seed = int(seed)
                condition_id = condition["id"]
                specs.append(
                    PilotRunSpec(
                        key=f"{network}:{condition_id}:simseed{seed}",
                        phase="precision",
                        run_type="fixed_condition",
                        relative_run_dir=(
                            f"{network}/{condition_id}/simseed_{seed}"
                        ),
                        network=network,
                        condition_id=condition_id,
                        intervention_enabled=bool(condition["enabled"]),
                        certainty=certainty,
                        effectiveness=effectiveness,
                        simulator_seed=seed,
                        iterations=iterations,
                        raw_level=phase["raw_level"],
                    )
                )
    return specs


def build_optimization_specs(
    protocol: dict[str, Any], *, iterations: int
) -> list[PilotRunSpec]:
    iterations = validate_positive_integer(iterations, "iterations")
    phase = protocol["optimization_budget_phase"]
    precision_prefixes = set(protocol["precision_phase"]["prefix_iterations"])
    if iterations not in precision_prefixes:
        raise ExperimentConfigurationError(
            "optimization pilot iterations must be one of the precision prefixes"
        )
    protocol_iterations = phase.get("iterations")
    if isinstance(protocol_iterations, int) and iterations != protocol_iterations:
        raise ExperimentConfigurationError(
            "optimization pilot iterations do not match the protocol-selected value"
        )

    # v2 remains readable so its BA1000-only dry-run can be reproduced.
    networks = phase.get("networks")
    if networks is None:
        networks = [phase["network"]]

    specs: list[PilotRunSpec] = []
    for network in networks:
        if network not in NETWORKS:
            raise ExperimentConfigurationError(f"unsupported network: {network}")
        for method in phase["methods"]:
            seeds = phase["optimizer_seeds"].get(method)
            replicates = phase["optimizer_replicates"]
            if seeds is None or len(seeds) != len(replicates):
                raise ExperimentConfigurationError(
                    f"optimizer seed count does not match replicates for {method}"
                )
            for replicate, optimizer_seed in zip(replicates, seeds, strict=True):
                specs.append(
                    PilotRunSpec(
                        key=f"{network}:{method}:optseed{replicate}",
                        phase="optimization_budget",
                        run_type="single_objective_optimization",
                        relative_run_dir=(
                            f"{network}/{method}/optseed_{int(replicate)}"
                        ),
                        network=network,
                        simulator_seed=int(phase["simulator_seed"]),
                        iterations=iterations,
                        method=method,
                        optimizer_replicate=int(replicate),
                        optimizer_seed=int(optimizer_seed),
                        trials=int(phase["max_trials"]),
                        startup_trials=int(phase["startup_trials"]),
                        raw_level=phase["raw_level"],
                    )
                )
    return specs


def command_for_spec(
    spec: PilotRunSpec,
    *,
    experiment_id: str,
    output_root: str | Path = SUMMER_EXPERIMENT_ROOT,
) -> list[str]:
    resolved_output_root = resolve_output_root(output_root)
    if spec.run_type == "fixed_condition":
        command = [
            sys.executable,
            str(REPO_ROOT / "run_fixed_condition.py"),
            "--stage",
            STAGE,
            "--experiment-id",
            experiment_id,
            "--network",
            spec.network,
            "--condition-id",
            str(spec.condition_id),
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

    if spec.run_type == "single_objective_optimization":
        return [
            sys.executable,
            str(REPO_ROOT / "optimize_single_objective.py"),
            "--stage",
            STAGE,
            "--experiment-id",
            experiment_id,
            "--network",
            spec.network,
            "--method",
            str(spec.method),
            "--optimizer-replicate",
            str(spec.optimizer_replicate),
            "--optimizer-seed",
            str(spec.optimizer_seed),
            "--simulator-seed",
            str(spec.simulator_seed),
            "--iterations",
            str(spec.iterations),
            "--trials",
            str(spec.trials),
            "--startup-trials",
            str(spec.startup_trials),
            "--raw-level",
            spec.raw_level,
            "--output-root",
            str(resolved_output_root),
        ]
    raise ExperimentConfigurationError(f"unsupported run type: {spec.run_type}")


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
    phase: str,
    protocol_path: Path,
    specs: Sequence[PilotRunSpec],
    output_root: Path,
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
        experiment_root / "pilot_execution_plan.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "stage": STAGE,
            "phase": phase,
            "updated_at": now_iso(),
            "protocol": {
                "path": protocol_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "git": git_state(),
            "counts": {
                "total": len(rows),
                "completed": sum(row["status"] == "completed" for row in rows),
                "pending": sum(row["status"] == "pending" for row in rows),
            },
            "runs": rows,
        },
    )


def _filter_specs(
    specs: Sequence[PilotRunSpec],
    *,
    networks: Sequence[str] | None,
    simulator_seeds: Sequence[int] | None,
    optimizer_replicates: Sequence[int] | None,
) -> list[PilotRunSpec]:
    selected = list(specs)
    if networks:
        selected = [spec for spec in selected if spec.network in networks]
    if simulator_seeds:
        seed_set = set(simulator_seeds)
        selected = [spec for spec in selected if spec.simulator_seed in seed_set]
    if optimizer_replicates:
        replicate_set = set(optimizer_replicates)
        selected = [
            spec
            for spec in selected
            if spec.optimizer_replicate in replicate_set
        ]
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 3 pilot protocol.")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--networks", nargs="+", choices=sorted(NETWORKS))
    parser.add_argument("--simulator-seeds", nargs="+", type=int)
    parser.add_argument("--optimizer-replicates", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_pilot(args: argparse.Namespace) -> Path | None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    if args.phase == "precision":
        if args.iterations is not None:
            raise ExperimentConfigurationError(
                "precision iterations are fixed by the protocol"
            )
        all_specs = build_precision_specs(protocol)
        purpose = "pilot_precision"
    else:
        if args.iterations is None:
            raise ExperimentConfigurationError(
                "optimization_budget requires --iterations from precision analysis"
            )
        all_specs = build_optimization_specs(protocol, iterations=args.iterations)
        purpose = "pilot_optimization_budget"

    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id(purpose)
    )
    selected = _filter_specs(
        all_specs,
        networks=args.networks,
        simulator_seeds=args.simulator_seeds,
        optimizer_replicates=args.optimizer_replicates,
    )
    if not selected:
        raise ExperimentConfigurationError("the pilot selection is empty")
    output_root = resolve_output_root(args.output_root)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "phase": args.phase,
                    "full_run_count": len(all_specs),
                    "selected_run_count": len(selected),
                    "commands": [
                        command_for_spec(
                            spec,
                            experiment_id=experiment_id,
                            output_root=output_root,
                        )
                        for spec in selected
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
    state = git_state()
    if state["dirty"] and not args.allow_dirty:
        raise ExperimentConfigurationError(
            "pilot execution requires a clean Git worktree; "
            "commit the execution code first"
        )

    experiment_root = output_root / STAGE / experiment_id
    if experiment_root.exists():
        if not args.resume:
            raise FileExistsError(
                "pilot experiment already exists; use --resume or a new experiment ID"
            )
        plan_path = experiment_root / "pilot_execution_plan.json"
        if not plan_path.is_file():
            raise ExperimentConfigurationError(
                "existing pilot experiment has no execution plan"
            )
        with plan_path.open("r", encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("phase") != args.phase:
            raise ExperimentConfigurationError(
                "existing pilot phase does not match the requested phase"
            )
        if previous_plan.get("protocol", {}).get("sha256") != sha256_file(
            protocol_path
        ):
            raise ExperimentConfigurationError(
                "protocol changed after this pilot experiment was created"
            )
    else:
        create_unique_run_directory(experiment_root)

    _write_execution_plan(
        experiment_root=experiment_root,
        experiment_id=experiment_id,
        phase=args.phase,
        protocol_path=protocol_path,
        specs=all_specs,
        output_root=output_root,
    )

    for index, spec in enumerate(selected, start=1):
        run_dir = experiment_root / spec.relative_run_dir
        status = _read_status(run_dir)
        if status == "completed":
            print(f"[{index}/{len(selected)}] skip completed: {spec.key}")
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
        print(f"[{index}/{len(selected)}] run: {spec.key}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        _write_execution_plan(
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            phase=args.phase,
            protocol_path=protocol_path,
            specs=all_specs,
            output_root=output_root,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"pilot child process failed for {spec.key} with code "
                f"{completed.returncode}"
            )

    return experiment_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_root = run_pilot(args)
    except (
        ExperimentConfigurationError,
        FileExistsError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if experiment_root is not None:
        print(f"Completed selected Stage 3 pilot runs: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
