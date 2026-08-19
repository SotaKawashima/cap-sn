"""Run the protocol-defined Stage 6 single-objective reoptimization study."""

from __future__ import annotations

import argparse
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
    write_json,
)
from optimize_single_objective import METHODS


DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT / "experiment_protocols" / "stage6_reoptimization_v1.json"
)
STAGE = "stage6_reoptimization"


@dataclass(frozen=True)
class ReoptimizationRunSpec:
    key: str
    relative_run_dir: str
    network: str
    method: str
    optimizer_replicate: int
    optimizer_seed: int
    simulator_seed: int
    iterations: int
    trials: int
    startup_trials: int
    raw_level: str


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)

    if protocol.get("schema_version") != 1:
        raise ExperimentConfigurationError("unsupported Stage 6 protocol schema")
    if protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError(
            "protocol stage must be stage6_reoptimization"
        )
    objective = protocol.get("objective", {})
    if objective.get("name") != OBJECTIVE_NAME:
        raise ExperimentConfigurationError("Stage 6 objective name is inconsistent")
    if objective.get("definition_version") != OBJECTIVE_DEFINITION_VERSION:
        raise ExperimentConfigurationError(
            "Stage 6 objective definition version is inconsistent"
        )
    if objective.get("direction") != "minimize":
        raise ExperimentConfigurationError("Stage 6 objective must be minimized")

    variables = protocol.get("design_variables", {})
    for name in ("certainty", "effectiveness"):
        bounds = variables.get(name, {})
        if not math.isclose(float(bounds.get("lower", math.nan)), 0.5):
            raise ExperimentConfigurationError(f"{name} lower bound must be 0.5")
        if not math.isclose(float(bounds.get("upper", math.nan)), 1.0):
            raise ExperimentConfigurationError(f"{name} upper bound must be 1.0")

    execution = protocol.get("execution")
    if not isinstance(execution, dict):
        raise ExperimentConfigurationError("protocol is missing execution settings")
    _validated_execution_settings(execution)
    return protocol


def _validated_execution_settings(execution: dict[str, Any]) -> None:
    networks = list(execution.get("networks", []))
    methods = list(execution.get("methods", []))
    replicates = [
        validate_positive_integer(value, "optimizer_replicate")
        for value in execution.get("optimizer_replicates", [])
    ]
    if not networks or len(networks) != len(set(networks)):
        raise ExperimentConfigurationError(
            "Stage 6 networks must be non-empty and unique"
        )
    unknown_networks = set(networks) - set(NETWORKS)
    if unknown_networks:
        raise ExperimentConfigurationError(
            f"unsupported Stage 6 networks: {sorted(unknown_networks)}"
        )
    if not methods or len(methods) != len(set(methods)):
        raise ExperimentConfigurationError(
            "Stage 6 methods must be non-empty and unique"
        )
    unknown_methods = set(methods) - set(METHODS)
    if unknown_methods:
        raise ExperimentConfigurationError(
            f"unsupported Stage 6 methods: {sorted(unknown_methods)}"
        )
    if not replicates or len(replicates) != len(set(replicates)):
        raise ExperimentConfigurationError(
            "optimizer replicates must be non-empty and unique"
        )

    optimizer_seeds = execution.get("optimizer_seeds", {})
    if set(optimizer_seeds) != set(methods):
        raise ExperimentConfigurationError(
            "optimizer seed methods do not match Stage 6 methods"
        )
    flattened: list[int] = []
    for method in methods:
        seeds = [
            validate_nonnegative_integer(value, f"{method}_optimizer_seed")
            for value in optimizer_seeds[method]
        ]
        if len(seeds) != len(replicates):
            raise ExperimentConfigurationError(
                f"optimizer seed count does not match replicates for {method}"
            )
        flattened.extend(seeds)
    if len(flattened) != len(set(flattened)):
        raise ExperimentConfigurationError(
            "optimizer seeds must be unique across Stage 6 methods"
        )

    iterations = validate_positive_integer(
        execution.get("iterations_per_evaluation"), "iterations_per_evaluation"
    )
    trials = validate_positive_integer(
        execution.get("evaluations_per_run"), "evaluations_per_run"
    )
    startup = validate_positive_integer(
        execution.get("bo_gp_startup_trials"), "bo_gp_startup_trials"
    )
    if startup > trials:
        raise ExperimentConfigurationError(
            "BO-GP startup trials cannot exceed the evaluation budget"
        )
    validate_nonnegative_integer(execution.get("simulator_seed"), "simulator_seed")
    if execution.get("raw_level") != "pop":
        raise ExperimentConfigurationError("Stage 6 raw_level must be pop")

    expected_runs = len(networks) * len(methods) * len(replicates)
    if int(execution.get("expected_run_count", -1)) != expected_runs:
        raise ExperimentConfigurationError("Stage 6 expected run count is incorrect")
    if int(execution.get("expected_trial_count", -1)) != expected_runs * trials:
        raise ExperimentConfigurationError(
            "Stage 6 expected trial count is incorrect"
        )
    expected_iterations = expected_runs * trials * iterations
    if (
        int(execution.get("expected_simulation_iteration_count", -1))
        != expected_iterations
    ):
        raise ExperimentConfigurationError(
            "Stage 6 expected simulation iteration count is incorrect"
        )


def build_run_specs(protocol: dict[str, Any]) -> list[ReoptimizationRunSpec]:
    execution = protocol["execution"]
    iterations = int(execution["iterations_per_evaluation"])
    trials = int(execution["evaluations_per_run"])
    simulator_seed = int(execution["simulator_seed"])
    startup_trials = int(execution["bo_gp_startup_trials"])
    raw_level = str(execution["raw_level"])
    replicates = [int(value) for value in execution["optimizer_replicates"]]

    specs: list[ReoptimizationRunSpec] = []
    for network in execution["networks"]:
        for method in execution["methods"]:
            seeds = [int(value) for value in execution["optimizer_seeds"][method]]
            for replicate, seed in zip(replicates, seeds, strict=True):
                specs.append(
                    ReoptimizationRunSpec(
                        key=f"{network}:{method}:optseed{replicate}",
                        relative_run_dir=(
                            f"{network}/{method}/optseed_{replicate}"
                        ),
                        network=str(network),
                        method=str(method),
                        optimizer_replicate=replicate,
                        optimizer_seed=seed,
                        simulator_seed=simulator_seed,
                        iterations=iterations,
                        trials=trials,
                        startup_trials=startup_trials,
                        raw_level=raw_level,
                    )
                )
    if len({spec.key for spec in specs}) != len(specs):
        raise ExperimentConfigurationError("Stage 6 run keys are not unique")
    return specs


def command_for_spec(
    spec: ReoptimizationRunSpec,
    *,
    experiment_id: str,
    output_root: str | Path = SUMMER_EXPERIMENT_ROOT,
) -> list[str]:
    resolved_output_root = resolve_output_root(output_root)
    return [
        sys.executable,
        str(REPO_ROOT / "optimize_single_objective.py"),
        "--stage",
        STAGE,
        "--purpose",
        "stage6_reoptimization",
        "--experiment-id",
        experiment_id,
        "--network",
        spec.network,
        "--method",
        spec.method,
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


def inspect_run_status(run_dir: Path, spec: ReoptimizationRunSpec) -> str:
    if not run_dir.exists():
        return "pending"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return "invalid_existing_directory"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return "invalid_manifest"
    status = str(manifest.get("status", "unknown"))
    if status != "completed":
        return status
    counts = manifest.get("counts", {})
    if (
        int(counts.get("complete", -1)) != spec.trials
        or int(counts.get("failed", -1)) != 0
        or int(counts.get("pruned", -1)) != 0
    ):
        return "invalid_completed_budget"
    return "completed"


def _filter_specs(
    specs: Sequence[ReoptimizationRunSpec],
    *,
    networks: Sequence[str] | None,
    methods: Sequence[str] | None,
    optimizer_replicates: Sequence[int] | None,
) -> list[ReoptimizationRunSpec]:
    selected = list(specs)
    if networks:
        selected = [spec for spec in selected if spec.network in set(networks)]
    if methods:
        selected = [spec for spec in selected if spec.method in set(methods)]
    if optimizer_replicates:
        selected = [
            spec
            for spec in selected
            if spec.optimizer_replicate in set(optimizer_replicates)
        ]
    return selected


def _write_execution_plan(
    *,
    experiment_root: Path,
    experiment_id: str,
    protocol_path: Path,
    specs: Sequence[ReoptimizationRunSpec],
    output_root: Path,
) -> None:
    rows = []
    for spec in specs:
        run_dir = experiment_root / spec.relative_run_dir
        rows.append(
            {
                **asdict(spec),
                "status": inspect_run_status(run_dir, spec),
                "command": command_for_spec(
                    spec,
                    experiment_id=experiment_id,
                    output_root=output_root,
                ),
            }
        )
    statuses = [row["status"] for row in rows]
    write_json(
        experiment_root / "reoptimization_execution_plan.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "stage": STAGE,
            "updated_at": now_iso(),
            "protocol": {
                "path": protocol_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "git": git_state(),
            "counts": {
                "total": len(rows),
                "completed": sum(status == "completed" for status in statuses),
                "pending": sum(status == "pending" for status in statuses),
                "other": sum(
                    status not in {"completed", "pending"} for status in statuses
                ),
            },
            "runs": rows,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 6 reoptimization protocol."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--networks", nargs="+", choices=sorted(NETWORKS))
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS))
    parser.add_argument("--optimizer-replicates", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_reoptimization(args: argparse.Namespace) -> Path | None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    all_specs = build_run_specs(protocol)
    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id("stage6_reoptimization")
    )
    selected = _filter_specs(
        all_specs,
        networks=args.networks,
        methods=args.methods,
        optimizer_replicates=args.optimizer_replicates,
    )
    if not selected:
        raise ExperimentConfigurationError("the Stage 6 selection is empty")
    output_root = resolve_output_root(args.output_root)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "stage": STAGE,
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
            "Stage 6 execution requires a clean Git worktree; "
            "commit the execution code first"
        )

    experiment_root = output_root / STAGE / experiment_id
    if experiment_root.exists():
        if not args.resume:
            raise FileExistsError(
                "Stage 6 experiment already exists; use --resume or a new "
                "experiment ID"
            )
        plan_path = experiment_root / "reoptimization_execution_plan.json"
        if not plan_path.is_file():
            raise ExperimentConfigurationError(
                "existing Stage 6 experiment has no execution plan"
            )
        with plan_path.open("r", encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("protocol", {}).get("sha256") != sha256_file(
            protocol_path
        ):
            raise ExperimentConfigurationError(
                "protocol changed after this Stage 6 experiment was created"
            )
    else:
        create_unique_run_directory(experiment_root)

    _write_execution_plan(
        experiment_root=experiment_root,
        experiment_id=experiment_id,
        protocol_path=protocol_path,
        specs=all_specs,
        output_root=output_root,
    )

    for index, spec in enumerate(selected, start=1):
        run_dir = experiment_root / spec.relative_run_dir
        status = inspect_run_status(run_dir, spec)
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
            protocol_path=protocol_path,
            specs=all_specs,
            output_root=output_root,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Stage 6 child process failed for {spec.key} with code "
                f"{completed.returncode}"
            )
        final_status = inspect_run_status(run_dir, spec)
        if final_status != "completed":
            raise RuntimeError(
                f"Stage 6 run {spec.key} ended with status '{final_status}'"
            )

    return experiment_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_root = run_reoptimization(args)
    except (
        ExperimentConfigurationError,
        FileExistsError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if experiment_root is not None:
        print(f"Completed selected Stage 6 runs: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
