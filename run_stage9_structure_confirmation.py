"""Run the protocol-defined Stage 9 structural confirmation experiment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from analysis.optimization_metrics import (
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
)
from experiment_runtime import (
    REPO_ROOT,
    RUST_BINARY,
    SUMMER_EXPERIMENT_ROOT,
    ExperimentConfigurationError,
    create_unique_run_directory,
    git_state,
    make_experiment_id,
    now_iso,
    read_intervention_opinion_csv,
    resolve_output_root,
    sha256_file,
    validate_experiment_id,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_safe_name,
    write_json,
)


DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT / "experiment_protocols" / "stage9_structure_confirmation_v1.json"
)
STAGE = "stage9_structure_confirmation"
EXECUTION_PLAN_NAME = "structure_confirmation_execution_plan.json"
EXPECTED_FAMILIES = {"lfr_community", "facebook_degree_rewire"}
EXPECTED_CONDITIONS = {"none", "legacy_balance", "prior_high"}
PREVIOUS_SIMULATOR_SEEDS = {
    20001,
    20002,
    20003,
    20004,
    20005,
    30001,
    40001,
    40002,
    40003,
    50001,
    50002,
    50003,
    50004,
    50005,
}


@dataclass(frozen=True)
class StructureConfirmationRunSpec:
    key: str
    relative_run_dir: str
    family: str
    network: str
    structure_level: str
    structure_value: float
    network_seed_index: int | None
    network_generation_seed: int | None
    num_agents: int
    network_config: str
    network_config_sha256: str
    avg_degree: float
    avg_clustering: float
    modularity: float
    internal_edge_ratio: float
    condition_id: str
    condition_role: str
    intervention_enabled: bool
    certainty: float | None
    effectiveness: float | None
    opinion_csv: str | None
    opinion_sha256: str | None
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


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ExperimentConfigurationError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigurationError(
            f"{field_name} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise ExperimentConfigurationError(f"{field_name} must be finite")
    return result


def _design_parameter(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0.5 <= result <= 1.0:
        raise ExperimentConfigurationError(f"{field_name} must be in [0.5, 1.0]")
    return result


def _validate_hash(path: Path, expected: Any, field_name: str) -> str:
    value = str(expected).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExperimentConfigurationError(f"{field_name} must be a SHA-256 hash")
    if not path.is_file():
        raise ExperimentConfigurationError(f"input file is missing: {path}")
    if sha256_file(path) != value:
        raise ExperimentConfigurationError(f"{field_name} does not match {path}")
    return value


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ExperimentConfigurationError("unsupported Stage 9 protocol schema")
    if protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError(f"protocol stage must be {STAGE}")
    objective = protocol.get("objective", {})
    if objective.get("name") != OBJECTIVE_NAME or objective.get(
        "definition_version"
    ) != OBJECTIVE_DEFINITION_VERSION:
        raise ExperimentConfigurationError("Stage 9 objective definition is stale")

    for source in protocol.get("decision_sources", []):
        path = _repo_path(source.get("path", ""), "decision_source.path")
        _validate_hash(path, source.get("sha256"), "decision_source.sha256")

    design = protocol.get("design")
    if not isinstance(design, dict):
        raise ExperimentConfigurationError("protocol is missing the Stage 9 design")
    seeds = [
        validate_nonnegative_integer(value, "simulator_seed")
        for value in design.get("simulator_seeds", [])
    ]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ExperimentConfigurationError(
            "Stage 9 simulator seeds must be a non-empty unique list"
        )
    if set(seeds) & PREVIOUS_SIMULATOR_SEEDS:
        raise ExperimentConfigurationError(
            "Stage 9 simulator seeds overlap Stages 3 through 8"
        )
    if seeds != list(protocol.get("seed_policy", {}).get(STAGE, [])):
        raise ExperimentConfigurationError(
            "Stage 9 design seeds do not match the seed policy"
        )
    iterations = validate_positive_integer(
        design.get("iterations_per_seed_block"), "iterations_per_seed_block"
    )
    if design.get("raw_level") != "pop":
        raise ExperimentConfigurationError("Stage 9 raw_level must be pop")

    families = design.get("network_families", [])
    family_ids = [
        validate_safe_name(str(item.get("id", "")), "family_id")
        for item in families
    ]
    if set(family_ids) != EXPECTED_FAMILIES or len(family_ids) != len(
        set(family_ids)
    ):
        raise ExperimentConfigurationError("Stage 9 network families are invalid")
    reference_levels = {
        str(item["id"]): str(item["reference_level"]) for item in families
    }

    instances = design.get("network_instances", [])
    instance_ids: list[str] = []
    level_counts: Counter[tuple[str, str]] = Counter()
    for instance in instances:
        network_id = validate_safe_name(str(instance.get("id", "")), "network_id")
        instance_ids.append(network_id)
        family = str(instance.get("family", ""))
        if family not in EXPECTED_FAMILIES:
            raise ExperimentConfigurationError(
                f"unsupported Stage 9 network family: {family}"
            )
        level = validate_safe_name(
            str(instance.get("structure_level", "")), "structure_level"
        )
        level_counts[(family, level)] += 1
        _finite_float(instance.get("structure_value"), "structure_value")
        validate_positive_integer(instance.get("num_agents"), "num_agents")
        seed_index = instance.get("network_seed_index")
        generation_seed = instance.get("network_generation_seed")
        if seed_index is not None:
            validate_positive_integer(seed_index, "network_seed_index")
        if generation_seed is not None:
            validate_nonnegative_integer(generation_seed, "network_generation_seed")
        for field in (
            "avg_degree",
            "avg_clustering",
            "modularity",
            "internal_edge_ratio",
        ):
            _finite_float(instance.get(field), f"{network_id}.{field}")
        config_path = _repo_path(
            instance.get("config_path", ""), f"{network_id}.config_path"
        )
        _validate_hash(
            config_path,
            instance.get("config_sha256"),
            f"{network_id}.config_sha256",
        )
    if len(instance_ids) != len(set(instance_ids)):
        raise ExperimentConfigurationError("Stage 9 network IDs must be unique")
    if len(instances) != int(design.get("expected_network_instance_count", -1)):
        raise ExperimentConfigurationError(
            "Stage 9 network instance count does not match the design"
        )
    expected_levels = {
        ("lfr_community", "strong"): 3,
        ("lfr_community", "middle"): 3,
        ("lfr_community", "weak"): 3,
        ("facebook_degree_rewire", "original"): 1,
        ("facebook_degree_rewire", "rewire_0p1"): 1,
        ("facebook_degree_rewire", "rewire_1p0"): 1,
        ("facebook_degree_rewire", "rewire_5p0"): 1,
    }
    if dict(level_counts) != expected_levels:
        raise ExperimentConfigurationError(
            "Stage 9 network-instance levels do not match the frozen design"
        )
    for family, reference in reference_levels.items():
        if level_counts[(family, reference)] == 0:
            raise ExperimentConfigurationError(
                f"Stage 9 reference level is missing for {family}"
            )

    conditions = design.get("conditions", [])
    condition_ids: list[str] = []
    disabled: list[str] = []
    for condition in conditions:
        condition_id = validate_safe_name(
            str(condition.get("id", "")), "condition_id"
        )
        condition_ids.append(condition_id)
        validate_safe_name(str(condition.get("role", "")), "condition_role")
        enabled = condition.get("enabled")
        if not isinstance(enabled, bool):
            raise ExperimentConfigurationError(
                f"condition {condition_id} must have boolean enabled"
            )
        if not enabled:
            disabled.append(condition_id)
            if condition.get("certainty") is not None or condition.get(
                "effectiveness"
            ) is not None:
                raise ExperimentConfigurationError(
                    "no intervention must have null design parameters"
                )
            continue
        certainty = _design_parameter(
            condition.get("certainty"), f"{condition_id}.certainty"
        )
        effectiveness = _design_parameter(
            condition.get("effectiveness"), f"{condition_id}.effectiveness"
        )
        if condition.get("opinion_csv") is not None:
            opinion_path = _repo_path(
                condition["opinion_csv"], f"{condition_id}.opinion_csv"
            )
            _validate_hash(
                opinion_path,
                condition.get("opinion_sha256"),
                f"{condition_id}.opinion_sha256",
            )
            _, applied = read_intervention_opinion_csv(opinion_path)
            if not math.isclose(certainty, applied["certainty"], abs_tol=1e-12) or (
                not math.isclose(
                    effectiveness, applied["effectiveness"], abs_tol=1e-12
                )
            ):
                raise ExperimentConfigurationError(
                    f"condition {condition_id} parameters do not match its CSV"
                )
    if set(condition_ids) != EXPECTED_CONDITIONS or len(condition_ids) != len(
        set(condition_ids)
    ):
        raise ExperimentConfigurationError("Stage 9 conditions are invalid")
    if disabled != ["none"]:
        raise ExperimentConfigurationError(
            "Stage 9 must contain exactly one disabled condition named none"
        )
    if len(conditions) != int(
        design.get("expected_condition_count_per_instance", -1)
    ):
        raise ExperimentConfigurationError(
            "Stage 9 condition count does not match the design"
        )

    expected_runs = len(instances) * len(conditions) * len(seeds)
    if expected_runs != int(design.get("expected_run_count", -1)):
        raise ExperimentConfigurationError(
            "Stage 9 expected run count is inconsistent"
        )
    if expected_runs * iterations != int(
        design.get("expected_simulation_iterations", -1)
    ):
        raise ExperimentConfigurationError(
            "Stage 9 expected simulation iterations are inconsistent"
        )


def load_protocol(
    path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    _validate_protocol(protocol)
    return protocol


def build_specs(protocol: dict[str, Any]) -> list[StructureConfirmationRunSpec]:
    _validate_protocol(protocol)
    design = protocol["design"]
    specs: list[StructureConfirmationRunSpec] = []
    for instance in design["network_instances"]:
        config_path = _repo_path(instance["config_path"], "network_config")
        for condition in design["conditions"]:
            enabled = bool(condition["enabled"])
            opinion_csv = condition.get("opinion_csv")
            resolved_opinion = (
                None
                if opinion_csv is None
                else _repo_path(opinion_csv, "opinion_csv").as_posix()
            )
            for seed_value in design["simulator_seeds"]:
                seed = int(seed_value)
                network_id = str(instance["id"])
                condition_id = str(condition["id"])
                specs.append(
                    StructureConfirmationRunSpec(
                        key=f"{network_id}:{condition_id}:simseed{seed}",
                        relative_run_dir=(
                            f"{network_id}/{condition_id}/simseed_{seed}"
                        ),
                        family=str(instance["family"]),
                        network=network_id,
                        structure_level=str(instance["structure_level"]),
                        structure_value=float(instance["structure_value"]),
                        network_seed_index=(
                            None
                            if instance["network_seed_index"] is None
                            else int(instance["network_seed_index"])
                        ),
                        network_generation_seed=(
                            None
                            if instance["network_generation_seed"] is None
                            else int(instance["network_generation_seed"])
                        ),
                        num_agents=int(instance["num_agents"]),
                        network_config=config_path.as_posix(),
                        network_config_sha256=str(
                            instance["config_sha256"]
                        ).lower(),
                        avg_degree=float(instance["avg_degree"]),
                        avg_clustering=float(instance["avg_clustering"]),
                        modularity=float(instance["modularity"]),
                        internal_edge_ratio=float(instance["internal_edge_ratio"]),
                        condition_id=condition_id,
                        condition_role=str(condition["role"]),
                        intervention_enabled=enabled,
                        certainty=(
                            float(condition["certainty"]) if enabled else None
                        ),
                        effectiveness=(
                            float(condition["effectiveness"]) if enabled else None
                        ),
                        opinion_csv=resolved_opinion,
                        opinion_sha256=(
                            None
                            if opinion_csv is None
                            else str(condition["opinion_sha256"]).lower()
                        ),
                        simulator_seed=seed,
                        iterations=int(design["iterations_per_seed_block"]),
                        raw_level=str(design["raw_level"]),
                    )
                )
    return specs


def command_for_spec(
    spec: StructureConfirmationRunSpec,
    *,
    experiment_id: str,
    output_root: str | Path = SUMMER_EXPERIMENT_ROOT,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "run_custom_fixed_condition.py"),
        "--stage",
        STAGE,
        "--experiment-id",
        experiment_id,
        "--purpose",
        STAGE,
        "--network-id",
        spec.network,
        "--network-config",
        spec.network_config,
        "--num-agents",
        str(spec.num_agents),
        "--condition-id",
        spec.condition_id,
        "--simulator-seed",
        str(spec.simulator_seed),
        "--iterations",
        str(spec.iterations),
        "--raw-level",
        spec.raw_level,
        "--output-root",
        str(resolve_output_root(output_root)),
    ]
    if spec.network_generation_seed is not None:
        command.extend(["--network-seed", str(spec.network_generation_seed)])
    if not spec.intervention_enabled:
        command.append("--no-intervention")
    elif spec.opinion_csv is not None:
        command.extend(["--intervention-opinion-csv", spec.opinion_csv])
    else:
        command.extend(
            [
                "--certainty",
                str(spec.certainty),
                "--effectiveness",
                str(spec.effectiveness),
            ]
        )
    return command


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
    specs: Sequence[StructureConfirmationRunSpec],
    output_root: Path,
    initial_git_state: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
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
    specs: Sequence[StructureConfirmationRunSpec],
    *,
    families: Sequence[str] | None,
    networks: Sequence[str] | None,
    conditions: Sequence[str] | None,
    simulator_seeds: Sequence[int] | None,
) -> list[StructureConfirmationRunSpec]:
    known_families = {spec.family for spec in specs}
    known_networks = {spec.network for spec in specs}
    known_conditions = {spec.condition_id for spec in specs}
    known_seeds = {spec.simulator_seed for spec in specs}
    requested_families = set(families or known_families)
    requested_networks = set(networks or known_networks)
    requested_conditions = set(conditions or known_conditions)
    requested_seeds = set(simulator_seeds or known_seeds)
    if requested_families - known_families:
        raise ExperimentConfigurationError("requested family is not in protocol")
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
        if spec.family in requested_families
        and spec.network in requested_networks
        and spec.condition_id in requested_conditions
        and spec.simulator_seed in requested_seeds
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Stage 9 structure confirmation protocol."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--families", nargs="+", choices=sorted(EXPECTED_FAMILIES))
    parser.add_argument("--networks", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--simulator-seeds", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_structure_confirmation(args: argparse.Namespace) -> Path | None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    all_specs = build_specs(protocol)
    selected_specs = _filter_specs(
        all_specs,
        families=args.families,
        networks=args.networks,
        conditions=args.conditions,
        simulator_seeds=args.simulator_seeds,
    )
    if not selected_specs:
        raise ExperimentConfigurationError("the Stage 9 selection is empty")
    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id("structure_confirmation")
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
            "Stage 9 execution requires a clean Git worktree; commit the "
            "execution code first"
        )
    experiment_root = output_root / STAGE / experiment_id
    if experiment_root.exists():
        if not args.resume:
            raise FileExistsError(
                "Stage 9 experiment already exists; use --resume or a new ID"
            )
        plan_path = experiment_root / EXECUTION_PLAN_NAME
        if not plan_path.is_file():
            raise ExperimentConfigurationError(
                "existing Stage 9 experiment has no execution plan"
            )
        with plan_path.open(encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("stage") != STAGE:
            raise ExperimentConfigurationError(
                "existing experiment stage does not match Stage 9"
            )
        if previous_plan.get("protocol", {}).get("sha256") != sha256_file(
            protocol_path
        ):
            raise ExperimentConfigurationError(
                "protocol changed after the Stage 9 experiment was created"
            )
        initial_git_state = previous_plan.get("git", {})
        if initial_git_state.get("commit") != current_git_state.get("commit"):
            raise ExperimentConfigurationError(
                "Git commit changed after the Stage 9 experiment was created"
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
                f"Stage 9 child process failed for {spec.key} with code "
                f"{completed.returncode}"
            )
    return experiment_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_root = run_structure_confirmation(args)
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
        print(f"Completed selected Stage 9 runs: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
