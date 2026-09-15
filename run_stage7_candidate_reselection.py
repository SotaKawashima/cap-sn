"""Run the simple-maximum supplement for amended Stage 7 reselection."""

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
    REPO_ROOT
    / "experiment_protocols"
    / "stage7_candidate_reselection_simple_max_v1.json"
)
STAGE = "stage7_candidate_reselection"
EXECUTION_PLAN_NAME = "candidate_reselection_execution_plan.json"


@dataclass(frozen=True)
class CandidateReselectionRunSpec:
    """One supplemental simple-maximum fixed-condition run."""

    key: str
    relative_run_dir: str
    network: str
    condition_id: str
    condition_role: str
    intervention_enabled: bool
    certainty: float
    effectiveness: float
    opinion_csv: None
    opinion_sha256: None
    simulator_seed: int
    iterations: int
    raw_level: str
    candidate_source: None
    source_method: None
    source_optimizer_replicate: None
    source_optimizer_seed: None
    source_simulator_seed: None
    source_final_best: None


def _repo_path(value: str, field_name: str) -> Path:
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
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigurationError(
            f"{field_name} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise ExperimentConfigurationError(f"{field_name} must be finite")
    return result


def _validate_frozen_file(entry: Any, field_name: str) -> Path:
    if not isinstance(entry, dict):
        raise ExperimentConfigurationError(f"{field_name} is missing")
    path = _repo_path(str(entry.get("path", "")), f"{field_name}.path")
    expected_hash = str(entry.get("sha256", "")).lower()
    if len(expected_hash) != 64 or sha256_file(path) != expected_hash:
        raise ExperimentConfigurationError(f"{field_name} SHA-256 mismatch")
    return path


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ExperimentConfigurationError("unsupported reselection schema")
    if protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError(
            f"protocol stage must be {STAGE}"
        )

    source = protocol.get("source_validation")
    if not isinstance(source, dict):
        raise ExperimentConfigurationError("source_validation is missing")
    _repo_path(str(source.get("experiment_root", "")), "source experiment root")
    _validate_frozen_file(source.get("protocol"), "source protocol")
    _validate_frozen_file(source.get("candidate_source"), "candidate source")
    if int(source.get("expected_run_count", -1)) != 189:
        raise ExperimentConfigurationError("source run count must be 189")
    if int(source.get("expected_candidate_count", -1)) != 54:
        raise ExperimentConfigurationError("source candidate count must be 54")

    design = protocol.get("design")
    if not isinstance(design, dict):
        raise ExperimentConfigurationError("design is missing")
    networks = list(design.get("networks", []))
    if len(networks) != len(set(networks)) or set(networks) != set(NETWORKS):
        raise ExperimentConfigurationError(
            "reselection must contain all three supported networks"
        )
    seeds = [
        validate_nonnegative_integer(value, "simulator_seed")
        for value in design.get("simulator_seeds", [])
    ]
    if len(seeds) != 3 or len(seeds) != len(set(seeds)):
        raise ExperimentConfigurationError(
            "reselection requires three unique validation seeds"
        )
    seed_policy = protocol.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise ExperimentConfigurationError("seed_policy is missing")
    if seeds != list(seed_policy.get("candidate_validation", [])):
        raise ExperimentConfigurationError(
            "design seeds do not match candidate_validation seeds"
        )
    seed_groups = [
        set(seed_policy.get(name, []))
        for name in (
            "development_and_fixed_confirmation",
            "stage6_exploration",
            "candidate_validation",
            "inspected_original_final_test",
            "stage9_structure_confirmation",
            "reserved_revised_final_test",
        )
    ]
    for index, group in enumerate(seed_groups):
        for other in seed_groups[index + 1 :]:
            if group & other:
                raise ExperimentConfigurationError("seed groups overlap")

    iterations = validate_positive_integer(
        design.get("iterations_per_seed_block"),
        "iterations_per_seed_block",
    )
    if design.get("raw_level") != "pop":
        raise ExperimentConfigurationError("raw_level must be pop")
    condition = design.get("supplemental_condition")
    if not isinstance(condition, dict):
        raise ExperimentConfigurationError("supplemental_condition is missing")
    if condition.get("id") != "simple_max":
        raise ExperimentConfigurationError(
            "supplemental condition must be simple_max"
        )
    if condition.get("enabled") is not True:
        raise ExperimentConfigurationError("simple_max must be enabled")
    if condition.get("opinion_mode") != "generated_from_design_variables":
        raise ExperimentConfigurationError(
            "simple_max must use the normal generated-opinion path"
        )
    certainty = _finite_float(condition.get("certainty"), "certainty")
    effectiveness = _finite_float(
        condition.get("effectiveness"), "effectiveness"
    )
    if certainty != 1.0 or effectiveness != 1.0:
        raise ExperimentConfigurationError(
            "simple_max certainty and effectiveness must both equal 1.0"
        )
    validate_safe_name(str(condition.get("role", "")), "condition role")

    expected_runs = len(networks) * len(seeds)
    if int(design.get("expected_run_count", -1)) != expected_runs:
        raise ExperimentConfigurationError("expected_run_count is inconsistent")
    if int(design.get("expected_simulation_iterations", -1)) != (
        expected_runs * iterations
    ):
        raise ExperimentConfigurationError(
            "expected_simulation_iterations is inconsistent"
        )
    if int(design.get("expected_combined_run_count", -1)) != (
        int(source["expected_run_count"]) + expected_runs
    ):
        raise ExperimentConfigurationError(
            "expected_combined_run_count is inconsistent"
        )

    rule = protocol.get("selection_rule")
    if not isinstance(rule, dict):
        raise ExperimentConfigurationError("selection_rule is missing")
    if list(rule.get("required_references", [])) != ["none", "simple_max"]:
        raise ExperimentConfigurationError(
            "required references must be none and simple_max"
        )
    if set(rule.get("descriptive_references", [])) != {
        "legacy_balance",
        "prior_high",
    }:
        raise ExperimentConfigurationError(
            "descriptive references must be legacy_balance and prior_high"
        )


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    resolved = Path(path).resolve()
    with resolved.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    _validate_protocol(protocol)
    return protocol


def build_specs(protocol: dict[str, Any]) -> list[CandidateReselectionRunSpec]:
    _validate_protocol(protocol)
    design = protocol["design"]
    condition = design["supplemental_condition"]
    specs: list[CandidateReselectionRunSpec] = []
    for network in design["networks"]:
        for seed in design["simulator_seeds"]:
            seed_value = int(seed)
            specs.append(
                CandidateReselectionRunSpec(
                    key=f"{network}:simple_max:simseed{seed_value}",
                    relative_run_dir=(
                        f"{network}/simple_max/simseed_{seed_value}"
                    ),
                    network=str(network),
                    condition_id="simple_max",
                    condition_role=str(condition["role"]),
                    intervention_enabled=True,
                    certainty=1.0,
                    effectiveness=1.0,
                    opinion_csv=None,
                    opinion_sha256=None,
                    simulator_seed=seed_value,
                    iterations=int(design["iterations_per_seed_block"]),
                    raw_level=str(design["raw_level"]),
                    candidate_source=None,
                    source_method=None,
                    source_optimizer_replicate=None,
                    source_optimizer_seed=None,
                    source_simulator_seed=None,
                    source_final_best=None,
                )
            )
    return specs


def command_for_spec(
    spec: CandidateReselectionRunSpec,
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
        "--simulator-seed",
        str(spec.simulator_seed),
        "--iterations",
        str(spec.iterations),
        "--raw-level",
        spec.raw_level,
        "--output-root",
        str(resolve_output_root(output_root)),
        "--certainty",
        str(spec.certainty),
        "--effectiveness",
        str(spec.effectiveness),
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
    protocol: dict[str, Any],
    specs: Sequence[CandidateReselectionRunSpec],
    output_root: Path,
    initial_git_state: dict[str, Any],
) -> None:
    rows = [
        {
            **asdict(spec),
            "status": _read_status(experiment_root / spec.relative_run_dir),
            "command": command_for_spec(
                spec, experiment_id=experiment_id, output_root=output_root
            ),
        }
        for spec in specs
    ]
    source = protocol["source_validation"]
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
            "source_validation": source,
            "git": initial_git_state,
            "counts": {
                "total": len(rows),
                "completed": sum(row["status"] == "completed" for row in rows),
                "pending": sum(row["status"] == "pending" for row in rows),
                "other": sum(
                    row["status"] not in {"completed", "pending"}
                    for row in rows
                ),
            },
            "runs": rows,
        },
    )


def _filter_specs(
    specs: Sequence[CandidateReselectionRunSpec],
    *,
    networks: Sequence[str] | None,
    simulator_seeds: Sequence[int] | None,
) -> list[CandidateReselectionRunSpec]:
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
        description="Run the frozen simple-maximum reselection supplement."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--networks", nargs="+", choices=sorted(NETWORKS))
    parser.add_argument("--simulator-seeds", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_candidate_reselection(args: argparse.Namespace) -> Path | None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    all_specs = build_specs(protocol)
    selected_specs = _filter_specs(
        all_specs,
        networks=args.networks,
        simulator_seeds=args.simulator_seeds,
    )
    if not selected_specs:
        raise ExperimentConfigurationError("the reselection run set is empty")
    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id("candidate_reselection")
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
            "reselection execution requires a clean Git worktree"
        )

    experiment_root = output_root / STAGE / experiment_id
    if experiment_root.exists():
        if not args.resume:
            raise FileExistsError(
                "reselection experiment already exists; use --resume"
            )
        plan_path = experiment_root / EXECUTION_PLAN_NAME
        if not plan_path.is_file():
            raise ExperimentConfigurationError(
                "existing experiment has no execution plan"
            )
        with plan_path.open(encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("stage") != STAGE:
            raise ExperimentConfigurationError("existing stage does not match")
        if previous_plan.get("protocol", {}).get("sha256") != sha256_file(
            protocol_path
        ):
            raise ExperimentConfigurationError(
                "protocol changed after experiment creation"
            )
        initial_git_state = previous_plan.get("git", {})
        if initial_git_state.get("commit") != current_git_state.get("commit"):
            raise ExperimentConfigurationError(
                "Git commit changed after experiment creation"
            )
    else:
        create_unique_run_directory(experiment_root)
        initial_git_state = current_git_state

    _write_execution_plan(
        experiment_root=experiment_root,
        experiment_id=experiment_id,
        protocol_path=protocol_path,
        protocol=protocol,
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
                f"cannot resume {spec.key} from status {status!r}"
            )
        print(f"[{index}/{len(selected_specs)}] run: {spec.key}", flush=True)
        completed = subprocess.run(
            command_for_spec(
                spec, experiment_id=experiment_id, output_root=output_root
            ),
            cwd=REPO_ROOT,
            check=False,
        )
        _write_execution_plan(
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            protocol_path=protocol_path,
            protocol=protocol,
            specs=all_specs,
            output_root=output_root,
            initial_git_state=initial_git_state,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"child process failed for {spec.key} with code "
                f"{completed.returncode}"
            )
    return experiment_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_root = run_candidate_reselection(args)
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
        print(f"Completed simple-maximum supplement: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
