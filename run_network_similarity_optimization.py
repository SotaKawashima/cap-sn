"""Run the frozen seven-graph exploration and ten-graph fixed comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from collections import Counter
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
    validate_safe_name,
    write_json,
)
from optimize_single_objective import METHODS
from run_stage6_reoptimization import load_protocol as load_stage6_protocol


STAGE = "network_similarity_optimization"
DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT / "experiment_protocols" / "network_similarity_optimization_v1.json"
)
PLAN_NAME = "network_similarity_execution_plan.json"
ADDED_IDS = {
    "ba1000_seed2", "ba1000_seed3", "ba1000_seed4",
    "facebook_brandeis99", "facebook_bucknell39", "facebook_rice31",
    "wiki_rfa_post2008",
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    relative_run_dir: str
    run_type: str
    network: str
    network_config: str
    num_agents: int
    simulator_seed: int
    iterations: int
    raw_level: str
    method: str | None = None
    optimizer_replicate: int | None = None
    optimizer_seed: int | None = None
    trials: int | None = None
    startup_trials: int | None = None
    condition: str | None = None


def _repo_path(value: str, field: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ExperimentConfigurationError(f"{field} must be inside the repository") from exc
    return path


def _check_hash(path: Path, expected: str, field: str) -> None:
    if not path.is_file():
        raise ExperimentConfigurationError(f"{field} is missing: {path}")
    if sha256_file(path) != expected:
        raise ExperimentConfigurationError(f"{field} SHA-256 mismatch: {path}")


def _config_inputs(config: Path) -> tuple[Path, Path]:
    with config.open("rb") as handle:
        data = tomllib.load(handle)
    base = (config.parent / data["path"]).resolve()
    edge = (base / data["graph"]).resolve()
    community = (base / data["community"]).resolve()
    for path in (edge, community):
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ExperimentConfigurationError(
                f"network input escapes the repository: {path}"
            ) from exc
        if not path.is_file():
            raise ExperimentConfigurationError(f"network input is missing: {path}")
    return edge, community


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("schema_version") != 1 or protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError("invalid network-similarity protocol")

    source = protocol["stage6_protocol"]
    source_path = _repo_path(source["path"], "stage6_protocol.path")
    _check_hash(source_path, source["sha256"], "Stage 6 protocol")
    stage6 = load_stage6_protocol(source_path)
    base = stage6["execution"]
    design = protocol["execution"]
    for field in (
        "methods", "optimizer_replicates", "simulator_seed",
        "iterations_per_evaluation", "evaluations_per_run", "bo_gp_startup_trials",
        "raw_level",
    ):
        if design.get(field) != base[field]:
            raise ExperimentConfigurationError(f"{field} differs from Stage 6")
    if design.get("fixed_conditions") != ["none", "simple_max"]:
        raise ExperimentConfigurationError("fixed conditions must be none and simple_max")
    if protocol.get("original_fixed_networks") != list(base["networks"]):
        raise ExperimentConfigurationError("original networks differ from Stage 6")
    if set(design["methods"]) != set(METHODS):
        raise ExperimentConfigurationError("optimization methods are incomplete")

    observed = protocol["observed_input_manifest"]
    observed_path = _repo_path(observed["path"], "observed_input_manifest.path")
    _check_hash(observed_path, observed["sha256"], "observed input manifest")
    with observed_path.open("r", encoding="utf-8") as handle:
        observed_data = json.load(handle)
    observed_rows = {row["name"]: row for row in observed_data["networks"]}

    added = protocol["added_networks"]
    if {row["id"] for row in added} != ADDED_IDS or len(added) != 7:
        raise ExperimentConfigurationError("added network inventory is not the frozen seven")
    families = Counter(row["family"] for row in added)
    if families != {"ba1000": 3, "facebook": 3, "wiki_vote": 1}:
        raise ExperimentConfigurationError("added network families are inconsistent")
    for row in added:
        network_id = validate_safe_name(row["id"], "network_id")
        if type(row["num_agents"]) is not int or row["num_agents"] <= 0:
            raise ExperimentConfigurationError(f"invalid num_agents for {network_id}")
        if "observed_manifest_name" in row:
            name = row["observed_manifest_name"]
            if name != network_id or name not in observed_rows:
                raise ExperimentConfigurationError(f"missing observed manifest: {name}")
            source_row = observed_rows[name]
            if source_row["nodes"] != row["num_agents"]:
                raise ExperimentConfigurationError(f"node count mismatch: {name}")
            config = observed_path.parent / name / "network.toml"
            for filename, digest in source_row["files_sha256"].items():
                _check_hash(config.parent / filename, digest, f"{name}/{filename}")
            edge, community = _config_inputs(config)
            if edge != config.parent / "edgelist.txt" or community != config.parent / "comm.csv":
                raise ExperimentConfigurationError(f"observed config points to other inputs: {name}")
        else:
            config = _repo_path(row["config"], f"{network_id}.config")
            _check_hash(config, row["config_sha256"], f"{network_id}.config")
            edge, community = _config_inputs(config)
            _check_hash(edge, row["edge_sha256"], f"{network_id}.edge")
            _check_hash(community, row["community_sha256"], f"{network_id}.community")
        row["resolved_config"] = config.relative_to(REPO_ROOT).as_posix()
    for network_id in protocol["original_fixed_networks"]:
        _config_inputs(NETWORKS[network_id].config_path)

    if design.get("expected_optimization_runs") != 126 or design.get("expected_fixed_runs") != 20:
        raise ExperimentConfigurationError("run counts differ from frozen design")
    if design.get("expected_total_runs") != 146:
        raise ExperimentConfigurationError("total run count differs from frozen design")
    if design.get("expected_simulation_iterations") != 632000:
        raise ExperimentConfigurationError("iteration count differs from frozen design")
    protocol["_stage6_execution"] = base
    return protocol


def build_run_specs(protocol: dict[str, Any]) -> list[RunSpec]:
    design = protocol["execution"]
    base = protocol["_stage6_execution"]
    seed = design["simulator_seed"]
    iterations = design["iterations_per_evaluation"]
    specs: list[RunSpec] = []
    for network in protocol["added_networks"]:
        network_id = network["id"]
        config = network["resolved_config"]
        for method in design["methods"]:
            for replicate, optimizer_seed in zip(
                design["optimizer_replicates"], base["optimizer_seeds"][method], strict=True
            ):
                specs.append(RunSpec(
                    key=f"{network_id}:{method}:optseed{replicate}",
                    relative_run_dir=f"{network_id}/{method}/optseed_{replicate}",
                    run_type="optimization", network=network_id,
                    network_config=config, num_agents=network["num_agents"],
                    simulator_seed=seed, iterations=iterations, raw_level="pop",
                    method=method, optimizer_replicate=replicate,
                    optimizer_seed=optimizer_seed, trials=design["evaluations_per_run"],
                    startup_trials=design["bo_gp_startup_trials"],
                ))
    for network in protocol["added_networks"] + [
        {"id": name, "resolved_config": NETWORKS[name].config_path.relative_to(REPO_ROOT).as_posix(),
         "num_agents": NETWORKS[name].num_agents}
        for name in protocol["original_fixed_networks"]
    ]:
        for condition in design["fixed_conditions"]:
            specs.append(RunSpec(
                key=f"{network['id']}:{condition}:simseed{seed}",
                relative_run_dir=f"{network['id']}/{condition}/simseed_{seed}",
                run_type="fixed", network=network["id"],
                network_config=network["resolved_config"], num_agents=network["num_agents"],
                simulator_seed=seed, iterations=iterations, raw_level="pop",
                condition=condition,
            ))
    if len(specs) != 146 or len({spec.key for spec in specs}) != 146:
        raise ExperimentConfigurationError("invalid run inventory")
    return specs


def command_for_spec(spec: RunSpec, *, experiment_id: str, output_root: Path) -> list[str]:
    common = ["--stage", STAGE, "--experiment-id", experiment_id,
              "--simulator-seed", str(spec.simulator_seed),
              "--iterations", str(spec.iterations), "--raw-level", spec.raw_level,
              "--output-root", str(output_root)]
    if spec.run_type == "optimization":
        return [sys.executable, str(REPO_ROOT / "optimize_single_objective.py"),
                "--purpose", STAGE, "--network", spec.network,
                "--network-config", spec.network_config,
                "--network-num-agents", str(spec.num_agents),
                "--method", str(spec.method),
                "--optimizer-replicate", str(spec.optimizer_replicate),
                "--optimizer-seed", str(spec.optimizer_seed),
                "--trials", str(spec.trials),
                "--startup-trials", str(spec.startup_trials), *common]
    command = [sys.executable, str(REPO_ROOT / "run_custom_fixed_condition.py"),
               "--purpose", STAGE, "--network-id", spec.network,
               "--network-config", spec.network_config,
               "--num-agents", str(spec.num_agents), "--condition-id", str(spec.condition)]
    if spec.condition == "none":
        command.append("--no-intervention")
    else:
        command.extend(["--certainty", "1.0", "--effectiveness", "1.0"])
    return [*command, *common]


def inspect_run_status(run_dir: Path, spec: RunSpec) -> str:
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
    if manifest.get("stage") != STAGE:
        return "invalid_completed_design"
    network = manifest.get("network", {})
    if (network.get("id") != spec.network
            or network.get("num_agents") != spec.num_agents
            or network.get("sha256") != sha256_file(REPO_ROOT / spec.network_config)
            or manifest.get("runtime", {}).get("simulator_seed") != spec.simulator_seed):
        return "invalid_completed_design"
    if manifest.get("runtime", {}).get("iteration_count") != spec.iterations:
        return "invalid_completed_design"
    if spec.run_type == "optimization":
        if manifest.get("run_type") != "single_objective_optimization":
            return "invalid_completed_design"
        counts = manifest.get("counts", {})
        optimization = manifest.get("optimization", {})
        if counts.get("complete") != spec.trials or counts.get("failed") != 0 or counts.get("pruned") != 0:
            return "invalid_completed_budget"
        if optimization.get("method") != spec.method or optimization.get("optimizer_seed") != spec.optimizer_seed:
            return "invalid_completed_design"
        if not (run_dir / "trials.csv").is_file():
            return "invalid_completed_outputs"
    else:
        if manifest.get("run_type") != "fixed_condition":
            return "invalid_completed_design"
        if manifest.get("intervention", {}).get("condition_id") != spec.condition:
            return "invalid_completed_design"
    if not (run_dir / "pop.arrow").is_file() and spec.run_type == "fixed":
        return "invalid_completed_outputs"
    return "completed"


def _write_plan(root: Path, *, experiment_id: str, protocol_path: Path,
                specs: Sequence[RunSpec], output_root: Path) -> None:
    rows = [{**asdict(spec), "status": inspect_run_status(root / spec.relative_run_dir, spec),
             "command": command_for_spec(spec, experiment_id=experiment_id, output_root=output_root)}
            for spec in specs]
    counts = Counter(row["status"] for row in rows)
    write_json(root / PLAN_NAME, {
        "schema_version": 1, "experiment_id": experiment_id, "stage": STAGE,
        "updated_at": now_iso(),
        "protocol": {"path": protocol_path.relative_to(REPO_ROOT).as_posix(),
                     "sha256": sha256_file(protocol_path)},
        "git": git_state(),
        "counts": {"total": len(rows), "completed": counts["completed"],
                   "pending": counts["pending"],
                   "other": len(rows) - counts["completed"] - counts["pending"]},
        "runs": rows,
    })


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id")
    parser.add_argument("--networks", nargs="+")
    parser.add_argument("--kinds", nargs="+", choices=["optimization", "fixed"])
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--optimizer-replicates", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Path | None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    specs = build_run_specs(protocol)
    for requested, allowed, field in (
        (args.networks, {spec.network for spec in specs}, "networks"),
        (args.methods, set(METHODS), "methods"),
        (args.optimizer_replicates, set(protocol["execution"]["optimizer_replicates"]),
         "optimizer_replicates"),
    ):
        if requested and set(requested) - allowed:
            raise ExperimentConfigurationError(
                f"unknown {field}: {sorted(set(requested) - allowed)}"
            )
    experiment_id = validate_experiment_id(args.experiment_id) if args.experiment_id else make_experiment_id(STAGE)
    selected = [spec for spec in specs
                if (not args.networks or spec.network in args.networks)
                and (not args.kinds or spec.run_type in args.kinds)
                and (not args.methods or spec.method in args.methods)
                and (not args.optimizer_replicates or spec.optimizer_replicate in args.optimizer_replicates)]
    if not selected:
        raise ExperimentConfigurationError("run selection is empty")
    output_root = resolve_output_root(args.output_root)
    if args.dry_run:
        print(json.dumps({"experiment_id": experiment_id, "stage": STAGE,
                          "full_run_count": len(specs), "selected_run_count": len(selected),
                          "commands": [command_for_spec(spec, experiment_id=experiment_id,
                                                        output_root=output_root) for spec in selected]}, indent=2))
        return None
    if not RUST_BINARY.is_file():
        raise ExperimentConfigurationError(f"release simulator binary is missing: {RUST_BINARY}")
    state = git_state()
    if state["dirty"] and not args.allow_dirty:
        raise ExperimentConfigurationError("execution requires a clean Git worktree")
    root = output_root / STAGE / experiment_id
    if root.exists():
        if not args.resume:
            raise FileExistsError(f"experiment exists; use --resume: {root}")
        plan_path = root / PLAN_NAME
        if not plan_path.is_file():
            raise ExperimentConfigurationError("existing experiment has no execution plan")
        with plan_path.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
        if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
            raise ExperimentConfigurationError("protocol changed after experiment creation")
        if plan.get("git", {}).get("commit") != state["commit"]:
            raise ExperimentConfigurationError("Git commit changed after experiment creation")
    else:
        create_unique_run_directory(root)
    _write_plan(root, experiment_id=experiment_id, protocol_path=protocol_path,
                specs=specs, output_root=output_root)
    for index, spec in enumerate(selected, 1):
        status = inspect_run_status(root / spec.relative_run_dir, spec)
        if status == "completed":
            print(f"[{index}/{len(selected)}] skip completed: {spec.key}", flush=True)
            continue
        if status != "pending":
            raise RuntimeError(f"cannot resume {spec.key} from status {status}")
        print(f"[{index}/{len(selected)}] run: {spec.key}", flush=True)
        result = subprocess.run(command_for_spec(spec, experiment_id=experiment_id,
                                                output_root=output_root), cwd=REPO_ROOT, check=False)
        _write_plan(root, experiment_id=experiment_id, protocol_path=protocol_path,
                    specs=specs, output_root=output_root)
        if result.returncode != 0:
            raise RuntimeError(f"child process failed for {spec.key}: {result.returncode}")
        if inspect_run_status(root / spec.relative_run_dir, spec) != "completed":
            raise RuntimeError(f"run did not complete cleanly: {spec.key}")
    return root


def main(argv: list[str] | None = None) -> int:
    try:
        result = run_experiment(parse_args(argv))
    except (ExperimentConfigurationError, FileExistsError, OSError, RuntimeError,
            KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(f"Completed selected network-similarity runs: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
