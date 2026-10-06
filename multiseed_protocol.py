"""Frozen design and provenance helpers for the multiseed extension."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from experiment_runtime import (
    AGENT_CONFIG, NETWORKS, REPO_ROOT, STRATEGY_TEMPLATE,
    ExperimentConfigurationError, config_manifest_entry, sha256_file, validate_safe_name,
)
from run_stage7_candidate_validation import CandidateValidationRunSpec

STAGE = "multiseed_reoptimization"
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "experiment_protocols/multiseed_reoptimization_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ExperimentConfigurationError(f"input must be inside the repository: {value}")
    return path


def collect_input_files() -> list[dict[str, str]]:
    paths: set[Path] = set()

    def collect_config(path: Path) -> None:
        path = path.resolve()
        if path in paths:
            return
        if not path.is_file() or not path.is_relative_to(REPO_ROOT):
            raise ExperimentConfigurationError(f"configuration input is missing: {path}")
        paths.add(path)

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, str) and Path(value).suffix in {".toml", ".csv"}:
                child = (path.parent / value).resolve()
                if child.suffix == ".toml":
                    collect_config(child)
                else:
                    if not child.is_file() or not child.is_relative_to(REPO_ROOT):
                        raise ExperimentConfigurationError(f"configuration input is missing: {child}")
                    paths.add(child)

        with path.open("rb") as handle:
            visit(tomllib.load(handle))

    collect_config(AGENT_CONFIG)
    collect_config(STRATEGY_TEMPLATE)
    for network in NETWORKS.values():
        config_path = network.config_path.resolve()
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        paths.add(config_path)
        for field in ("graph", "community"):
            path = (config_path.parent / config["path"] / config[field]).resolve()
            if not path.is_file() or not path.is_relative_to(REPO_ROOT):
                raise ExperimentConfigurationError(f"network input is missing: {path}")
            paths.add(path)
    return [config_manifest_entry(path) for path in sorted(paths)]


def verify_input_files(entries: list[dict[str, str]]) -> None:
    if not entries:
        raise ExperimentConfigurationError("frozen input inventory is empty")
    for entry in entries:
        path = repo_path(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ExperimentConfigurationError(f"frozen input changed: {entry['path']}")


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol = read_json(path)
    if (protocol.get("schema_version"), protocol.get("stage"), protocol.get("status")) != (
        1, STAGE, "frozen_before_execution"
    ):
        raise ExperimentConfigurationError("invalid frozen multiseed protocol")
    inherited: dict[str, dict[str, Any]] = {}
    for source in protocol["decision_sources"]:
        source_path = repo_path(source["path"])
        if sha256_file(source_path) != source["sha256"]:
            raise ExperimentConfigurationError(f"decision source changed: {source['path']}")
        inherited[source["role"]] = read_json(source_path)
    original = inherited["inherited_search_design"]
    amended = inherited["inherited_candidate_selection_rule"]
    execution = protocol["execution"]
    for key in ("networks", "methods", "optimizer_replicates", "optimizer_seeds",
                "evaluations_per_run", "bo_gp_startup_trials", "raw_level"):
        if execution[key] != original["execution"][key]:
            raise ExperimentConfigurationError(f"inherited search setting changed: {key}")
    for key in ("certainty", "effectiveness", "application_precision_decimal_places"):
        if protocol["design_variables"][key] != original["design_variables"][key]:
            raise ExperimentConfigurationError(f"inherited variable changed: {key}")
    rule = protocol["selection_rule"]
    for key in ("required_references", "eligibility", "candidate_order", "region_grouping"):
        if rule[key] != amended["selection_rule"][key]:
            raise ExperimentConfigurationError(f"inherited selection rule changed: {key}")
    for key in ("target_candidates_per_network", "maximum_regions_per_network", "representative",
                "partial_fill_policy", "no_qualified_policy"):
        if rule["shortlist"][key] != amended["selection_rule"]["shortlist"][key]:
            raise ExperimentConfigurationError(f"inherited shortlist rule changed: {key}")
    objective = protocol["objective"]
    if any(objective[key] != original["objective"][key]
           for key in ("name", "definition_version", "direction")):
        raise ExperimentConfigurationError("objective definition changed")
    if (objective["variance_penalty"] or objective["total_iterations_per_evaluation"] != 500
            or objective["aggregation"] != "arithmetic_mean_of_five_100_iteration_block_means"):
        raise ExperimentConfigurationError("the objective must be a 500-iteration mean")
    groups = [protocol["seed_policy"][phase] for phase in ("exploration", "validation", "final_test")]
    if [len(group) for group in groups] != [5, 3, 5]:
        raise ExperimentConfigurationError("seed group sizes must be 5 / 3 / 5")
    flattened = sum(groups, [])
    if any(type(seed) is not int or seed < 0 for seed in flattened) or len(set(flattened)) != 13:
        raise ExperimentConfigurationError("simulator seeds must be nonnegative and disjoint")
    for phase, group in zip(("execution", "validation", "final_test"), groups, strict=True):
        if protocol[phase]["simulator_seeds"] != group or protocol[phase]["iterations_per_seed_block"] != 100:
            raise ExperimentConfigurationError(f"seed or iteration settings disagree: {phase}")
    if [item["id"] for item in protocol["comparators"]] != ["none", "simple_max"]:
        raise ExperimentConfigurationError("required comparators must be none and simple_max")
    none, maximum = protocol["comparators"]
    if none["enabled"] or not maximum["enabled"] or (maximum["certainty"], maximum["effectiveness"]) != (1.0, 1.0):
        raise ExperimentConfigurationError("comparator settings changed")
    if (none["certainty"] is not None or none["effectiveness"] is not None
            or none["opinion_mode"] != "none"
            or maximum["opinion_mode"] != "generated_from_design_variables"):
        raise ExperimentConfigurationError("comparator opinion policy changed")
    bootstrap = protocol["inference"]["bootstrap"]
    for key in ("method", "levels", "repetitions", "confidence_level"):
        if bootstrap[key] != amended["inference"]["bootstrap"][key]:
            raise ExperimentConfigurationError(f"inherited bootstrap setting changed: {key}")
    final = inherited["inherited_final_evaluation_and_reporting_policy"]
    if (bootstrap["validation_seed"] != amended["inference"]["bootstrap"]["seed"]
            or bootstrap["final_test_seed"] != final["inference"]["bootstrap"]["seed"]
            or protocol["inference"]["minimum_important_effect_threshold"] is not None):
        raise ExperimentConfigurationError("inference policy changed")
    quality = protocol["quality_and_resume_policy"]
    required_quality = {
        "required_complete_evaluations_per_optimization_run": 50,
        "required_complete_blocks_per_evaluation": 5,
        "required_iterations_per_block": 100,
        "allowed_failed_evaluations_at_acceptance": 0,
        "allowed_pruned_evaluations_at_acceptance": 0,
        "recalculate_objective_from_pop_arrow": True,
        "numeric_tolerance": 1e-12,
    }
    if any(quality[key] != value for key, value in required_quality.items()):
        raise ExperimentConfigurationError("quality policy changed")
    expected = {
        "expected_optimization_run_count": 54,
        "expected_evaluation_count": 2700,
        "expected_optimization_block_count": 13500,
        "expected_optimization_iteration_count": 1350000,
        "expected_exploration_reference_block_count": 30,
    }
    if any(execution[key] != value for key, value in expected.items()):
        raise ExperimentConfigurationError("multiseed exploration budget is inconsistent")
    if protocol["validation"]["expected_block_count_before_deduplication"] != 180 or protocol["final_test"]["expected_block_count_before_coordinate_deduplication"] != 120:
        raise ExperimentConfigurationError("validation or final-test budget is inconsistent")
    historical = protocol["final_test"]["historical_candidate_source"]
    if sha256_file(repo_path(historical["path"])) != historical["sha256"]:
        raise ExperimentConfigurationError("historical candidate source changed")
    snapshot = protocol.get("input_inventory")
    if snapshot is not None:
        source_path = repo_path(snapshot["path"])
        if sha256_file(source_path) != snapshot["sha256"]:
            raise ExperimentConfigurationError("frozen input inventory changed")
        entries = read_json(source_path)["files"]
        verify_input_files(entries)
        if collect_input_files() != entries:
            raise ExperimentConfigurationError("configuration dependencies changed")
    return protocol


@dataclass(frozen=True)
class OptimizationSpec:
    key: str
    relative_run_dir: str
    network: str
    method: str
    optimizer_replicate: int
    optimizer_seed: int
    simulator_seeds: tuple[int, ...]
    iterations: int
    trials: int
    startup_trials: int


def optimization_specs(protocol: dict[str, Any]) -> list[OptimizationSpec]:
    settings = protocol["execution"]
    return [
        OptimizationSpec(
            key=f"{network}:{method}:optseed{replicate}",
            relative_run_dir=f"optimization/{network}/{method}/optseed_{replicate}",
            network=network, method=method, optimizer_replicate=replicate,
            optimizer_seed=seed, simulator_seeds=tuple(settings["simulator_seeds"]),
            iterations=100, trials=50, startup_trials=10,
        )
        for network in settings["networks"]
        for method in settings["methods"]
        for replicate, seed in zip(settings["optimizer_replicates"], settings["optimizer_seeds"][method], strict=True)
    ]


def block_spec(
    *, phase: str, network: str, condition_id: str, simulator_seed: int,
    certainty: float | None = None, effectiveness: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> CandidateValidationRunSpec:
    metadata = metadata or {}
    validate_safe_name(phase, "phase")
    validate_safe_name(condition_id, "condition_id")
    if network not in NETWORKS or type(simulator_seed) is not int or simulator_seed < 0:
        raise ExperimentConfigurationError("invalid network or simulator seed")
    enabled = condition_id != "none"
    if enabled and (certainty is None or effectiveness is None or
                    any(not math.isfinite(value) or not 0.5 <= value <= 1.0 for value in (certainty, effectiveness))):
        raise ExperimentConfigurationError("invalid candidate coordinates")
    return CandidateValidationRunSpec(
        key=f"{network}:{condition_id}:simseed{simulator_seed}",
        relative_run_dir=f"{phase}/{network}/{condition_id}/simseed_{simulator_seed}",
        network=network, condition_id=condition_id,
        condition_role=str(metadata.get("condition_role", "reference")),
        intervention_enabled=enabled, certainty=certainty, effectiveness=effectiveness,
        opinion_csv=None, opinion_sha256=None, simulator_seed=simulator_seed,
        iterations=100, raw_level="pop", candidate_source=metadata.get("candidate_source"),
        source_method=metadata.get("source_method"),
        source_optimizer_replicate=metadata.get("source_optimizer_replicate"),
        source_optimizer_seed=metadata.get("source_optimizer_seed"),
        source_simulator_seed=metadata.get("source_simulator_seed"),
        source_final_best=metadata.get("source_final_best"),
    )
