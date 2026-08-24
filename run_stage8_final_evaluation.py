"""Run the frozen Stage 8 final evaluation experiment."""

from __future__ import annotations

import argparse
import csv
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
    NETWORKS,
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
    REPO_ROOT / "experiment_protocols" / "stage8_final_evaluation_v1.json"
)
STAGE = "stage8_final_evaluation"
EXECUTION_PLAN_NAME = "final_evaluation_execution_plan.json"
REQUIRED_CANDIDATE_COLUMNS = {
    "network",
    "condition_id",
    "certainty",
    "effectiveness",
    "selection_order",
    "reporting_role",
    "selection_mode",
    "qualified",
    "region_id",
    "source_method",
    "source_optimizer_replicate",
    "candidate_source",
    "source_final_best",
    "stage7_validation_mean_jcum",
    "stage7_validation_between_seed_sd",
    "stage7_none_relative_suppression",
    "stage7_none_relative_ci_low",
    "stage7_none_relative_ci_high",
    "stage7_legacy_balance_relative_suppression",
    "stage7_legacy_balance_relative_ci_low",
    "stage7_legacy_balance_relative_ci_high",
    "stage7_prior_high_relative_suppression",
    "stage7_prior_high_relative_ci_low",
    "stage7_prior_high_relative_ci_high",
    "stage7_overall_order",
}


@dataclass(frozen=True)
class FinalEvaluationRunSpec:
    key: str
    relative_run_dir: str
    network: str
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
    candidate_source: str | None
    source_method: str | None
    source_optimizer_replicate: int | None
    source_optimizer_seed: int | None
    source_simulator_seed: int | None
    source_final_best: float | None
    selection_order: int | None
    reporting_role: str | None
    selection_mode: str | None
    qualified: bool | None
    region_id: str | None
    stage7_validation_mean_jcum: float | None
    stage7_validation_between_seed_sd: float | None
    stage7_none_relative_suppression: float | None
    stage7_legacy_balance_relative_suppression: float | None
    stage7_prior_high_relative_suppression: float | None


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


def _design_parameter(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0.5 <= result <= 1.0:
        raise ExperimentConfigurationError(f"{field_name} must be in [0.5, 1.0]")
    return result


def _boolean(value: Any, field_name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ExperimentConfigurationError(f"{field_name} must be True or False")


def read_candidate_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    source = protocol.get("candidate_source")
    if not isinstance(source, dict):
        raise ExperimentConfigurationError("protocol is missing candidate_source")
    path = _repo_path(str(source.get("path", "")), "candidate_source.path")
    expected_hash = str(source.get("sha256", "")).lower()
    if sha256_file(path) != expected_hash:
        raise ExperimentConfigurationError(
            "candidate source SHA-256 does not match the frozen protocol"
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_CANDIDATE_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ExperimentConfigurationError(
                "candidate source is missing required columns"
            )
        raw_rows = list(reader)

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        network = str(raw["network"])
        if network not in NETWORKS:
            raise ExperimentConfigurationError(
                f"unsupported candidate network: {network}"
            )
        condition_id = validate_safe_name(
            str(raw["condition_id"]), "condition_id"
        )
        method = str(raw["source_method"])
        if method not in {"bo_gp", "cma_es", "random_search"}:
            raise ExperimentConfigurationError(
                f"unsupported candidate source method: {method}"
            )
        replicate = validate_positive_integer(
            int(raw["source_optimizer_replicate"]),
            "source_optimizer_replicate",
        )
        expected_condition_id = f"cand_{method}_r{replicate:02d}"
        if condition_id != expected_condition_id:
            raise ExperimentConfigurationError(
                f"candidate ID {condition_id} does not match {expected_condition_id}"
            )
        candidate_source = str(raw["candidate_source"])
        expected_source = f"{network}:{method}:optseed{replicate}"
        if candidate_source != expected_source:
            raise ExperimentConfigurationError(
                f"candidate {network}:{condition_id} has an invalid source"
            )

        selection_order = validate_positive_integer(
            int(raw["selection_order"]), "selection_order"
        )
        qualified = _boolean(raw["qualified"], "qualified")
        selection_mode = validate_safe_name(
            str(raw["selection_mode"]), "selection_mode"
        )
        reporting_role = validate_safe_name(
            str(raw["reporting_role"]), "reporting_role"
        )
        region_id = validate_safe_name(str(raw["region_id"]), "region_id")
        if qualified:
            expected_mode = "qualified_candidates_selected"
            expected_role = (
                "primary_candidate"
                if selection_order == 1
                else "secondary_region_representative"
            )
        else:
            expected_mode = "no_qualified_candidate_exploratory_fallback"
            expected_role = (
                "primary_exploratory_fallback"
                if selection_order == 1
                else "secondary_exploratory_fallback"
            )
        if selection_mode != expected_mode or reporting_role != expected_role:
            raise ExperimentConfigurationError(
                f"candidate {network}:{condition_id} has inconsistent frozen roles"
            )

        rows.append(
            {
                **raw,
                "network": network,
                "condition_id": condition_id,
                "certainty": _design_parameter(raw["certainty"], "certainty"),
                "effectiveness": _design_parameter(
                    raw["effectiveness"], "effectiveness"
                ),
                "selection_order": selection_order,
                "reporting_role": reporting_role,
                "selection_mode": selection_mode,
                "qualified": qualified,
                "region_id": region_id,
                "source_method": method,
                "source_optimizer_replicate": replicate,
                "candidate_source": candidate_source,
                "source_final_best": _finite_float(
                    raw["source_final_best"], "source_final_best"
                ),
                "stage7_validation_mean_jcum": _finite_float(
                    raw["stage7_validation_mean_jcum"],
                    "stage7_validation_mean_jcum",
                ),
                "stage7_validation_between_seed_sd": _finite_float(
                    raw["stage7_validation_between_seed_sd"],
                    "stage7_validation_between_seed_sd",
                ),
                "stage7_none_relative_suppression": _finite_float(
                    raw["stage7_none_relative_suppression"],
                    "stage7_none_relative_suppression",
                ),
                "stage7_legacy_balance_relative_suppression": _finite_float(
                    raw["stage7_legacy_balance_relative_suppression"],
                    "stage7_legacy_balance_relative_suppression",
                ),
                "stage7_prior_high_relative_suppression": _finite_float(
                    raw["stage7_prior_high_relative_suppression"],
                    "stage7_prior_high_relative_suppression",
                ),
                "stage7_overall_order": validate_positive_integer(
                    int(raw["stage7_overall_order"]), "stage7_overall_order"
                ),
            }
        )
    return rows


def _validate_decision_sources(protocol: dict[str, Any]) -> None:
    for source in protocol.get("decision_sources", []):
        if source.get("availability") == "provenance_only_not_required_at_execution":
            continue
        path = _repo_path(str(source.get("path", "")), "decision_source.path")
        expected_hash = str(source.get("sha256", "")).lower()
        if sha256_file(path) != expected_hash:
            raise ExperimentConfigurationError(
                f"decision source hash mismatch: {source.get('role')}"
            )


def _validate_comparators(protocol: dict[str, Any]) -> None:
    comparators = protocol["design"].get("comparators", [])
    comparator_ids: list[str] = []
    disabled: list[str] = []
    for comparator in comparators:
        comparator_id = validate_safe_name(
            str(comparator.get("id", "")), "comparator_id"
        )
        comparator_ids.append(comparator_id)
        validate_safe_name(str(comparator.get("role", "")), "comparator_role")
        validate_safe_name(
            str(comparator.get("reporting_role", "")),
            "comparator_reporting_role",
        )
        enabled = comparator.get("enabled")
        if not isinstance(enabled, bool):
            raise ExperimentConfigurationError(
                f"comparator {comparator_id} must have a boolean enabled value"
            )
        if not enabled:
            disabled.append(comparator_id)
            if comparator.get("certainty") is not None or comparator.get(
                "effectiveness"
            ) is not None:
                raise ExperimentConfigurationError(
                    "the no-intervention comparator must have null parameters"
                )
            continue
        certainty = _design_parameter(
            comparator.get("certainty"), f"{comparator_id}.certainty"
        )
        effectiveness = _design_parameter(
            comparator.get("effectiveness"), f"{comparator_id}.effectiveness"
        )
        configured_csv = comparator.get("opinion_csv")
        if configured_csv is None:
            continue
        opinion_path = _repo_path(
            str(configured_csv), f"{comparator_id}.opinion_csv"
        )
        expected_hash = str(comparator.get("opinion_sha256", "")).lower()
        if sha256_file(opinion_path) != expected_hash:
            raise ExperimentConfigurationError(
                f"comparator {comparator_id} opinion CSV hash mismatch"
            )
        _, applied = read_intervention_opinion_csv(opinion_path)
        if certainty != applied["certainty"] or (
            effectiveness != applied["effectiveness"]
        ):
            raise ExperimentConfigurationError(
                f"comparator {comparator_id} parameters do not match its CSV"
            )
    if len(comparator_ids) != len(set(comparator_ids)):
        raise ExperimentConfigurationError("Stage 8 comparator IDs must be unique")
    if disabled != ["none"]:
        raise ExperimentConfigurationError(
            "Stage 8 must contain exactly one disabled comparator named none"
        )
    if set(comparator_ids) != {"none", "legacy_balance", "prior_high"}:
        raise ExperimentConfigurationError(
            "Stage 8 comparators must be none, legacy_balance, and prior_high"
        )


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ExperimentConfigurationError("unsupported Stage 8 protocol schema")
    if protocol.get("stage") != STAGE:
        raise ExperimentConfigurationError(
            "protocol stage must be stage8_final_evaluation"
        )
    if protocol.get("status") != "frozen_before_execution":
        raise ExperimentConfigurationError(
            "Stage 8 protocol must be frozen before execution"
        )
    objective = protocol.get("objective", {})
    if objective.get("name") != OBJECTIVE_NAME or objective.get(
        "definition_version"
    ) != OBJECTIVE_DEFINITION_VERSION:
        raise ExperimentConfigurationError("Stage 8 objective is inconsistent")
    _validate_decision_sources(protocol)

    design = protocol.get("design")
    if not isinstance(design, dict):
        raise ExperimentConfigurationError("protocol is missing the Stage 8 design")
    networks = list(design.get("networks", []))
    if set(networks) != set(NETWORKS) or len(networks) != len(set(networks)):
        raise ExperimentConfigurationError(
            "Stage 8 must contain each supported network exactly once"
        )
    final_seeds = [
        validate_nonnegative_integer(value, "simulator_seed")
        for value in design.get("simulator_seeds", [])
    ]
    if not final_seeds or len(final_seeds) != len(set(final_seeds)):
        raise ExperimentConfigurationError(
            "Stage 8 simulator seeds must be a non-empty unique list"
        )
    seed_policy = protocol.get("seed_policy", {})
    if final_seeds != list(seed_policy.get("stage8_final_test", [])):
        raise ExperimentConfigurationError(
            "Stage 8 design seeds do not match stage8_final_test seeds"
        )
    seed_groups = [
        set(seed_policy.get(name, []))
        for name in (
            "development_and_fixed_confirmation",
            "stage6_exploration",
            "stage7_candidate_validation",
            "stage8_final_test",
        )
    ]
    for index, group in enumerate(seed_groups):
        for other in seed_groups[index + 1 :]:
            if group & other:
                raise ExperimentConfigurationError("Stage 8 seed groups overlap")

    iterations = validate_positive_integer(
        design.get("iterations_per_seed_block"), "iterations_per_seed_block"
    )
    if design.get("raw_level") != "pop":
        raise ExperimentConfigurationError("Stage 8 raw_level must be pop")

    candidates = read_candidate_rows(protocol)
    expected_candidate_count = int(design.get("candidate_count", -1))
    expected_per_network = int(design.get("candidate_count_per_network", -1))
    if len(candidates) != expected_candidate_count:
        raise ExperimentConfigurationError(
            "Stage 8 candidate count does not match the protocol"
        )
    candidate_keys = [
        (row["network"], row["condition_id"]) for row in candidates
    ]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ExperimentConfigurationError("Stage 8 candidate keys must be unique")
    per_network = Counter(row["network"] for row in candidates)
    if set(per_network) != set(networks) or set(per_network.values()) != {
        expected_per_network
    }:
        raise ExperimentConfigurationError(
            "Stage 8 candidate counts per network do not match the protocol"
        )
    for network in networks:
        rows = [row for row in candidates if row["network"] == network]
        if {row["selection_order"] for row in rows} != set(
            range(1, expected_per_network + 1)
        ):
            raise ExperimentConfigurationError(
                f"Stage 8 selection orders are invalid for {network}"
            )
        if len({row["region_id"] for row in rows}) != expected_per_network:
            raise ExperimentConfigurationError(
                f"Stage 8 candidates do not represent distinct regions for {network}"
            )
        if len({row["qualified"] for row in rows}) != 1:
            raise ExperimentConfigurationError(
                f"Stage 8 qualification status is inconsistent for {network}"
            )
    reporting = protocol.get("candidate_reporting_policy", {})
    qualified_networks = set(reporting.get("qualified_networks", []))
    fallback_networks = set(reporting.get("exploratory_fallback_networks", []))
    if qualified_networks | fallback_networks != set(networks) or (
        qualified_networks & fallback_networks
    ):
        raise ExperimentConfigurationError(
            "Stage 8 reporting network groups are invalid"
        )
    observed_qualified = {
        row["network"] for row in candidates if row["qualified"]
    }
    if observed_qualified != qualified_networks:
        raise ExperimentConfigurationError(
            "candidate qualification does not match the reporting policy"
        )

    _validate_comparators(protocol)
    comparator_ids = {
        str(row["id"]) for row in design.get("comparators", [])
    }
    if comparator_ids & {row["condition_id"] for row in candidates}:
        raise ExperimentConfigurationError("candidate and comparator IDs overlap")
    conditions_per_network = expected_per_network + len(comparator_ids)
    if int(design.get("condition_count_per_network", -1)) != conditions_per_network:
        raise ExperimentConfigurationError(
            "Stage 8 condition_count_per_network is inconsistent"
        )
    expected_runs = len(networks) * conditions_per_network * len(final_seeds)
    if int(design.get("expected_run_count", -1)) != expected_runs:
        raise ExperimentConfigurationError(
            "Stage 8 expected_run_count is inconsistent"
        )
    if int(design.get("expected_simulation_iterations", -1)) != (
        expected_runs * iterations
    ):
        raise ExperimentConfigurationError(
            "Stage 8 expected_simulation_iterations is inconsistent"
        )
    if protocol.get("inference", {}).get("minimum_important_effect_threshold") is not None:
        raise ExperimentConfigurationError(
            "Stage 8 must not define a post-hoc minimum effect threshold"
        )


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    resolved = Path(path).resolve()
    with resolved.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    _validate_protocol(protocol)
    return protocol


def build_specs(protocol: dict[str, Any]) -> list[FinalEvaluationRunSpec]:
    _validate_protocol(protocol)
    design = protocol["design"]
    candidates = read_candidate_rows(protocol)
    candidates_by_network: dict[str, list[dict[str, Any]]] = {
        network: [] for network in design["networks"]
    }
    for candidate in candidates:
        candidates_by_network[candidate["network"]].append(candidate)
    for values in candidates_by_network.values():
        values.sort(key=lambda row: int(row["selection_order"]))

    specs: list[FinalEvaluationRunSpec] = []
    for network in design["networks"]:
        conditions: list[dict[str, Any]] = []
        for candidate in candidates_by_network[network]:
            conditions.append(
                {
                    "id": candidate["condition_id"],
                    "role": "final_candidate",
                    "enabled": True,
                    "certainty": candidate["certainty"],
                    "effectiveness": candidate["effectiveness"],
                    "opinion_csv": None,
                    "opinion_sha256": None,
                    **{
                        field: candidate[field]
                        for field in (
                            "candidate_source",
                            "source_method",
                            "source_optimizer_replicate",
                            "source_final_best",
                            "selection_order",
                            "reporting_role",
                            "selection_mode",
                            "qualified",
                            "region_id",
                            "stage7_validation_mean_jcum",
                            "stage7_validation_between_seed_sd",
                            "stage7_none_relative_suppression",
                            "stage7_legacy_balance_relative_suppression",
                            "stage7_prior_high_relative_suppression",
                        )
                    },
                }
            )
        for comparator in design["comparators"]:
            opinion_csv = comparator.get("opinion_csv")
            opinion_path = (
                _repo_path(str(opinion_csv), f"{comparator['id']}.opinion_csv")
                if opinion_csv is not None
                else None
            )
            conditions.append(
                {
                    "id": comparator["id"],
                    "role": comparator["role"],
                    "enabled": bool(comparator["enabled"]),
                    "certainty": comparator["certainty"],
                    "effectiveness": comparator["effectiveness"],
                    "opinion_csv": (
                        None if opinion_path is None else opinion_path.as_posix()
                    ),
                    "opinion_sha256": comparator.get("opinion_sha256"),
                    "candidate_source": None,
                    "source_method": None,
                    "source_optimizer_replicate": None,
                    "source_final_best": None,
                    "selection_order": None,
                    "reporting_role": comparator.get("reporting_role"),
                    "selection_mode": None,
                    "qualified": None,
                    "region_id": None,
                    "stage7_validation_mean_jcum": None,
                    "stage7_validation_between_seed_sd": None,
                    "stage7_none_relative_suppression": None,
                    "stage7_legacy_balance_relative_suppression": None,
                    "stage7_prior_high_relative_suppression": None,
                }
            )
        for condition in conditions:
            for seed in design["simulator_seeds"]:
                condition_id = str(condition["id"])
                seed_value = int(seed)
                specs.append(
                    FinalEvaluationRunSpec(
                        key=f"{network}:{condition_id}:simseed{seed_value}",
                        relative_run_dir=(
                            f"{network}/{condition_id}/simseed_{seed_value}"
                        ),
                        network=str(network),
                        condition_id=condition_id,
                        condition_role=str(condition["role"]),
                        intervention_enabled=bool(condition["enabled"]),
                        certainty=(
                            float(condition["certainty"])
                            if condition["enabled"]
                            else None
                        ),
                        effectiveness=(
                            float(condition["effectiveness"])
                            if condition["enabled"]
                            else None
                        ),
                        opinion_csv=condition["opinion_csv"],
                        opinion_sha256=condition["opinion_sha256"],
                        simulator_seed=seed_value,
                        iterations=int(design["iterations_per_seed_block"]),
                        raw_level=str(design["raw_level"]),
                        candidate_source=condition["candidate_source"],
                        source_method=condition["source_method"],
                        source_optimizer_replicate=condition[
                            "source_optimizer_replicate"
                        ],
                        source_optimizer_seed=None,
                        source_simulator_seed=None,
                        source_final_best=condition["source_final_best"],
                        selection_order=condition["selection_order"],
                        reporting_role=condition["reporting_role"],
                        selection_mode=condition["selection_mode"],
                        qualified=condition["qualified"],
                        region_id=condition["region_id"],
                        stage7_validation_mean_jcum=condition[
                            "stage7_validation_mean_jcum"
                        ],
                        stage7_validation_between_seed_sd=condition[
                            "stage7_validation_between_seed_sd"
                        ],
                        stage7_none_relative_suppression=condition[
                            "stage7_none_relative_suppression"
                        ],
                        stage7_legacy_balance_relative_suppression=condition[
                            "stage7_legacy_balance_relative_suppression"
                        ],
                        stage7_prior_high_relative_suppression=condition[
                            "stage7_prior_high_relative_suppression"
                        ],
                    )
                )
    return specs


def command_for_spec(
    spec: FinalEvaluationRunSpec,
    *,
    experiment_id: str,
    output_root: str | Path = SUMMER_EXPERIMENT_ROOT,
) -> list[str]:
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
        str(resolve_output_root(output_root)),
    ]
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
    protocol: dict[str, Any],
    specs: Sequence[FinalEvaluationRunSpec],
    output_root: Path,
    initial_git_state: dict[str, Any],
) -> None:
    rows = [
        {
            **asdict(spec),
            "status": _read_status(experiment_root / spec.relative_run_dir),
            "command": command_for_spec(
                spec,
                experiment_id=experiment_id,
                output_root=output_root,
            ),
        }
        for spec in specs
    ]
    candidate_path = _repo_path(
        protocol["candidate_source"]["path"], "candidate_source.path"
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
            "candidate_source": {
                "path": candidate_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(candidate_path),
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
    specs: Sequence[FinalEvaluationRunSpec],
    *,
    networks: Sequence[str] | None,
    conditions: Sequence[str] | None,
    simulator_seeds: Sequence[int] | None,
) -> list[FinalEvaluationRunSpec]:
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
        description="Run the frozen Stage 8 final evaluation protocol."
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


def run_final_evaluation(args: argparse.Namespace) -> Path | None:
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
        raise ExperimentConfigurationError("the Stage 8 selection is empty")
    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id("final_evaluation")
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
            "Stage 8 execution requires a clean Git worktree; commit the "
            "execution code first"
        )

    experiment_root = output_root / STAGE / experiment_id
    if experiment_root.exists():
        if not args.resume:
            raise FileExistsError(
                "Stage 8 experiment already exists; use --resume or a new "
                "experiment ID"
            )
        plan_path = experiment_root / EXECUTION_PLAN_NAME
        if not plan_path.is_file():
            raise ExperimentConfigurationError(
                "existing Stage 8 experiment has no execution plan"
            )
        with plan_path.open(encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("stage") != STAGE:
            raise ExperimentConfigurationError(
                "existing experiment stage does not match Stage 8"
            )
        if previous_plan.get("protocol", {}).get("sha256") != sha256_file(
            protocol_path
        ):
            raise ExperimentConfigurationError(
                "protocol changed after this Stage 8 experiment was created"
            )
        candidate_path = _repo_path(
            protocol["candidate_source"]["path"], "candidate_source.path"
        )
        if previous_plan.get("candidate_source", {}).get(
            "sha256"
        ) != sha256_file(candidate_path):
            raise ExperimentConfigurationError(
                "candidate source changed after this Stage 8 experiment was created"
            )
        initial_git_state = previous_plan.get("git", {})
        if initial_git_state.get("commit") != current_git_state.get("commit"):
            raise ExperimentConfigurationError(
                "Git commit changed after this Stage 8 experiment was created"
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
                f"cannot resume {spec.key} from existing status '{status}'"
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
            protocol=protocol,
            specs=all_specs,
            output_root=output_root,
            initial_git_state=initial_git_state,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Stage 8 child process failed for {spec.key} with code "
                f"{completed.returncode}"
            )
    return experiment_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_root = run_final_evaluation(args)
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
        print(f"Completed selected Stage 8 runs: {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
