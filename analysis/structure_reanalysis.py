"""Audit and reaggregate the legacy network-structure experiments.

The legacy batches predate the summer objective, but retain raw ``pop.arrow``
and ``agent.arrow`` outputs.  This module recomputes the current cumulative
selfish fraction without modifying the old files and keeps network-generation
and simulator randomness as separate metadata.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis.optimization_metrics import (
    LEGACY_METRIC_NAME,
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
    compute_selfish_metrics,
    validate_pop_agent_consistency,
)


STAGE = "stage9_structure_reanalysis"
REQUIRED_STRATEGIES = ("balance", "effective_high", "certainty_high")
STRUCTURAL_METRICS = (
    "num_nodes",
    "num_edges",
    "avg_degree",
    "avg_clustering",
    "transitivity",
    "mu",
    "lfr_modularity",
    "lfr_internal_edge_ratio",
    "louvain_modularity",
    "louvain_internal_edge_ratio",
    "swap_ratio",
    "top20_largest_comm_ratio",
    "middle20_largest_comm_ratio",
    "bottom20_largest_comm_ratio",
    "top20_external_ratio_mean",
    "middle20_external_ratio_mean",
    "bottom20_external_ratio_mean",
    "top20_participation_mean",
    "middle20_participation_mean",
    "bottom20_participation_mean",
)


@dataclass(frozen=True)
class LegacyRunSpec:
    dataset: str
    evidence_scope: str
    analysis_note: str
    network: str
    strategy: str
    structure_level: str
    block_kind: str
    block_id: str
    network_seed_index: int | None
    network_generation_seed: int | None
    simulator_seed: int
    num_agents: int
    pop_path: Path
    agent_path: Path
    metadata: dict[str, Any]

    @property
    def run_key(self) -> str:
        return (
            f"{self.dataset}:{self.structure_level}:{self.block_id}:"
            f"{self.network}:{self.strategy}:simseed{self.simulator_seed}"
        )


@dataclass(frozen=True)
class StructureReanalysisData:
    iterations: pd.DataFrame
    runs: pd.DataFrame
    audit: pd.DataFrame
    dataset_scope: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = _read_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported Stage 9 protocol schema")
    if protocol.get("stage") != STAGE:
        raise ValueError(f"protocol stage must be {STAGE}")
    if protocol.get("objective", {}).get("name") != OBJECTIVE_NAME:
        raise ValueError("Stage 9 protocol objective does not match current code")
    if (
        protocol.get("objective", {}).get("definition_version")
        != OBJECTIVE_DEFINITION_VERSION
    ):
        raise ValueError("Stage 9 objective definition version is stale")
    strategies = protocol.get("strategies", {})
    if tuple(strategies) != REQUIRED_STRATEGIES:
        raise ValueError(
            "Stage 9 strategies must be ordered as balance, effective_high, "
            "certainty_high"
        )
    dataset_ids = [str(item.get("id")) for item in protocol.get("datasets", [])]
    if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Stage 9 dataset IDs must be non-empty and unique")
    return protocol


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _network_seed_index(network: str, metadata: dict[str, Any]) -> int | None:
    value = metadata.get("seed_index")
    if value is not None:
        return int(value)
    match = re.search(r"(?:^|_)seed(\d+)$", network)
    if match:
        return int(match.group(1))
    if network == "ba1000":
        return 1
    return None


def _generation_seed(metadata: dict[str, Any]) -> int | None:
    for field in ("actual_seed", "requested_seed", "graph_seed", "rewire_seed"):
        value = metadata.get(field)
        if value is not None:
            return int(value)
    return None


def _parse_arrow_location(
    root: Path, pop_path: Path, layout: str, default_seed: int
) -> tuple[str, str, int]:
    relative = pop_path.relative_to(root)
    parts = relative.parts
    if layout == "standard":
        if len(parts) != 4 or parts[2] != "result":
            raise ValueError(f"unexpected standard path: {relative}")
        return parts[0], parts[1], int(default_seed)
    if layout == "simseed_network_strategy":
        if len(parts) != 5 or parts[3] != "result":
            raise ValueError(f"unexpected simulator-seed path: {relative}")
        match = re.fullmatch(r"seed_(\d+)", parts[0])
        if not match:
            raise ValueError(f"invalid simulator-seed directory: {relative}")
        return parts[1], parts[2], int(match.group(1))
    raise ValueError(f"unsupported Stage 9 path layout: {layout}")


def discover_legacy_runs(
    protocol: dict[str, Any], *, repo_root: str | Path
) -> list[LegacyRunSpec]:
    root = Path(repo_root).resolve()
    strategy_ids = set(protocol["strategies"])
    specs: list[LegacyRunSpec] = []
    for dataset in protocol["datasets"]:
        dataset_id = str(dataset["id"])
        run_root = _repo_path(root, dataset["run_root"])
        if not run_root.is_dir():
            raise FileNotFoundError(f"Stage 9 run root is missing: {run_root}")

        metadata_by_network: dict[str, dict[str, Any]] = {}
        if dataset.get("metadata_csv"):
            metadata_path = _repo_path(root, dataset["metadata_csv"])
            metadata_frame = pd.read_csv(metadata_path)
            if "network" not in metadata_frame:
                raise ValueError(f"metadata has no network column: {metadata_path}")
            if metadata_frame["network"].duplicated().any():
                raise ValueError(f"metadata network IDs are not unique: {metadata_path}")
            metadata_by_network = {
                str(row["network"]): {
                    column: _plain_value(row[column])
                    for column in metadata_frame.columns
                }
                for _, row in metadata_frame.iterrows()
            }

        for pop_path in sorted(run_root.rglob("*_pop.arrow")):
            network, strategy, simulator_seed = _parse_arrow_location(
                run_root,
                pop_path,
                str(dataset["path_layout"]),
                int(dataset.get("default_simulator_seed", 0)),
            )
            if strategy not in strategy_ids:
                raise ValueError(
                    f"unknown strategy {strategy!r} in {pop_path.relative_to(root)}"
                )
            metadata = dict(metadata_by_network.get(network, {}))
            if metadata_by_network and not metadata:
                raise ValueError(
                    f"network {network!r} is missing from metadata for {dataset_id}"
                )
            format_values = {"network": network, **metadata}
            try:
                structure_level = str(dataset["structure_level_template"]).format(
                    **format_values
                )
            except KeyError as exc:
                raise ValueError(
                    f"structure template for {dataset_id} requires {exc.args[0]}"
                ) from exc

            fixed_by_network = dataset.get("fixed_num_agents_by_network", {})
            if network in fixed_by_network:
                num_agents = int(fixed_by_network[network])
            elif dataset.get("fixed_num_agents") is not None:
                num_agents = int(dataset["fixed_num_agents"])
            elif metadata.get("num_nodes") is not None:
                num_agents = int(metadata["num_nodes"])
            else:
                raise ValueError(f"num_agents is unknown for {dataset_id}:{network}")

            network_seed_index = _network_seed_index(network, metadata)
            block_kind = str(dataset["block_kind"])
            if block_kind == "network_seed":
                if network_seed_index is None:
                    raise ValueError(f"network seed is unknown for {dataset_id}:{network}")
                block_id = f"network_seed_{network_seed_index}"
            elif block_kind == "simulator_seed":
                block_id = f"simulator_seed_{simulator_seed}"
            elif block_kind == "observed_network":
                block_id = f"observed_{network}"
            else:
                raise ValueError(f"unsupported block kind: {block_kind}")

            agent_path = pop_path.with_name(
                pop_path.name.replace("_pop.arrow", "_agent.arrow")
            )
            specs.append(
                LegacyRunSpec(
                    dataset=dataset_id,
                    evidence_scope=str(dataset["evidence_scope"]),
                    analysis_note=str(dataset["analysis_note"]),
                    network=network,
                    strategy=strategy,
                    structure_level=structure_level,
                    block_kind=block_kind,
                    block_id=block_id,
                    network_seed_index=network_seed_index,
                    network_generation_seed=_generation_seed(metadata),
                    simulator_seed=simulator_seed,
                    num_agents=num_agents,
                    pop_path=pop_path,
                    agent_path=agent_path,
                    metadata=metadata,
                )
            )

    keys = [spec.run_key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("Stage 9 run discovery produced duplicate run keys")
    expected = protocol.get("expected_raw_run_count")
    if expected is not None and len(specs) != int(expected):
        raise ValueError(
            f"Stage 9 discovered {len(specs)} runs, expected {int(expected)}"
        )
    return specs


def _metadata_projection(spec: LegacyRunSpec) -> dict[str, Any]:
    return {column: spec.metadata.get(column) for column in STRUCTURAL_METRICS}


def load_legacy_structure_data(
    specs: Iterable[LegacyRunSpec], *, expected_iterations: int
) -> StructureReanalysisData:
    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in specs:
        audit = {
            "run_key": spec.run_key,
            "dataset": spec.dataset,
            "network": spec.network,
            "strategy": spec.strategy,
            "files_present": False,
            "pop_valid": False,
            "agent_consistent": False,
            "configuration_provenance": "batch_script_and_log_only",
            "valid": False,
            "error": None,
        }
        try:
            missing = [
                path.name
                for path in (spec.pop_path, spec.agent_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(f"missing raw files: {missing}")
            audit["files_present"] = True

            pop = pd.read_feather(spec.pop_path)
            agent = pd.read_feather(spec.agent_path)
            result = compute_selfish_metrics(
                pop,
                num_agents=spec.num_agents,
                expected_iterations=expected_iterations,
            )
            audit["pop_valid"] = True
            validate_pop_agent_consistency(
                pop,
                agent,
                num_agents=spec.num_agents,
                expected_iterations=expected_iterations,
            )
            audit["agent_consistent"] = True

            identifiers = {
                "run_key": spec.run_key,
                "dataset": spec.dataset,
                "evidence_scope": spec.evidence_scope,
                "network": spec.network,
                "structure_level": spec.structure_level,
                "strategy": spec.strategy,
                "block_kind": spec.block_kind,
                "block_id": spec.block_id,
                "network_seed_index": spec.network_seed_index,
                "network_generation_seed": spec.network_generation_seed,
                "simulator_seed": spec.simulator_seed,
                "num_agents": spec.num_agents,
            }
            metrics = result.per_iteration.copy()
            for position, (field, value) in enumerate(identifiers.items()):
                metrics.insert(position, field, value)
            iteration_frames.append(metrics)
            run_rows.append(
                {
                    **identifiers,
                    **_metadata_projection(spec),
                    "n_iterations": int(result.summary["n_iterations"]),
                    "jcum": float(result.summary[OBJECTIVE_NAME]),
                    "jpeak": float(result.summary["peak_new_selfish_ratio"]),
                    "legacy_metric": float(result.summary[LEGACY_METRIC_NAME]),
                    "mean_recorded_steps": float(
                        result.summary["mean_recorded_steps"]
                    ),
                    "pop_path": spec.pop_path.as_posix(),
                    "agent_path": spec.agent_path.as_posix(),
                }
            )
            audit["valid"] = True
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"{spec.run_key}: {audit['error']}")
        audit_rows.append(audit)

    if failures:
        raise ValueError(
            f"Stage 9 raw-data audit failed for {len(failures)} runs:\n"
            + "\n".join(failures[:25])
        )
    iterations = pd.concat(iteration_frames, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    audit = pd.DataFrame(audit_rows)
    dataset_scope = (
        runs.groupby(
            ["dataset", "evidence_scope", "block_kind"], sort=True, dropna=False
        )
        .agg(
            raw_run_count=("run_key", "size"),
            network_instance_count=("network", "nunique"),
            structure_level_count=("structure_level", "nunique"),
            block_count=("block_id", "nunique"),
            simulator_seed_count=("simulator_seed", "nunique"),
            total_iterations=("n_iterations", "sum"),
        )
        .reset_index()
    )
    dataset_scope["has_no_intervention"] = False
    dataset_scope["available_estimands"] = (
        "Jcum,Jpeak,strategy_contrast; eta_unavailable"
    )
    return StructureReanalysisData(
        iterations=iterations.sort_values(
            ["dataset", "structure_level", "block_id", "strategy", "num_iter"]
        ).reset_index(drop=True),
        runs=runs.sort_values("run_key").reset_index(drop=True),
        audit=audit.sort_values("run_key").reset_index(drop=True),
        dataset_scope=dataset_scope,
    )


def _bootstrap_mean(
    blocks: list[np.ndarray], *, repetitions: int, seed: int
) -> dict[str, Any]:
    if not blocks or any(len(block) < 2 for block in blocks):
        raise ValueError("bootstrap requires non-empty blocks with at least two rows")
    estimate = float(np.mean([block.mean() for block in blocks]))
    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    samples = np.zeros(repetitions, dtype=float)
    if n_blocks == 1:
        block = blocks[0]
        indices = rng.integers(0, len(block), size=(repetitions, len(block)))
        samples = block[indices].mean(axis=1)
        scope = "iteration_only"
    else:
        for _ in range(n_blocks):
            selected = rng.integers(0, n_blocks, size=repetitions)
            contribution = np.empty(repetitions, dtype=float)
            for block_index, block in enumerate(blocks):
                rows = np.flatnonzero(selected == block_index)
                if len(rows) == 0:
                    continue
                indices = rng.integers(
                    0, len(block), size=(len(rows), len(block))
                )
                contribution[rows] = block[indices].mean(axis=1)
            samples += contribution / n_blocks
        scope = "hierarchical_block_and_iteration"
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_half_width": float((high - low) / 2),
        "n_blocks": n_blocks,
        "n_per_block_min": min(len(block) for block in blocks),
        "uncertainty_scope": scope,
    }


def build_level_summary(
    iterations: pd.DataFrame, *, repetitions: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "dataset",
        "evidence_scope",
        "structure_level",
        "strategy",
        "block_kind",
    ]
    for offset, (key, group) in enumerate(
        iterations.groupby(keys, sort=True, dropna=False)
    ):
        block_jcum = [
            frame[OBJECTIVE_NAME].to_numpy(float)
            for _, frame in group.groupby("block_id", sort=True)
        ]
        block_jpeak = [
            frame["peak_new_selfish_ratio"].to_numpy(float)
            for _, frame in group.groupby("block_id", sort=True)
        ]
        jcum = _bootstrap_mean(
            block_jcum, repetitions=repetitions, seed=seed + offset * 2
        )
        jpeak = _bootstrap_mean(
            block_jpeak, repetitions=repetitions, seed=seed + offset * 2 + 1
        )
        block_means = group.groupby("block_id")[OBJECTIVE_NAME].mean()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_iterations": int(len(group)),
                "between_block_sd_jcum": (
                    float(block_means.std()) if len(block_means) > 1 else None
                ),
                **{f"jcum_{name}": value for name, value in jcum.items()},
                **{f"jpeak_{name}": value for name, value in jpeak.items()},
            }
        )
    return pd.DataFrame(rows)


def _paired_effect(
    frame: pd.DataFrame,
    *,
    reference_column: str,
    candidate_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    data = frame[["block_id", reference_column, candidate_column]].dropna()
    blocks = [
        group[[reference_column, candidate_column]].to_numpy(float)
        for _, group in data.groupby("block_id", sort=True)
    ]
    if not blocks or any(len(block) < 2 for block in blocks):
        raise ValueError("paired bootstrap requires at least two rows per block")
    reference_estimate = float(np.mean([block[:, 0].mean() for block in blocks]))
    candidate_estimate = float(np.mean([block[:, 1].mean() for block in blocks]))
    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    reference_samples = np.zeros(repetitions, dtype=float)
    candidate_samples = np.zeros(repetitions, dtype=float)
    if n_blocks == 1:
        block = blocks[0]
        indices = rng.integers(0, len(block), size=(repetitions, len(block)))
        reference_samples = block[indices, 0].mean(axis=1)
        candidate_samples = block[indices, 1].mean(axis=1)
        scope = "paired_iterations_only"
    else:
        for _ in range(n_blocks):
            selected = rng.integers(0, n_blocks, size=repetitions)
            reference_contribution = np.empty(repetitions, dtype=float)
            candidate_contribution = np.empty(repetitions, dtype=float)
            for block_index, block in enumerate(blocks):
                rows = np.flatnonzero(selected == block_index)
                if len(rows) == 0:
                    continue
                indices = rng.integers(
                    0, len(block), size=(len(rows), len(block))
                )
                reference_contribution[rows] = block[indices, 0].mean(axis=1)
                candidate_contribution[rows] = block[indices, 1].mean(axis=1)
            reference_samples += reference_contribution / n_blocks
            candidate_samples += candidate_contribution / n_blocks
        scope = "paired_hierarchical_block_and_iteration"
    difference = reference_estimate - candidate_estimate
    differences = reference_samples - candidate_samples
    relative = difference / reference_estimate
    relatives = differences / reference_samples
    low, high = np.quantile(differences, [0.025, 0.975])
    relative_low, relative_high = np.quantile(relatives, [0.025, 0.975])
    block_means = [block.mean(axis=0) for block in blocks]
    block_differences = np.array([row[0] - row[1] for row in block_means])
    return {
        "n_blocks": n_blocks,
        "n_per_block_min": min(len(block) for block in blocks),
        "uncertainty_scope": scope,
        "reference_estimate": reference_estimate,
        "candidate_estimate": candidate_estimate,
        "reference_minus_candidate": difference,
        "difference_ci_low": float(low),
        "difference_ci_high": float(high),
        "relative_difference": relative,
        "relative_ci_low": float(relative_low),
        "relative_ci_high": float(relative_high),
        "positive_blocks": int((block_differences > 0).sum()),
        "zero_blocks": int((block_differences == 0).sum()),
        "negative_blocks": int((block_differences < 0).sum()),
    }


def build_strategy_contrasts(
    iterations: pd.DataFrame,
    comparisons: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "structure_level", "evidence_scope", "block_kind"]
    offset = 0
    for key, group in iterations.groupby(keys, sort=True, dropna=False):
        index = ["block_id", "num_iter"]
        for metric in (OBJECTIVE_NAME, "peak_new_selfish_ratio"):
            wide = group.pivot(index=index, columns="strategy", values=metric).reset_index()
            for comparison in comparisons:
                reference = str(comparison["reference"])
                candidate = str(comparison["candidate"])
                result = _paired_effect(
                    wide,
                    reference_column=reference,
                    candidate_column=candidate,
                    repetitions=repetitions,
                    seed=seed + offset,
                )
                rows.append(
                    {
                        **dict(zip(keys, key, strict=True)),
                        "metric": metric,
                        "comparison_id": str(comparison["id"]),
                        "reference": reference,
                        "candidate": candidate,
                        "estimand": "combined_strategy_contrast_not_isolated_parameter",
                        **result,
                    }
                )
                offset += 1
    return pd.DataFrame(rows)


def build_paired_structure_contrasts(
    iterations: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    for dataset in protocol["datasets"]:
        reference = dataset.get("paired_structure_reference")
        if reference is None:
            continue
        subset = iterations[iterations["dataset"] == dataset["id"]]
        levels = sorted(set(subset["structure_level"]))
        if reference not in levels:
            raise ValueError(
                f"paired structure reference {reference!r} is missing from "
                f"{dataset['id']}"
            )
        for strategy in REQUIRED_STRATEGIES:
            strategy_data = subset[subset["strategy"] == strategy]
            for metric in (OBJECTIVE_NAME, "peak_new_selfish_ratio"):
                wide = strategy_data.pivot(
                    index=["block_id", "num_iter"],
                    columns="structure_level",
                    values=metric,
                ).reset_index()
                for candidate in levels:
                    if candidate == reference:
                        continue
                    result = _paired_effect(
                        wide,
                        reference_column=str(reference),
                        candidate_column=str(candidate),
                        repetitions=repetitions,
                        seed=seed + offset,
                    )
                    rows.append(
                        {
                            "dataset": str(dataset["id"]),
                            "evidence_scope": str(dataset["evidence_scope"]),
                            "strategy": strategy,
                            "metric": metric,
                            "reference_structure": str(reference),
                            "candidate_structure": str(candidate),
                            "estimand": "controlled_structure_variant_contrast",
                            **result,
                        }
                    )
                    offset += 1
    return pd.DataFrame(rows)


def build_strategy_sensitivity(
    iterations: pd.DataFrame, *, repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block = (
        iterations.groupby(
            [
                "dataset",
                "evidence_scope",
                "structure_level",
                "block_kind",
                "block_id",
                "strategy",
            ],
            sort=True,
        )[[OBJECTIVE_NAME, "peak_new_selfish_ratio"]]
        .mean()
        .reset_index()
    )
    index = [
        "dataset",
        "evidence_scope",
        "structure_level",
        "block_kind",
        "block_id",
    ]
    jcum = block.pivot(index=index, columns="strategy", values=OBJECTIVE_NAME).reset_index()
    jpeak = block.pivot(
        index=index, columns="strategy", values="peak_new_selfish_ratio"
    ).reset_index()
    detail = jcum[index].copy()
    for strategy in REQUIRED_STRATEGIES:
        detail[f"jcum_{strategy}"] = jcum[strategy]
        detail[f"jpeak_{strategy}"] = jpeak[strategy]
    detail["jcum_strategy_range"] = jcum[list(REQUIRED_STRATEGIES)].max(axis=1) - jcum[
        list(REQUIRED_STRATEGIES)
    ].min(axis=1)
    detail["jpeak_strategy_range"] = jpeak[list(REQUIRED_STRATEGIES)].max(axis=1) - jpeak[
        list(REQUIRED_STRATEGIES)
    ].min(axis=1)
    detail["jcum_lowest_strategy"] = jcum[list(REQUIRED_STRATEGIES)].idxmin(axis=1)
    detail["jpeak_lowest_strategy"] = jpeak[list(REQUIRED_STRATEGIES)].idxmin(axis=1)

    rows: list[dict[str, Any]] = []
    group_keys = ["dataset", "evidence_scope", "structure_level", "block_kind"]
    for offset, (key, group) in enumerate(detail.groupby(group_keys, sort=True)):
        if len(group) >= 2:
            jcum_estimate = _bootstrap_mean(
                [group["jcum_strategy_range"].to_numpy(float)],
                repetitions=repetitions,
                seed=seed + offset * 2,
            )
            jpeak_estimate = _bootstrap_mean(
                [group["jpeak_strategy_range"].to_numpy(float)],
                repetitions=repetitions,
                seed=seed + offset * 2 + 1,
            )
            jcum_low = jcum_estimate["ci_low"]
            jcum_high = jcum_estimate["ci_high"]
            jpeak_low = jpeak_estimate["ci_low"]
            jpeak_high = jpeak_estimate["ci_high"]
            interval_scope = "descriptive_resampling_of_block_means"
        else:
            jcum_low = None
            jcum_high = None
            jpeak_low = None
            jpeak_high = None
            interval_scope = "single_block_no_block_interval"
        # The rows are already block-level means, so no lower-level resampling is
        # available here.  The interval only summarizes observed block variation.
        rows.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "n_blocks": int(len(group)),
                "mean_jcum_strategy_range": float(
                    group["jcum_strategy_range"].mean()
                ),
                "min_jcum_strategy_range": float(
                    group["jcum_strategy_range"].min()
                ),
                "max_jcum_strategy_range": float(
                    group["jcum_strategy_range"].max()
                ),
                "mean_jpeak_strategy_range": float(
                    group["jpeak_strategy_range"].mean()
                ),
                "jcum_block_interval_low": jcum_low,
                "jcum_block_interval_high": jcum_high,
                "jpeak_block_interval_low": jpeak_low,
                "jpeak_block_interval_high": jpeak_high,
                "interval_scope": interval_scope,
            }
        )
    return detail, pd.DataFrame(rows)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def build_structure_metric_associations(runs: pd.DataFrame) -> pd.DataFrame:
    outcomes = (
        runs.groupby(
            ["dataset", "network", "structure_level", "strategy"], sort=True
        )["jcum"]
        .mean()
        .unstack("strategy")
        .reset_index()
    )
    outcomes["mean_across_strategies"] = outcomes[list(REQUIRED_STRATEGIES)].mean(
        axis=1
    )
    outcomes["strategy_range"] = outcomes[list(REQUIRED_STRATEGIES)].max(
        axis=1
    ) - outcomes[list(REQUIRED_STRATEGIES)].min(axis=1)
    metadata = runs[["dataset", "network", *STRUCTURAL_METRICS]].drop_duplicates(
        ["dataset", "network"]
    )
    frame = outcomes.merge(
        metadata, on=["dataset", "network"], how="left", validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    outcome_columns = [*REQUIRED_STRATEGIES, "mean_across_strategies", "strategy_range"]
    for dataset, group in frame.groupby("dataset", sort=True):
        for metric in STRUCTURAL_METRICS:
            for outcome in outcome_columns:
                valid = group[[metric, outcome]].dropna()
                if (
                    len(valid) < 3
                    or valid[metric].nunique() < 3
                    or valid[outcome].nunique() < 2
                ):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "structural_metric": metric,
                        "outcome": outcome,
                        "n_network_instances": int(len(valid)),
                        "spearman_rho": _spearman(valid[metric], valid[outcome]),
                        "interpretation_scope": "descriptive_no_causal_or_multiplicity_claim",
                    }
                )
    return pd.DataFrame(rows)


def load_formal_fixed_effects(
    protocol: dict[str, Any], *, repo_root: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(repo_root).resolve()
    source = protocol["formal_fixed_effect_source"]
    manifest_path = _repo_path(root, source["analysis_manifest"])
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("formal Stage 4 source analysis is not completed")
    selected = set(str(value) for value in source["selected_conditions"])
    condition_summary = pd.read_csv(_repo_path(root, source["condition_summary"]))
    condition_summary = condition_summary[
        condition_summary["condition_id"].isin(selected | {"none"})
    ].copy()
    effects = pd.read_csv(_repo_path(root, source["all_enabled_vs_none"]))
    effects = effects[effects["candidate"].isin(selected)].copy()
    condition_summary.insert(0, "source_stage", "stage4_fixed_confirmation")
    effects.insert(0, "source_stage", "stage4_fixed_confirmation")
    return condition_summary.reset_index(drop=True), effects.reset_index(drop=True)


def build_decision(
    data: StructureReanalysisData,
    strategy_contrasts: pd.DataFrame,
    paired_structure_contrasts: pd.DataFrame,
    formal_effects: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "status": "existing_structure_reanalysis_complete",
        "raw_run_count": int(len(data.runs)),
        "valid_raw_run_count": int(data.audit["valid"].sum()),
        "old_structure_runs_include_no_intervention": False,
        "eta_available_from_old_structure_runs": False,
        "formal_cross_network_eta_available": not formal_effects.empty,
        "strategy_contrast_count": int(len(strategy_contrasts)),
        "paired_structure_contrast_count": int(len(paired_structure_contrasts)),
        "interpretation_rules": [
            "Legacy structure runs support Jcum, Jpeak, and combined strategy contrasts only.",
            "The three legacy strategies do not isolate certainty and effectiveness because both coordinates change.",
            "Synthetic-network conclusions must retain network-seed uncertainty.",
            "Observed Facebook and Wiki-vote results apply to those fixed observed graphs.",
            "Rewiring changes multiple structural metrics and cannot identify one causal metric.",
        ],
        "additional_experiment_decision": "targeted_no_intervention_or_current_fixed_condition_confirmation_required_before_structure_specific_eta_claims",
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_structure_reanalysis(
    protocol: dict[str, Any],
    *,
    repo_root: str | Path,
    output_root: str | Path,
    repetitions: int | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Stage 9 output already exists: {output}")
    tables = output / "tables"
    tables.mkdir(parents=True)

    bootstrap_repetitions = int(
        repetitions or protocol["bootstrap"]["repetitions"]
    )
    bootstrap_seed = int(protocol["bootstrap"]["seed"])
    expected_iterations = int(protocol["expected_iterations_per_run"])

    specs = discover_legacy_runs(protocol, repo_root=repo)
    data = load_legacy_structure_data(
        specs, expected_iterations=expected_iterations
    )
    level_summary = build_level_summary(
        data.iterations,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    strategy_contrasts = build_strategy_contrasts(
        data.iterations,
        protocol["strategy_comparisons"],
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 10000,
    )
    paired_structure_contrasts = build_paired_structure_contrasts(
        data.iterations,
        protocol,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 20000,
    )
    sensitivity_detail, sensitivity_summary = build_strategy_sensitivity(
        data.iterations,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 30000,
    )
    associations = build_structure_metric_associations(data.runs)
    formal_conditions, formal_effects = load_formal_fixed_effects(
        protocol, repo_root=repo
    )
    decision = build_decision(
        data,
        strategy_contrasts,
        paired_structure_contrasts,
        formal_effects,
    )

    data.iterations.to_parquet(tables / "iteration_metrics.parquet", index=False)
    data.runs.to_csv(tables / "run_inventory.csv", index=False)
    data.audit.to_csv(tables / "data_audit.csv", index=False)
    data.dataset_scope.to_csv(tables / "dataset_scope.csv", index=False)
    level_summary.to_csv(tables / "level_summary.csv", index=False)
    strategy_contrasts.to_csv(tables / "strategy_contrasts.csv", index=False)
    paired_structure_contrasts.to_csv(
        tables / "paired_structure_contrasts.csv", index=False
    )
    sensitivity_detail.to_csv(
        tables / "strategy_sensitivity_detail.csv", index=False
    )
    sensitivity_summary.to_csv(
        tables / "strategy_sensitivity_summary.csv", index=False
    )
    associations.to_csv(tables / "structure_metric_associations.csv", index=False)
    formal_conditions.to_csv(
        tables / "formal_cross_network_conditions.csv", index=False
    )
    formal_effects.to_csv(tables / "formal_cross_network_effects.csv", index=False)

    summary = {
        "raw_run_count": int(len(data.runs)),
        "valid_raw_run_count": int(data.audit["valid"].sum()),
        "iteration_metric_count": int(len(data.iterations)),
        "dataset_count": int(data.runs["dataset"].nunique()),
        "network_instance_count": int(
            data.runs[["dataset", "network"]].drop_duplicates().shape[0]
        ),
        "strategy_contrast_count": int(len(strategy_contrasts)),
        "paired_structure_contrast_count": int(len(paired_structure_contrasts)),
        "formal_cross_network_effect_count": int(len(formal_effects)),
        "bootstrap_repetitions": bootstrap_repetitions,
    }
    manifest = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "objective_name": OBJECTIVE_NAME,
        "objective_definition_version": OBJECTIVE_DEFINITION_VERSION,
        "analysis_summary": summary,
        "limitations": decision["interpretation_rules"],
    }
    write_json(output / "analysis_manifest.json", manifest)
    write_json(output / "analysis_summary.json", summary)
    write_json(output / "decision.json", decision)
    return {
        "output_root": output,
        "manifest": manifest,
        "summary": summary,
        "decision": decision,
    }
