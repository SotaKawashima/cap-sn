"""Audit and analyze the targeted Stage 9 structure confirmation experiment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from analysis.optimization_metrics import (
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
    compute_selfish_metrics,
)
STAGE = "stage9_structure_confirmation"
LFR_FAMILY = "lfr_community"
REWIRE_FAMILY = "facebook_degree_rewire"
INTERVENTION_IDS = ("legacy_balance", "prior_high")


@dataclass(frozen=True)
class StructureConfirmationData:
    iterations: pd.DataFrame
    runs: pd.DataFrame
    audit: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _matches(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _manifest_errors(manifest: dict[str, Any], spec: Any) -> list[str]:
    network = manifest.get("network", {})
    intervention = manifest.get("intervention", {})
    runtime = manifest.get("runtime", {})
    objective = manifest.get("objective", {})
    expected = {
        "stage": STAGE,
        "run_type": "fixed_condition",
        "network_id": spec.network,
        "network_seed": spec.network_generation_seed,
        "num_agents": int(spec.num_agents),
        "condition_id": spec.condition_id,
        "intervention_enabled": bool(spec.intervention_enabled),
        "simulator_seed": int(spec.simulator_seed),
        "iterations": int(spec.iterations),
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_DEFINITION_VERSION,
    }
    observed = {
        "stage": manifest.get("stage"),
        "run_type": manifest.get("run_type"),
        "network_id": network.get("id"),
        "network_seed": network.get("network_seed"),
        "num_agents": network.get("num_agents"),
        "condition_id": intervention.get("condition_id"),
        "intervention_enabled": intervention.get("enabled"),
        "simulator_seed": runtime.get("simulator_seed"),
        "iterations": runtime.get("iteration_count"),
        "objective_name": objective.get("name"),
        "objective_version": objective.get("definition_version"),
    }
    errors = [
        f"{field}={observed[field]!r}, expected={value!r}"
        for field, value in expected.items()
        if observed[field] != value
    ]
    if network.get("sha256") != spec.network_config_sha256:
        errors.append("network config SHA-256 does not match the protocol")

    applied = intervention.get("applied_parameters")
    if spec.intervention_enabled:
        if not isinstance(applied, dict):
            errors.append("applied_parameters is missing")
        else:
            for field in ("certainty", "effectiveness"):
                if not _matches(applied.get(field), getattr(spec, field)):
                    errors.append(
                        f"applied_{field}={applied.get(field)!r}, "
                        f"expected={getattr(spec, field)!r}"
                    )
        if spec.opinion_csv is not None:
            if intervention.get("opinion_mode") != "existing_csv":
                errors.append("prior_high did not use the existing opinion CSV")
            for field in ("opinion_source", "opinion_csv"):
                entry = intervention.get(field) or {}
                if entry.get("sha256") != spec.opinion_sha256:
                    errors.append(f"{field} SHA-256 does not match prior_high")
    elif applied is not None:
        errors.append("no-intervention run has applied parameters")
    return errors


def _metric_errors(
    saved: pd.DataFrame,
    recalculated: pd.DataFrame,
    *,
    expected_iterations: int,
    tolerance: float,
) -> list[str]:
    if len(saved) != expected_iterations:
        return [
            f"metrics row count={len(saved)}, expected={expected_iterations}"
        ]
    if "num_iter" not in saved:
        return ["metrics.csv has no num_iter column"]
    left = saved.sort_values("num_iter").reset_index(drop=True)
    right = recalculated.sort_values("num_iter").reset_index(drop=True)
    if left["num_iter"].astype(int).tolist() != right["num_iter"].astype(int).tolist():
        return ["metrics iteration IDs do not match recalculation"]
    errors: list[str] = []
    for column in right.columns:
        if column not in left:
            errors.append(f"metrics.csv is missing {column}")
            continue
        if pd.api.types.is_numeric_dtype(right[column]):
            observed = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
            expected = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
            if not np.allclose(
                observed,
                expected,
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            ):
                errors.append(f"metrics column {column} differs from raw pop")
    return errors


def load_structure_confirmation(
    experiment_root: str | Path,
    *,
    expected_specs: Sequence[Any],
    numeric_tolerance: float = 1e-12,
) -> StructureConfirmationData:
    """Load all runs and independently reproduce current metrics from pop.arrow."""

    root = Path(experiment_root).resolve()
    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in expected_specs:
        run_dir = root / spec.relative_run_dir
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.csv"
        pop_path = run_dir / "pop.arrow"
        audit: dict[str, Any] = {
            "run_key": spec.key,
            "family": spec.family,
            "network": spec.network,
            "condition_id": spec.condition_id,
            "manifest_valid": False,
            "metrics_reproduced": False,
            "pop_valid": False,
            "valid": False,
            "error": None,
        }
        try:
            missing = [
                path.name
                for path in (manifest_path, metrics_path, pop_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(f"missing completed outputs: {missing}")
            manifest = _read_json(manifest_path)
            errors = _manifest_errors(manifest, spec)
            if manifest.get("status") != "completed":
                errors.append(f"status={manifest.get('status')!r}, expected='completed'")
            if errors:
                raise ValueError("; ".join(errors))
            audit["manifest_valid"] = True

            pop = pd.read_feather(pop_path)
            metrics = compute_selfish_metrics(
                pop,
                num_agents=int(spec.num_agents),
                expected_iterations=int(spec.iterations),
            )
            audit["pop_valid"] = True
            saved = pd.read_csv(metrics_path)
            errors = _metric_errors(
                saved,
                metrics.per_iteration,
                expected_iterations=int(spec.iterations),
                tolerance=numeric_tolerance,
            )
            manifest_value = manifest.get("objective", {}).get("value")
            if not _matches(
                manifest_value, metrics.objective_value, numeric_tolerance
            ):
                errors.append("manifest objective differs from raw-pop recalculation")
            if errors:
                raise ValueError("; ".join(errors))
            audit["metrics_reproduced"] = True

            identifiers = {
                "run_key": spec.key,
                "family": spec.family,
                "network": spec.network,
                "structure_level": spec.structure_level,
                "structure_value": float(spec.structure_value),
                "network_seed_index": spec.network_seed_index,
                "network_generation_seed": spec.network_generation_seed,
                "simulator_seed": int(spec.simulator_seed),
                "condition_id": spec.condition_id,
                "condition_role": spec.condition_role,
                "intervention_enabled": bool(spec.intervention_enabled),
                "certainty": spec.certainty,
                "effectiveness": spec.effectiveness,
                "num_agents": int(spec.num_agents),
                "avg_degree": float(spec.avg_degree),
                "avg_clustering": float(spec.avg_clustering),
                "modularity": float(spec.modularity),
                "internal_edge_ratio": float(spec.internal_edge_ratio),
            }
            frame = metrics.per_iteration.copy()
            for position, (field, value) in enumerate(identifiers.items()):
                frame.insert(position, field, value)
            iteration_frames.append(frame)
            run_rows.append(
                {
                    **identifiers,
                    "n_iterations": int(metrics.summary["n_iterations"]),
                    "jcum": float(metrics.summary[OBJECTIVE_NAME]),
                    "jpeak": float(metrics.summary["peak_new_selfish_ratio"]),
                    "pop_path": pop_path.as_posix(),
                    "manifest_path": manifest_path.as_posix(),
                }
            )
            audit["valid"] = True
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"{spec.key}: {audit['error']}")
        audit_rows.append(audit)

    if failures:
        raise ValueError(
            f"Stage 9 confirmation audit failed for {len(failures)} runs:\n"
            + "\n".join(failures[:25])
        )
    iterations = pd.concat(iteration_frames, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    audit = pd.DataFrame(audit_rows)
    return StructureConfirmationData(
        iterations=iterations.sort_values(
            ["family", "structure_value", "network", "condition_id", "simulator_seed", "num_iter"]
        ).reset_index(drop=True),
        runs=runs.sort_values("run_key").reset_index(drop=True),
        audit=audit.sort_values("run_key").reset_index(drop=True),
    )


def _two_level_samples(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = [
        group.sort_values("num_iter")[list(value_columns)].to_numpy(float)
        for _, group in frame.groupby("simulator_seed", sort=True)
    ]
    if len(blocks) < 2 or any(len(block) < 2 for block in blocks):
        raise ValueError("two-level bootstrap requires at least two seed blocks")
    estimate = np.mean([block.mean(axis=0) for block in blocks], axis=0)
    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    samples = np.zeros((repetitions, len(value_columns)), dtype=float)
    for _ in range(n_blocks):
        selected = rng.integers(0, n_blocks, size=repetitions)
        contribution = np.empty_like(samples)
        for block_index, block in enumerate(blocks):
            rows = np.flatnonzero(selected == block_index)
            if len(rows) == 0:
                continue
            indices = rng.integers(0, len(block), size=(len(rows), len(block)))
            contribution[rows] = block[indices].mean(axis=1)
        samples += contribution / n_blocks
    return estimate, samples


def _three_level_samples(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    networks: list[list[np.ndarray]] = []
    for _, network_group in frame.groupby("network", sort=True):
        seed_blocks = [
            group.sort_values("num_iter")[list(value_columns)].to_numpy(float)
            for _, group in network_group.groupby("simulator_seed", sort=True)
        ]
        if len(seed_blocks) < 2 or any(len(block) < 2 for block in seed_blocks):
            raise ValueError(
                "three-level bootstrap requires at least two simulator seeds per network"
            )
        networks.append(seed_blocks)
    if len(networks) < 2:
        raise ValueError("three-level bootstrap requires at least two networks")

    estimate = np.mean(
        [
            np.mean([block.mean(axis=0) for block in seed_blocks], axis=0)
            for seed_blocks in networks
        ],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    n_networks = len(networks)
    samples = np.zeros((repetitions, len(value_columns)), dtype=float)
    for _ in range(n_networks):
        selected_networks = rng.integers(0, n_networks, size=repetitions)
        network_contribution = np.empty_like(samples)
        for network_index, seed_blocks in enumerate(networks):
            rows = np.flatnonzero(selected_networks == network_index)
            if len(rows) == 0:
                continue
            n_seed_blocks = len(seed_blocks)
            selected_seed_means = np.zeros(
                (len(rows), len(value_columns)), dtype=float
            )
            for _ in range(n_seed_blocks):
                selected_seeds = rng.integers(0, n_seed_blocks, size=len(rows))
                seed_contribution = np.empty_like(selected_seed_means)
                for seed_index, block in enumerate(seed_blocks):
                    local_rows = np.flatnonzero(selected_seeds == seed_index)
                    if len(local_rows) == 0:
                        continue
                    indices = rng.integers(
                        0,
                        len(block),
                        size=(len(local_rows), len(block)),
                    )
                    seed_contribution[local_rows] = block[indices].mean(axis=1)
                selected_seed_means += seed_contribution / n_seed_blocks
            network_contribution[rows] = selected_seed_means
        samples += network_contribution / n_networks
    return estimate, samples


def _hierarchical_samples(
    frame: pd.DataFrame,
    *,
    family: str,
    value_columns: Sequence[str],
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if family == LFR_FAMILY:
        estimate, samples = _three_level_samples(
            frame,
            value_columns=value_columns,
            repetitions=repetitions,
            seed=seed,
        )
        return estimate, samples, "network_seed_simulator_seed_iteration"
    if family == REWIRE_FAMILY:
        estimate, samples = _two_level_samples(
            frame,
            value_columns=value_columns,
            repetitions=repetitions,
            seed=seed,
        )
        return estimate, samples, "simulator_seed_iteration_fixed_graph"
    raise ValueError(f"unsupported structure family: {family}")


def _interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def build_structure_level_summary(
    iterations: pd.DataFrame, *, repetitions: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "family",
        "structure_level",
        "structure_value",
        "condition_id",
        "condition_role",
        "intervention_enabled",
        "certainty",
        "effectiveness",
        "num_agents",
    ]
    for offset, (key, group) in enumerate(
        iterations.groupby(keys, sort=True, dropna=False)
    ):
        family = str(key[0])
        estimate, samples, scope = _hierarchical_samples(
            group,
            family=family,
            value_columns=[OBJECTIVE_NAME, "peak_new_selfish_ratio"],
            repetitions=repetitions,
            seed=seed + offset,
        )
        jcum_low, jcum_high = _interval(samples[:, 0])
        jpeak_low, jpeak_high = _interval(samples[:, 1])
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "network_instance_count": int(group["network"].nunique()),
                "simulator_seed_count": int(group["simulator_seed"].nunique()),
                "iteration_count": int(len(group)),
                "uncertainty_scope": scope,
                "jcum_estimate": float(estimate[0]),
                "jcum_ci_low": jcum_low,
                "jcum_ci_high": jcum_high,
                "jpeak_estimate": float(estimate[1]),
                "jpeak_ci_low": jpeak_low,
                "jpeak_ci_high": jpeak_high,
                "mean_avg_degree": float(group["avg_degree"].mean()),
                "mean_avg_clustering": float(group["avg_clustering"].mean()),
                "mean_modularity": float(group["modularity"].mean()),
                "mean_internal_edge_ratio": float(
                    group["internal_edge_ratio"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _wide_conditions(group: pd.DataFrame, candidate: str) -> pd.DataFrame:
    selected = group[group["condition_id"].isin(["none", candidate])]
    key = ["network", "simulator_seed", "num_iter"]
    if selected.duplicated(key + ["condition_id"]).any():
        raise ValueError("structure confirmation contains duplicate paired rows")
    wide = selected.pivot(
        index=key,
        columns="condition_id",
        values=[OBJECTIVE_NAME, "peak_new_selfish_ratio"],
    )
    expected = {
        (OBJECTIVE_NAME, "none"),
        (OBJECTIVE_NAME, candidate),
        ("peak_new_selfish_ratio", "none"),
        ("peak_new_selfish_ratio", candidate),
    }
    if not expected.issubset(wide.columns):
        raise ValueError(f"paired conditions are incomplete for {candidate}")
    wide.columns = [f"{metric}__{condition}" for metric, condition in wide.columns]
    return wide.reset_index()


def _paired_effect(
    group: pd.DataFrame,
    *,
    family: str,
    candidate: str,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    wide = _wide_conditions(group, candidate)
    columns = [
        f"{OBJECTIVE_NAME}__none",
        f"{OBJECTIVE_NAME}__{candidate}",
        "peak_new_selfish_ratio__none",
        f"peak_new_selfish_ratio__{candidate}",
    ]
    estimate, samples, scope = _hierarchical_samples(
        wide,
        family=family,
        value_columns=columns,
        repetitions=repetitions,
        seed=seed,
    )
    reference, intervention, peak_reference, peak_intervention = estimate
    delta = float(reference - intervention)
    eta = float(delta / reference)
    peak_delta = float(peak_reference - peak_intervention)
    delta_samples = samples[:, 0] - samples[:, 1]
    eta_samples = delta_samples / samples[:, 0]
    peak_delta_samples = samples[:, 2] - samples[:, 3]
    delta_low, delta_high = _interval(delta_samples)
    eta_low, eta_high = _interval(eta_samples)
    peak_low, peak_high = _interval(peak_delta_samples)
    if delta_low > 0:
        interpretation = "reduction"
    elif delta_high < 0:
        interpretation = "increase"
    elif delta > 0:
        interpretation = "reduction_tendency_with_uncertainty"
    else:
        interpretation = "no_clear_reduction"
    network_deltas = (
        wide.groupby("network", sort=True)[columns[:2]].mean().assign(
            delta=lambda frame: frame[columns[0]] - frame[columns[1]]
        )["delta"]
    )
    simulator_deltas = (
        wide.groupby(["network", "simulator_seed"], sort=True)[columns[:2]]
        .mean()
        .assign(delta=lambda frame: frame[columns[0]] - frame[columns[1]])[
            "delta"
        ]
    )
    result = {
        "uncertainty_scope": scope,
        "network_instance_count": int(wide["network"].nunique()),
        "simulator_seed_block_count": int(
            wide[["network", "simulator_seed"]].drop_duplicates().shape[0]
        ),
        "positive_network_instances": int((network_deltas > 0).sum()),
        "zero_network_instances": int((network_deltas == 0).sum()),
        "negative_network_instances": int((network_deltas < 0).sum()),
        "positive_simulator_seed_blocks": int((simulator_deltas > 0).sum()),
        "zero_simulator_seed_blocks": int((simulator_deltas == 0).sum()),
        "negative_simulator_seed_blocks": int((simulator_deltas < 0).sum()),
        "reference_estimate": float(reference),
        "intervention_estimate": float(intervention),
        "absolute_suppression": delta,
        "absolute_ci_low": delta_low,
        "absolute_ci_high": delta_high,
        "relative_suppression": eta,
        "relative_ci_low": eta_low,
        "relative_ci_high": eta_high,
        "peak_reference_estimate": float(peak_reference),
        "peak_intervention_estimate": float(peak_intervention),
        "peak_absolute_suppression": peak_delta,
        "peak_absolute_ci_low": peak_low,
        "peak_absolute_ci_high": peak_high,
        "interpretation": interpretation,
    }
    return result, {
        "reference": samples[:, 0],
        "intervention": samples[:, 1],
        "delta": delta_samples,
        "eta": eta_samples,
        "peak_delta": peak_delta_samples,
    }


def build_intervention_effects(
    iterations: pd.DataFrame, *, repetitions: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    for (family, level, value), group in iterations.groupby(
        ["family", "structure_level", "structure_value"], sort=True
    ):
        for candidate in INTERVENTION_IDS:
            result, _ = _paired_effect(
                group,
                family=str(family),
                candidate=candidate,
                repetitions=repetitions,
                seed=seed + offset,
            )
            num_agents = int(group["num_agents"].iloc[0])
            rows.append(
                {
                    "family": family,
                    "structure_level": level,
                    "structure_value": float(value),
                    "reference": "none",
                    "candidate": candidate,
                    "num_agents": num_agents,
                    **result,
                    "equivalent_agents": result["absolute_suppression"]
                    * num_agents,
                    "equivalent_agents_ci_low": result["absolute_ci_low"]
                    * num_agents,
                    "equivalent_agents_ci_high": result["absolute_ci_high"]
                    * num_agents,
                    "mean_avg_degree": float(group["avg_degree"].mean()),
                    "mean_avg_clustering": float(
                        group["avg_clustering"].mean()
                    ),
                    "mean_modularity": float(group["modularity"].mean()),
                    "mean_internal_edge_ratio": float(
                        group["internal_edge_ratio"].mean()
                    ),
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def build_structure_effect_contrasts(
    iterations: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    reference_levels = {
        str(item["id"]): str(item["reference_level"])
        for item in protocol["design"]["network_families"]
    }
    rows: list[dict[str, Any]] = []
    offset = 0
    for family, family_group in iterations.groupby("family", sort=True):
        reference_level = reference_levels[str(family)]
        level_groups = {
            str(level): group
            for level, group in family_group.groupby("structure_level", sort=True)
        }
        for candidate in INTERVENTION_IDS:
            reference_result, reference_samples = _paired_effect(
                level_groups[reference_level],
                family=str(family),
                candidate=candidate,
                repetitions=repetitions,
                seed=seed + offset,
            )
            offset += 1
            for level, group in level_groups.items():
                if level == reference_level:
                    continue
                candidate_result, candidate_samples = _paired_effect(
                    group,
                    family=str(family),
                    candidate=candidate,
                    repetitions=repetitions,
                    seed=seed + offset,
                )
                offset += 1
                baseline_difference = (
                    candidate_result["reference_estimate"]
                    - reference_result["reference_estimate"]
                )
                delta_difference = (
                    candidate_result["absolute_suppression"]
                    - reference_result["absolute_suppression"]
                )
                eta_difference = (
                    candidate_result["relative_suppression"]
                    - reference_result["relative_suppression"]
                )
                peak_difference = (
                    candidate_result["peak_absolute_suppression"]
                    - reference_result["peak_absolute_suppression"]
                )
                baseline_samples = (
                    candidate_samples["reference"]
                    - reference_samples["reference"]
                )
                delta_samples = (
                    candidate_samples["delta"] - reference_samples["delta"]
                )
                eta_samples = candidate_samples["eta"] - reference_samples["eta"]
                peak_samples = (
                    candidate_samples["peak_delta"]
                    - reference_samples["peak_delta"]
                )
                baseline_low, baseline_high = _interval(baseline_samples)
                delta_low, delta_high = _interval(delta_samples)
                eta_low, eta_high = _interval(eta_samples)
                peak_low, peak_high = _interval(peak_samples)
                if eta_low > 0:
                    interpretation = "more_suppressive_than_reference_structure"
                elif eta_high < 0:
                    interpretation = "less_suppressive_than_reference_structure"
                else:
                    interpretation = "no_clear_difference_in_suppression"
                rows.append(
                    {
                        "family": family,
                        "reference_structure": reference_level,
                        "candidate_structure": level,
                        "reference_structure_value": float(
                            level_groups[reference_level]["structure_value"].iloc[0]
                        ),
                        "candidate_structure_value": float(
                            group["structure_value"].iloc[0]
                        ),
                        "condition_id": candidate,
                        "baseline_jcum_difference_candidate_minus_reference": baseline_difference,
                        "baseline_jcum_difference_ci_low": baseline_low,
                        "baseline_jcum_difference_ci_high": baseline_high,
                        "absolute_suppression_difference_candidate_minus_reference": delta_difference,
                        "absolute_suppression_difference_ci_low": delta_low,
                        "absolute_suppression_difference_ci_high": delta_high,
                        "relative_suppression_difference_candidate_minus_reference": eta_difference,
                        "relative_suppression_difference_ci_low": eta_low,
                        "relative_suppression_difference_ci_high": eta_high,
                        "peak_suppression_difference_candidate_minus_reference": peak_difference,
                        "peak_suppression_difference_ci_low": peak_low,
                        "peak_suppression_difference_ci_high": peak_high,
                        "interpretation": interpretation,
                        "causal_metric_identified": False,
                    }
                )
    return pd.DataFrame(rows)


def build_network_instance_effects(
    iterations: pd.DataFrame, *, repetitions: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    for (family, network, level, value), group in iterations.groupby(
        ["family", "network", "structure_level", "structure_value"], sort=True
    ):
        for candidate in INTERVENTION_IDS:
            result, _ = _paired_effect(
                group,
                family=REWIRE_FAMILY,
                candidate=candidate,
                repetitions=repetitions,
                seed=seed + offset,
            )
            rows.append(
                {
                    "family": family,
                    "network": network,
                    "structure_level": level,
                    "structure_value": float(value),
                    "network_seed_index": group["network_seed_index"].iloc[0],
                    "network_generation_seed": group[
                        "network_generation_seed"
                    ].iloc[0],
                    "candidate": candidate,
                    **result,
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def build_simulator_seed_effects(iterations: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        iterations.groupby(
            [
                "family",
                "network",
                "structure_level",
                "structure_value",
                "network_seed_index",
                "simulator_seed",
                "condition_id",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            mean_jcum=(OBJECTIVE_NAME, "mean"),
            mean_jpeak=("peak_new_selfish_ratio", "mean"),
        )
        .reset_index()
    )
    wide = grouped.pivot(
        index=[
            "family",
            "network",
            "structure_level",
            "structure_value",
            "network_seed_index",
            "simulator_seed",
        ],
        columns="condition_id",
        values=["mean_jcum", "mean_jpeak"],
    )
    rows: list[dict[str, Any]] = []
    for index, values in wide.iterrows():
        metadata = dict(zip(wide.index.names, index, strict=True))
        for candidate in INTERVENTION_IDS:
            reference = float(values[("mean_jcum", "none")])
            intervention = float(values[("mean_jcum", candidate)])
            delta = reference - intervention
            rows.append(
                {
                    **metadata,
                    "candidate": candidate,
                    "reference_jcum": reference,
                    "intervention_jcum": intervention,
                    "absolute_suppression": delta,
                    "relative_suppression": delta / reference,
                    "peak_absolute_suppression": float(
                        values[("mean_jpeak", "none")]
                        - values[("mean_jpeak", candidate)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_decision(
    effects: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> dict[str, Any]:
    family_rows: list[dict[str, Any]] = []
    for family, group in effects.groupby("family", sort=True):
        family_contrasts = contrasts[contrasts["family"] == family]
        family_rows.append(
            {
                "family": family,
                "clear_reduction_count": int((group["interpretation"] == "reduction").sum()),
                "clear_increase_count": int((group["interpretation"] == "increase").sum()),
                "uncertain_effect_count": int(
                    group["interpretation"].isin(
                        [
                            "reduction_tendency_with_uncertainty",
                            "no_clear_reduction",
                        ]
                    ).sum()
                ),
                "clear_structure_effect_difference_count": int(
                    (
                        family_contrasts["interpretation"]
                        != "no_clear_difference_in_suppression"
                    ).sum()
                ),
            }
        )
    return {
        "status": "structure_confirmation_complete",
        "families": family_rows,
        "interpretation_rules": [
            "Separate no-intervention baseline level from intervention effect.",
            "Treat Jcum as primary and Jpeak as secondary.",
            "Retain network-generation uncertainty for LFR conclusions.",
            "Treat Facebook rewiring as one fixed observed graph comparison.",
            "Do not attribute a rewiring result to one causal network metric.",
        ],
    }


def run_structure_confirmation_analysis(
    protocol: dict[str, Any],
    *,
    expected_specs: Sequence[Any],
    experiment_root: str | Path,
    output_root: str | Path,
    repetitions: int | None = None,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Stage 9 analysis output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    tables.mkdir()

    bootstrap = protocol["inference"]["bootstrap"]
    repetitions = int(repetitions or bootstrap["repetitions"])
    seed = int(bootstrap["seed"])
    data = load_structure_confirmation(
        experiment_root,
        expected_specs=expected_specs,
        numeric_tolerance=1e-12,
    )
    level_summary = build_structure_level_summary(
        data.iterations,
        repetitions=repetitions,
        seed=seed,
    )
    effects = build_intervention_effects(
        data.iterations,
        repetitions=repetitions,
        seed=seed + 1000,
    )
    contrasts = build_structure_effect_contrasts(
        data.iterations,
        protocol,
        repetitions=repetitions,
        seed=seed + 2000,
    )
    instance_effects = build_network_instance_effects(
        data.iterations,
        repetitions=repetitions,
        seed=seed + 3000,
    )
    seed_effects = build_simulator_seed_effects(data.iterations)
    decision = build_decision(effects, contrasts)

    data.iterations.to_parquet(tables / "iteration_metrics.parquet", index=False)
    data.runs.to_csv(tables / "run_inventory.csv", index=False)
    data.audit.to_csv(tables / "data_audit.csv", index=False)
    level_summary.to_csv(tables / "structure_level_summary.csv", index=False)
    effects.to_csv(tables / "intervention_effects.csv", index=False)
    contrasts.to_csv(tables / "structure_effect_contrasts.csv", index=False)
    instance_effects.to_csv(tables / "network_instance_effects.csv", index=False)
    seed_effects.to_csv(tables / "simulator_seed_effects.csv", index=False)

    summary = {
        "run_count": int(len(data.runs)),
        "valid_run_count": int(data.audit["valid"].sum()),
        "iteration_metric_count": int(len(data.iterations)),
        "family_count": int(data.runs["family"].nunique()),
        "network_instance_count": int(data.runs["network"].nunique()),
        "structure_level_count": int(
            data.runs[["family", "structure_level"]].drop_duplicates().shape[0]
        ),
        "intervention_effect_count": int(len(effects)),
        "structure_effect_contrast_count": int(len(contrasts)),
        "bootstrap_repetitions": repetitions,
    }
    manifest = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "experiment_root": Path(experiment_root).resolve().as_posix(),
        "objective": {
            "name": OBJECTIVE_NAME,
            "definition_version": OBJECTIVE_DEFINITION_VERSION,
        },
        "bootstrap": {
            "method": bootstrap["method"],
            "repetitions": repetitions,
            "seed": seed,
        },
        "summary": summary,
        "outputs": sorted(path.name for path in tables.iterdir()),
    }
    for name, value in (
        ("analysis_manifest.json", manifest),
        ("analysis_summary.json", summary),
        ("decision.json", decision),
    ):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return {
        "output_root": output,
        "summary": summary,
        "decision": decision,
    }
