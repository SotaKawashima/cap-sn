"""Audit, effect estimation, and selection helpers for Stage 7."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from analysis.optimization_metrics import (
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
    compute_selfish_metrics_from_arrow,
)
from analysis.pilot_analysis import hierarchical_mean_interval
from experiment_runtime import NETWORKS, sha256_file


STAGE = "stage7_candidate_validation"


@dataclass(frozen=True)
class CandidateValidationData:
    """Validated Stage 7 data at iteration, run, and audit levels."""

    iterations: pd.DataFrame
    runs: pd.DataFrame
    audit: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _float_matches(observed: Any, expected: Any, tolerance: float) -> bool:
    if observed is None or expected is None:
        return observed is None and expected is None
    try:
        return math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _manifest_errors(
    manifest: dict[str, Any], spec: Any, *, tolerance: float
) -> list[str]:
    intervention = manifest.get("intervention", {})
    objective = manifest.get("objective", {})
    observed = {
        "status": manifest.get("status"),
        "stage": manifest.get("stage"),
        "run_type": manifest.get("run_type"),
        "network": manifest.get("network", {}).get("id"),
        "condition_id": intervention.get("condition_id"),
        "intervention_enabled": intervention.get("enabled"),
        "simulator_seed": manifest.get("runtime", {}).get("simulator_seed"),
        "iterations": manifest.get("runtime", {}).get("iteration_count"),
        "objective_name": objective.get("name"),
        "objective_version": objective.get("definition_version"),
    }
    expected = {
        "status": "completed",
        "stage": STAGE,
        "run_type": "fixed_condition",
        "network": spec.network,
        "condition_id": spec.condition_id,
        "intervention_enabled": bool(spec.intervention_enabled),
        "simulator_seed": int(spec.simulator_seed),
        "iterations": int(spec.iterations),
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_DEFINITION_VERSION,
    }
    errors = [
        f"{field}={observed[field]!r}, expected={value!r}"
        for field, value in expected.items()
        if observed[field] != value
    ]

    applied = intervention.get("applied_parameters")
    if spec.intervention_enabled:
        if not isinstance(applied, dict):
            errors.append("applied_parameters is missing")
        else:
            for field in ("certainty", "effectiveness"):
                expected_value = getattr(spec, field)
                if not _float_matches(applied.get(field), expected_value, tolerance):
                    errors.append(
                        f"applied_{field}={applied.get(field)!r}, "
                        f"expected={expected_value!r}"
                    )
        if spec.opinion_csv is None:
            if intervention.get("opinion_mode") != "generated_from_design_variables":
                errors.append("generated intervention has an unexpected opinion mode")
        else:
            if intervention.get("opinion_mode") != "existing_csv":
                errors.append("existing-CSV intervention has an unexpected opinion mode")
            expected_hash = spec.opinion_sha256 or sha256_file(spec.opinion_csv)
            for field in ("opinion_source", "opinion_csv"):
                entry = intervention.get(field) or {}
                if entry.get("sha256") != expected_hash:
                    errors.append(
                        f"{field}_sha256={entry.get('sha256')!r}, "
                        f"expected={expected_hash!r}"
                    )
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
        return [f"metrics row count={len(saved)}, expected={expected_iterations}"]
    if "num_iter" not in saved.columns:
        return ["saved metrics have no num_iter column"]
    observed_ids = set(pd.to_numeric(saved["num_iter"], errors="coerce").dropna())
    if observed_ids != set(range(expected_iterations)):
        return ["metrics iteration IDs do not match the expected range"]

    left = saved.sort_values("num_iter").reset_index(drop=True)
    right = recalculated.sort_values("num_iter").reset_index(drop=True)
    missing = [column for column in right.columns if column not in left.columns]
    if missing:
        return [f"saved metrics are missing columns: {missing}"]
    errors: list[str] = []
    for column in right.columns:
        if column == "num_iter":
            continue
        observed = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        expected = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if not np.allclose(
            observed,
            expected,
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        ):
            errors.append(f"saved metric does not reproduce: {column}")
    return errors


def load_candidate_validation(
    experiment_root: str | Path,
    *,
    expected_specs: Sequence[Any],
    numeric_tolerance: float = 1e-12,
) -> CandidateValidationData:
    """Load all expected runs and reproduce metrics from retained pop Arrow."""

    root = Path(experiment_root).resolve()
    if numeric_tolerance < 0 or not math.isfinite(numeric_tolerance):
        raise ValueError("numeric_tolerance must be finite and non-negative")
    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in expected_specs:
        run_dir = root / spec.relative_run_dir
        paths = {
            "manifest": run_dir / "manifest.json",
            "metrics": run_dir / "metrics.csv",
            "summary": run_dir / "metrics_summary.json",
            "pop": run_dir / "pop.arrow",
        }
        audit: dict[str, Any] = {
            "run_key": spec.key,
            "network": spec.network,
            "condition_id": spec.condition_id,
            "simulator_seed": int(spec.simulator_seed),
            "files_present": False,
            "manifest_valid": False,
            "metrics_reproduced": False,
            "raw_policy_valid": False,
            "valid": False,
            "error": None,
        }
        try:
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing files: {missing}")
            audit["files_present"] = True
            manifest = _read_json(paths["manifest"])
            errors = _manifest_errors(manifest, spec, tolerance=numeric_tolerance)
            if errors:
                raise ValueError("; ".join(errors))
            audit["manifest_valid"] = True

            unexpected_raw = [
                name
                for name in ("info.arrow", "agent.arrow")
                if (run_dir / name).exists()
            ]
            outputs = manifest.get("outputs", {})
            if unexpected_raw or "info_arrow" in outputs or "agent_arrow" in outputs:
                raise ValueError(
                    f"raw-level pop policy violated: unexpected={unexpected_raw}"
                )
            audit["raw_policy_valid"] = True

            num_agents = int(manifest["network"]["num_agents"])
            recalculated = compute_selfish_metrics_from_arrow(
                paths["pop"],
                num_agents=num_agents,
                expected_iterations=int(spec.iterations),
            )
            saved = pd.read_csv(paths["metrics"])
            metric_errors = _metric_errors(
                saved,
                recalculated.per_iteration,
                expected_iterations=int(spec.iterations),
                tolerance=numeric_tolerance,
            )
            if metric_errors:
                raise ValueError("; ".join(metric_errors))
            summary = _read_json(paths["summary"])
            if not _float_matches(
                summary.get(OBJECTIVE_NAME),
                recalculated.objective_value,
                numeric_tolerance,
            ) or not _float_matches(
                manifest.get("objective", {}).get("value"),
                recalculated.objective_value,
                numeric_tolerance,
            ):
                raise ValueError("saved objective does not match raw pop data")
            audit["metrics_reproduced"] = True

            iteration = recalculated.per_iteration.copy()
            iteration.insert(0, "run_key", spec.key)
            iteration.insert(1, "network", spec.network)
            iteration.insert(2, "condition_id", spec.condition_id)
            iteration.insert(3, "condition_role", spec.condition_role)
            iteration.insert(4, "intervention_enabled", spec.intervention_enabled)
            iteration.insert(5, "certainty", spec.certainty)
            iteration.insert(6, "effectiveness", spec.effectiveness)
            iteration.insert(7, "candidate_source", spec.candidate_source)
            iteration.insert(8, "source_method", spec.source_method)
            iteration.insert(
                9,
                "source_optimizer_replicate",
                spec.source_optimizer_replicate,
            )
            iteration.insert(10, "source_final_best", spec.source_final_best)
            iteration.insert(11, "simulator_seed", spec.simulator_seed)
            iteration.insert(12, "num_agents", num_agents)
            iteration_frames.append(iteration)

            timing = manifest.get("timing_sec", {})
            run_rows.append(
                {
                    "run_key": spec.key,
                    "relative_run_dir": spec.relative_run_dir,
                    "network": spec.network,
                    "condition_id": spec.condition_id,
                    "condition_role": spec.condition_role,
                    "intervention_enabled": bool(spec.intervention_enabled),
                    "certainty": spec.certainty,
                    "effectiveness": spec.effectiveness,
                    "candidate_source": spec.candidate_source,
                    "source_method": spec.source_method,
                    "source_optimizer_replicate": spec.source_optimizer_replicate,
                    "source_optimizer_seed": spec.source_optimizer_seed,
                    "source_simulator_seed": spec.source_simulator_seed,
                    "source_final_best": spec.source_final_best,
                    "simulator_seed": int(spec.simulator_seed),
                    "num_agents": num_agents,
                    "n_iterations": int(spec.iterations),
                    OBJECTIVE_NAME: recalculated.objective_value,
                    "peak_new_selfish_ratio": recalculated.summary[
                        "peak_new_selfish_ratio"
                    ],
                    "simulation_sec": timing.get("simulation"),
                    "metric_calculation_sec": timing.get("metric_calculation"),
                    "total_sec": timing.get("total"),
                    "pop_bytes": int(paths["pop"].stat().st_size),
                }
            )
            audit["valid"] = True
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"{spec.key}: {audit['error']}")
        audit_rows.append(audit)

    if failures:
        preview = "\n".join(failures[:10])
        raise ValueError(
            f"Stage 7 data audit failed for {len(failures)} runs:\n{preview}"
        )
    iterations = pd.concat(iteration_frames, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    audit = pd.DataFrame(audit_rows)
    return CandidateValidationData(
        iterations=iterations.sort_values(
            ["network", "condition_id", "simulator_seed", "num_iter"]
        ).reset_index(drop=True),
        runs=runs.sort_values("run_key").reset_index(drop=True),
        audit=audit.sort_values("run_key").reset_index(drop=True),
    )


def build_seed_summary(iterations: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "network",
        "condition_id",
        "condition_role",
        "intervention_enabled",
        "certainty",
        "effectiveness",
        "candidate_source",
        "source_method",
        "source_optimizer_replicate",
        "source_final_best",
        "simulator_seed",
        "num_agents",
    ]
    grouped = iterations.groupby(keys, dropna=False, sort=True)
    return grouped.agg(
        n=(OBJECTIVE_NAME, "size"),
        mean_jcum=(OBJECTIVE_NAME, "mean"),
        std_jcum=(OBJECTIVE_NAME, "std"),
        sem_jcum=(OBJECTIVE_NAME, "sem"),
        mean_peak_new_selfish_ratio=("peak_new_selfish_ratio", "mean"),
    ).reset_index()


def build_condition_summary(
    iterations: pd.DataFrame, *, repetitions: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "network",
        "condition_id",
        "condition_role",
        "intervention_enabled",
        "certainty",
        "effectiveness",
        "candidate_source",
        "source_method",
        "source_optimizer_replicate",
        "source_final_best",
        "num_agents",
    ]
    for offset, (key, group) in enumerate(
        iterations.groupby(keys, dropna=False, sort=True)
    ):
        estimate = hierarchical_mean_interval(
            group,
            block_column="simulator_seed",
            value_column=OBJECTIVE_NAME,
            repetitions=repetitions,
            seed=seed + offset,
        )
        block_means = group.groupby("simulator_seed")[OBJECTIVE_NAME].mean()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_iterations": int(len(group)),
                "between_seed_sd": float(block_means.std()),
                "mean_peak_new_selfish_ratio": float(
                    group["peak_new_selfish_ratio"].mean()
                ),
                **estimate,
            }
        )
    return pd.DataFrame(rows)


def _paired_hierarchical_effect(
    frame: pd.DataFrame,
    *,
    reference_column: str,
    candidate_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    data = frame[["simulator_seed", reference_column, candidate_column]].dropna()
    blocks = [
        group[[reference_column, candidate_column]].to_numpy(dtype=float)
        for _, group in data.groupby("simulator_seed", sort=True)
    ]
    if len(blocks) < 2 or any(len(block) < 2 for block in blocks):
        raise ValueError("paired bootstrap requires at least two nontrivial blocks")

    reference_estimate = float(np.mean([block[:, 0].mean() for block in blocks]))
    candidate_estimate = float(np.mean([block[:, 1].mean() for block in blocks]))
    delta_estimate = reference_estimate - candidate_estimate
    eta_estimate = delta_estimate / reference_estimate

    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    reference_samples = np.zeros(repetitions, dtype=float)
    candidate_samples = np.zeros(repetitions, dtype=float)
    for _ in range(n_blocks):
        selected_blocks = rng.integers(0, n_blocks, size=repetitions)
        reference_contribution = np.empty(repetitions, dtype=float)
        candidate_contribution = np.empty(repetitions, dtype=float)
        for block_index, block in enumerate(blocks):
            rows = np.flatnonzero(selected_blocks == block_index)
            if len(rows) == 0:
                continue
            indices = rng.integers(0, len(block), size=(len(rows), len(block)))
            reference_contribution[rows] = block[indices, 0].mean(axis=1)
            candidate_contribution[rows] = block[indices, 1].mean(axis=1)
        reference_samples += reference_contribution / n_blocks
        candidate_samples += candidate_contribution / n_blocks
    delta_samples = reference_samples - candidate_samples
    eta_samples = delta_samples / reference_samples
    delta_low, delta_high = np.quantile(delta_samples, [0.025, 0.975])
    eta_low, eta_high = np.quantile(eta_samples, [0.025, 0.975])
    if delta_low > 0:
        interpretation = "reduction"
    elif delta_estimate > 0:
        interpretation = "reduction_tendency_with_uncertainty"
    elif delta_high < 0:
        interpretation = "increase"
    else:
        interpretation = "no_clear_reduction"
    return {
        "n_blocks": n_blocks,
        "n_per_block": min(len(block) for block in blocks),
        "reference_estimate": reference_estimate,
        "candidate_estimate": candidate_estimate,
        "absolute_suppression": delta_estimate,
        "absolute_ci_low": float(delta_low),
        "absolute_ci_high": float(delta_high),
        "relative_suppression": eta_estimate,
        "relative_ci_low": float(eta_low),
        "relative_ci_high": float(eta_high),
        "interpretation": interpretation,
    }


def _candidate_metadata(iterations: pd.DataFrame) -> pd.DataFrame:
    candidates = iterations[iterations["condition_role"] == "stage6_candidate"]
    columns = [
        "network",
        "condition_id",
        "certainty",
        "effectiveness",
        "candidate_source",
        "source_method",
        "source_optimizer_replicate",
        "source_final_best",
        "num_agents",
    ]
    metadata = candidates[columns].drop_duplicates()
    if metadata.duplicated(["network", "condition_id"]).any():
        raise ValueError("candidate metadata is not unique")
    return metadata.reset_index(drop=True)


def build_candidate_effects(
    iterations: pd.DataFrame,
    *,
    reference_id: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    key = ["network", "simulator_seed", "num_iter"]
    if iterations.duplicated(key + ["condition_id"]).any():
        raise ValueError("iteration table contains duplicate paired observations")
    wide = iterations.pivot(
        index=key, columns="condition_id", values=OBJECTIVE_NAME
    ).reset_index()
    if reference_id not in wide:
        raise ValueError(f"reference condition is missing: {reference_id}")
    metadata = _candidate_metadata(iterations)
    rows: list[dict[str, Any]] = []
    for offset, candidate in metadata.sort_values(
        ["network", "condition_id"]
    ).reset_index(drop=True).iterrows():
        network = str(candidate["network"])
        candidate_id = str(candidate["condition_id"])
        group = wide[wide["network"] == network]
        if candidate_id not in group:
            raise ValueError(f"candidate condition is missing: {candidate_id}")
        effect = _paired_hierarchical_effect(
            group,
            reference_column=reference_id,
            candidate_column=candidate_id,
            repetitions=repetitions,
            seed=seed + offset,
        )
        block_means = group.groupby("simulator_seed")[[reference_id, candidate_id]].mean()
        block_deltas = block_means[reference_id] - block_means[candidate_id]
        num_agents = int(candidate["num_agents"])
        rows.append(
            {
                **candidate.to_dict(),
                "reference_id": reference_id,
                "positive_seed_blocks": int((block_deltas > 0).sum()),
                "zero_seed_blocks": int((block_deltas == 0).sum()),
                "negative_seed_blocks": int((block_deltas < 0).sum()),
                "minimum_block_suppression": float(block_deltas.min()),
                "maximum_block_suppression": float(block_deltas.max()),
                **effect,
                "equivalent_agents": effect["absolute_suppression"] * num_agents,
                "equivalent_agents_ci_low": effect["absolute_ci_low"] * num_agents,
                "equivalent_agents_ci_high": effect["absolute_ci_high"] * num_agents,
            }
        )
    return pd.DataFrame(rows)


def build_candidate_block_performance(iterations: pd.DataFrame) -> pd.DataFrame:
    seed_summary = build_seed_summary(iterations)
    references = (
        seed_summary[
            seed_summary["condition_id"].isin(
                ["none", "legacy_balance", "prior_high"]
            )
        ][["network", "simulator_seed", "condition_id", "mean_jcum"]]
        .pivot(
            index=["network", "simulator_seed"],
            columns="condition_id",
            values="mean_jcum",
        )
        .reset_index()
    )
    candidates = seed_summary[seed_summary["condition_role"] == "stage6_candidate"].copy()
    candidates["validation_block_rank"] = candidates.groupby(
        ["network", "simulator_seed"], sort=False
    )["mean_jcum"].rank(method="average", ascending=True)
    candidates = candidates.merge(
        references,
        on=["network", "simulator_seed"],
        how="left",
        validate="many_to_one",
    )
    if candidates[["none", "legacy_balance", "prior_high"]].isna().any().any():
        raise ValueError("one or more Stage 7 reference block means are missing")
    for reference in ("none", "legacy_balance", "prior_high"):
        candidates[f"absolute_suppression_vs_{reference}"] = (
            candidates[reference] - candidates["mean_jcum"]
        )
        candidates[f"relative_suppression_vs_{reference}"] = (
            candidates[f"absolute_suppression_vs_{reference}"]
            / candidates[reference]
        )
    candidates["validation_minus_exploration"] = (
        candidates["mean_jcum"] - candidates["source_final_best"]
    )
    return candidates.sort_values(
        ["network", "simulator_seed", "validation_block_rank", "condition_id"]
    ).reset_index(drop=True)


def build_exploration_validation_comparison(
    block_performance: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "network",
        "condition_id",
        "certainty",
        "effectiveness",
        "candidate_source",
        "source_method",
        "source_optimizer_replicate",
        "source_final_best",
    ]
    result = (
        block_performance.groupby(keys, dropna=False, sort=True)
        .agg(
            validation_mean_jcum=("mean_jcum", "mean"),
            validation_between_seed_sd=("mean_jcum", "std"),
            validation_min_jcum=("mean_jcum", "min"),
            validation_max_jcum=("mean_jcum", "max"),
        )
        .reset_index()
    )
    result["validation_minus_exploration"] = (
        result["validation_mean_jcum"] - result["source_final_best"]
    )
    result["validation_to_exploration_ratio"] = (
        result["validation_mean_jcum"] / result["source_final_best"]
    )
    return result


def _complete_linkage_distance(
    first: Sequence[int], second: Sequence[int], points: np.ndarray
) -> float:
    return max(
        float(np.linalg.norm(points[left] - points[right]))
        for left in first
        for right in second
    )


def build_candidate_clusters(
    candidate_metadata: pd.DataFrame, *, maximum_distance: float
) -> pd.DataFrame:
    if maximum_distance <= 0 or not math.isfinite(maximum_distance):
        raise ValueError("maximum_distance must be finite and positive")
    rows: list[dict[str, Any]] = []
    for network, group in candidate_metadata.groupby("network", sort=True):
        frame = group.sort_values("condition_id").reset_index(drop=True)
        points = frame[["certainty", "effectiveness"]].to_numpy(float)
        clusters: list[tuple[int, ...]] = [(index,) for index in range(len(frame))]
        while True:
            choices: list[
                tuple[float, tuple[str, ...], int, int]
            ] = []
            for left in range(len(clusters)):
                for right in range(left + 1, len(clusters)):
                    distance = _complete_linkage_distance(
                        clusters[left], clusters[right], points
                    )
                    if distance <= maximum_distance + 1e-15:
                        member_ids = tuple(
                            sorted(
                                frame.loc[
                                    list(clusters[left] + clusters[right]),
                                    "condition_id",
                                ].astype(str)
                            )
                        )
                        choices.append(
                            (round(distance, 12), member_ids, left, right)
                        )
            if not choices:
                break
            _, _, left, right = min(choices)
            merged = tuple(sorted(clusters[left] + clusters[right]))
            clusters = [
                cluster
                for index, cluster in enumerate(clusters)
                if index not in {left, right}
            ]
            clusters.append(merged)
            clusters.sort(
                key=lambda cluster: tuple(
                    sorted(frame.loc[list(cluster), "condition_id"].astype(str))
                )
            )
        clusters.sort(
            key=lambda cluster: (
                float(frame.loc[list(cluster), "certainty"].min()),
                float(frame.loc[list(cluster), "effectiveness"].min()),
                tuple(sorted(frame.loc[list(cluster), "condition_id"].astype(str))),
            )
        )
        for number, cluster in enumerate(clusters, start=1):
            members = frame.loc[list(cluster)]
            region_id = f"region_{number:02d}"
            for _, member in members.iterrows():
                rows.append(
                    {
                        "network": network,
                        "condition_id": member["condition_id"],
                        "region_id": region_id,
                        "region_member_count": int(len(members)),
                        "region_members": ";".join(
                            sorted(members["condition_id"].astype(str))
                        ),
                        "region_certainty_min": float(members["certainty"].min()),
                        "region_certainty_max": float(members["certainty"].max()),
                        "region_effectiveness_min": float(
                            members["effectiveness"].min()
                        ),
                        "region_effectiveness_max": float(
                            members["effectiveness"].max()
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["network", "region_id", "condition_id"]
    ).reset_index(drop=True)


def _effect_columns(effect: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        "network",
        "condition_id",
        "positive_seed_blocks",
        "negative_seed_blocks",
        "absolute_suppression",
        "absolute_ci_low",
        "absolute_ci_high",
        "relative_suppression",
        "relative_ci_low",
        "relative_ci_high",
        "interpretation",
    ]
    result = effect[columns].copy()
    return result.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in columns
            if column not in {"network", "condition_id"}
        }
    )


def build_candidate_selection(
    block_performance: pd.DataFrame,
    effects_vs_none: pd.DataFrame,
    effects_vs_legacy: pd.DataFrame,
    effects_vs_prior: pd.DataFrame,
    *,
    maximum_cluster_distance: float,
    maximum_regions_per_network: int,
    minimum_positive_seed_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    group_keys = [
        "network",
        "condition_id",
        "certainty",
        "effectiveness",
        "candidate_source",
        "source_method",
        "source_optimizer_replicate",
        "source_final_best",
        "num_agents",
    ]
    ranking = (
        block_performance.groupby(group_keys, dropna=False, sort=True)
        .agg(
            validation_mean_jcum=("mean_jcum", "mean"),
            validation_between_seed_sd=("mean_jcum", "std"),
            median_validation_block_rank=("validation_block_rank", "median"),
            worst_validation_block_rank=("validation_block_rank", "max"),
            best_validation_block_rank=("validation_block_rank", "min"),
            mean_peak_new_selfish_ratio=(
                "mean_peak_new_selfish_ratio",
                "mean",
            ),
        )
        .reset_index()
    )
    ranking["validation_minus_exploration"] = (
        ranking["validation_mean_jcum"] - ranking["source_final_best"]
    )
    for effect, prefix in (
        (effects_vs_none, "none"),
        (effects_vs_legacy, "legacy_balance"),
        (effects_vs_prior, "prior_high"),
    ):
        ranking = ranking.merge(
            _effect_columns(effect, prefix),
            on=["network", "condition_id"],
            how="left",
            validate="one_to_one",
        )
    ranking["eligible_vs_none"] = (
        (ranking["none_absolute_suppression"] > 0)
        & (
            ranking["none_positive_seed_blocks"]
            >= minimum_positive_seed_blocks
        )
    )
    ranking["eligible_vs_legacy_balance"] = (
        (ranking["legacy_balance_absolute_suppression"] > 0)
        & (
            ranking["legacy_balance_positive_seed_blocks"]
            >= minimum_positive_seed_blocks
        )
    )
    ranking["qualified"] = (
        ranking["eligible_vs_none"]
        & ranking["eligible_vs_legacy_balance"]
    )

    clusters = build_candidate_clusters(
        ranking[
            ["network", "condition_id", "certainty", "effectiveness"]
        ],
        maximum_distance=maximum_cluster_distance,
    )
    ranking = ranking.merge(
        clusters,
        on=["network", "condition_id"],
        how="left",
        validate="one_to_one",
    )
    ranking["selected_for_final_test"] = False
    ranking["selection_mode"] = "not_selected"
    ranking["selection_order"] = pd.Series(pd.NA, index=ranking.index, dtype="Int64")

    order_columns = [
        "median_validation_block_rank",
        "worst_validation_block_rank",
        "validation_mean_jcum",
        "condition_id",
    ]
    network_decisions: list[dict[str, Any]] = []
    for network, group in ranking.groupby("network", sort=True):
        ordered_all = group.sort_values(order_columns, kind="stable")
        ranking.loc[ordered_all.index, "overall_order"] = np.arange(
            1, len(ordered_all) + 1
        )
        qualified = ordered_all[ordered_all["qualified"]]
        if qualified.empty:
            pool = ordered_all
            mode = "no_qualified_candidate_exploratory_fallback"
        else:
            pool = qualified
            mode = "qualified_candidates_selected"
        representatives = pool.drop_duplicates("region_id", keep="first").head(
            maximum_regions_per_network
        )
        for order, index in enumerate(representatives.index, start=1):
            ranking.loc[index, "selected_for_final_test"] = True
            ranking.loc[index, "selection_mode"] = mode
            ranking.loc[index, "selection_order"] = order
        network_decisions.append(
            {
                "network": str(network),
                "candidate_count": int(len(group)),
                "qualified_candidate_count": int(group["qualified"].sum()),
                "qualified_region_count": int(
                    group.loc[group["qualified"], "region_id"].nunique()
                ),
                "selection_mode": mode,
                "selected_count": int(len(representatives)),
                "selected_candidates": [
                    {
                        "condition_id": str(row["condition_id"]),
                        "region_id": str(row["region_id"]),
                        "certainty": float(row["certainty"]),
                        "effectiveness": float(row["effectiveness"]),
                        "qualified": bool(row["qualified"]),
                    }
                    for _, row in representatives.iterrows()
                ],
            }
        )

    ranking["overall_order"] = ranking["overall_order"].astype("Int64")
    ranking = ranking.sort_values(
        ["network", "selected_for_final_test", "selection_order", "overall_order"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    selected = ranking[ranking["selected_for_final_test"]].copy()

    cluster_rows: list[dict[str, Any]] = []
    for (network, region_id), group in ranking.groupby(
        ["network", "region_id"], sort=True
    ):
        selected_ids = group.loc[
            group["selected_for_final_test"], "condition_id"
        ].astype(str)
        qualified = group[group["qualified"]]
        cluster_rows.append(
            {
                "network": network,
                "region_id": region_id,
                "member_count": int(len(group)),
                "members": ";".join(sorted(group["condition_id"].astype(str))),
                "qualified_member_count": int(len(qualified)),
                "qualified_members": ";".join(
                    sorted(qualified["condition_id"].astype(str))
                ),
                "certainty_min": float(group["certainty"].min()),
                "certainty_max": float(group["certainty"].max()),
                "effectiveness_min": float(group["effectiveness"].min()),
                "effectiveness_max": float(group["effectiveness"].max()),
                "selected_representative": (
                    selected_ids.iloc[0] if len(selected_ids) else None
                ),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows)

    decision = {
        "status": "candidate_selection_complete",
        "stage7_complete": True,
        "candidate_selection_complete": True,
        "validation_estimates_are_final_performance": False,
        "effect_claim_allowed": False,
        "final_test_candidates_frozen": True,
        "final_test_seeds_used": False,
        "next_stage": "reserved_final_test",
        "network_decisions": network_decisions,
    }
    return ranking, cluster_summary, selected, decision
