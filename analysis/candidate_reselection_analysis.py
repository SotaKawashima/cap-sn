"""Helpers for the amended Stage 7 candidate reselection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from analysis.candidate_validation_analysis import (
    CandidateValidationData,
    build_candidate_clusters,
    build_seed_summary,
)
from analysis.optimization_metrics import OBJECTIVE_NAME


REQUIRED_REFERENCES = ("none", "simple_max")
DESCRIPTIVE_REFERENCES = ("legacy_balance", "prior_high")
ALL_REFERENCES = REQUIRED_REFERENCES + DESCRIPTIVE_REFERENCES


def combine_validation_data(
    source: CandidateValidationData,
    supplemental: CandidateValidationData,
    *,
    expected_source_runs: int,
    expected_supplemental_runs: int,
    expected_iterations_per_run: int,
) -> CandidateValidationData:
    """Combine audited original validation data with simple_max runs."""

    if len(source.runs) != expected_source_runs:
        raise ValueError(
            f"source run count={len(source.runs)}, expected={expected_source_runs}"
        )
    if len(supplemental.runs) != expected_supplemental_runs:
        raise ValueError(
            "supplemental run count="
            f"{len(supplemental.runs)}, expected={expected_supplemental_runs}"
        )
    for name, data in (("source", source), ("supplemental", supplemental)):
        if not data.audit["valid"].all():
            raise ValueError(f"{name} data contains an invalid run")
        if not data.runs["n_iterations"].eq(expected_iterations_per_run).all():
            raise ValueError(f"{name} data has an unexpected iteration count")

    iteration_frames: list[pd.DataFrame] = []
    run_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for label, data in (("source_stage7", source), ("simple_max", supplemental)):
        iterations = data.iterations.copy()
        iterations.insert(0, "dataset_source", label)
        iteration_frames.append(iterations)
        runs = data.runs.copy()
        runs.insert(0, "dataset_source", label)
        run_frames.append(runs)
        audit = data.audit.copy()
        audit.insert(0, "dataset_source", label)
        audit_frames.append(audit)

    combined_iterations = pd.concat(iteration_frames, ignore_index=True)
    combined_runs = pd.concat(run_frames, ignore_index=True)
    combined_audit = pd.concat(audit_frames, ignore_index=True)
    iteration_key = ["network", "condition_id", "simulator_seed", "num_iter"]
    run_key = ["network", "condition_id", "simulator_seed"]
    if combined_iterations.duplicated(iteration_key).any():
        raise ValueError("combined iteration data contains duplicate observations")
    if combined_runs.duplicated(run_key).any():
        raise ValueError("combined run inventory contains duplicate runs")
    expected_iteration_count = (
        (expected_source_runs + expected_supplemental_runs)
        * expected_iterations_per_run
    )
    if len(combined_iterations) != expected_iteration_count:
        raise ValueError(
            "combined iteration count="
            f"{len(combined_iterations)}, expected={expected_iteration_count}"
        )

    source_conditions = set(
        source.runs.loc[
            source.runs["condition_role"].eq("stage6_candidate"),
            ["network", "condition_id"],
        ].itertuples(index=False, name=None)
    )
    if len(source_conditions) != 54:
        raise ValueError("source data does not contain 54 unique candidates")
    if set(supplemental.runs["condition_id"]) != {"simple_max"}:
        raise ValueError("supplemental data must contain only simple_max")

    return CandidateValidationData(
        iterations=combined_iterations.sort_values(iteration_key).reset_index(
            drop=True
        ),
        runs=combined_runs.sort_values(run_key).reset_index(drop=True),
        audit=combined_audit.sort_values(
            ["dataset_source", "run_key"]
        ).reset_index(drop=True),
    )


def build_candidate_block_performance(
    iterations: pd.DataFrame,
    *,
    candidate_role: str = "stage6_candidate",
) -> pd.DataFrame:
    """Build candidate ranks and all four reference contrasts by seed block."""

    seed_summary = build_seed_summary(iterations)
    references = (
        seed_summary[seed_summary["condition_id"].isin(ALL_REFERENCES)]
        [["network", "simulator_seed", "condition_id", "mean_jcum"]]
        .pivot(
            index=["network", "simulator_seed"],
            columns="condition_id",
            values="mean_jcum",
        )
        .reset_index()
    )
    missing_columns = set(ALL_REFERENCES) - set(references.columns)
    if missing_columns:
        raise ValueError(f"reference conditions are missing: {missing_columns}")

    candidates = seed_summary[
        seed_summary["condition_role"].eq(candidate_role)
    ].copy()
    candidates["validation_block_rank"] = candidates.groupby(
        ["network", "simulator_seed"], sort=False
    )["mean_jcum"].rank(method="average", ascending=True)
    candidates = candidates.merge(
        references,
        on=["network", "simulator_seed"],
        how="left",
        validate="many_to_one",
    )
    if candidates[list(ALL_REFERENCES)].isna().any().any():
        raise ValueError("one or more reference block means are missing")
    for reference in ALL_REFERENCES:
        absolute = f"absolute_suppression_vs_{reference}"
        candidates[absolute] = candidates[reference] - candidates["mean_jcum"]
        candidates[f"relative_suppression_vs_{reference}"] = (
            candidates[absolute] / candidates[reference]
        )
    candidates["validation_minus_exploration"] = (
        candidates["mean_jcum"] - candidates["source_final_best"]
    )
    return candidates.sort_values(
        ["network", "simulator_seed", "validation_block_rank", "condition_id"]
    ).reset_index(drop=True)


def _effect_columns(effect: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        "network",
        "condition_id",
        "positive_seed_blocks",
        "zero_seed_blocks",
        "negative_seed_blocks",
        "minimum_block_suppression",
        "maximum_block_suppression",
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
    effects_by_reference: dict[str, pd.DataFrame],
    *,
    maximum_cluster_distance: float,
    target_candidates_per_network: int,
    minimum_positive_seed_blocks: int,
    reserved_revised_final_test_seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the amended none-and-simple_max eligibility rule."""

    if set(effects_by_reference) != set(ALL_REFERENCES):
        raise ValueError("effects_by_reference must contain all references")
    if target_candidates_per_network < 1:
        raise ValueError("target_candidates_per_network must be positive")

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
                "mean_peak_new_selfish_ratio", "mean"
            ),
        )
        .reset_index()
    )
    ranking["validation_minus_exploration"] = (
        ranking["validation_mean_jcum"] - ranking["source_final_best"]
    )
    for reference in ALL_REFERENCES:
        ranking = ranking.merge(
            _effect_columns(effects_by_reference[reference], reference),
            on=["network", "condition_id"],
            how="left",
            validate="one_to_one",
        )
        ranking[f"eligible_vs_{reference}"] = (
            (ranking[f"{reference}_absolute_suppression"] > 0)
            & (
                ranking[f"{reference}_positive_seed_blocks"]
                >= minimum_positive_seed_blocks
            )
        )
    ranking["qualified"] = (
        ranking["eligible_vs_none"] & ranking["eligible_vs_simple_max"]
    )

    clusters = build_candidate_clusters(
        ranking[["network", "condition_id", "certainty", "effectiveness"]],
        maximum_distance=maximum_cluster_distance,
    )
    ranking = ranking.merge(
        clusters,
        on=["network", "condition_id"],
        how="left",
        validate="one_to_one",
    )
    ranking["selected_for_revised_final_test"] = False
    ranking["selection_mode"] = "not_selected"
    ranking["selection_role"] = "not_selected"
    ranking["selection_order"] = pd.Series(
        pd.NA, index=ranking.index, dtype="Int64"
    )

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
        ordered_qualified = ordered_all[ordered_all["qualified"]]
        qualified_representatives = ordered_qualified.drop_duplicates(
            "region_id", keep="first"
        )
        chosen: list[tuple[int, str]] = []
        used_regions: set[str] = set()
        for index, row in qualified_representatives.iterrows():
            if len(chosen) >= target_candidates_per_network:
                break
            region_id = str(row["region_id"])
            chosen.append((int(index), "qualified_candidate"))
            used_regions.add(region_id)

        if len(chosen) < target_candidates_per_network:
            for index, row in ordered_all.iterrows():
                if len(chosen) >= target_candidates_per_network:
                    break
                region_id = str(row["region_id"])
                if region_id in used_regions:
                    continue
                chosen.append((int(index), "exploratory_fallback"))
                used_regions.add(region_id)

        qualified_selected = sum(
            role == "qualified_candidate" for _, role in chosen
        )
        if qualified_selected == 0:
            mode = "no_qualified_candidate_exploratory_fallback"
        elif qualified_selected < len(chosen):
            mode = "qualified_candidates_with_exploratory_fill"
        else:
            mode = "qualified_candidates_selected"

        selected_rows: list[dict[str, Any]] = []
        for order, (index, role) in enumerate(chosen, start=1):
            ranking.loc[index, "selected_for_revised_final_test"] = True
            ranking.loc[index, "selection_mode"] = mode
            ranking.loc[index, "selection_role"] = role
            ranking.loc[index, "selection_order"] = order
            row = ranking.loc[index]
            selected_rows.append(
                {
                    "condition_id": str(row["condition_id"]),
                    "region_id": str(row["region_id"]),
                    "certainty": float(row["certainty"]),
                    "effectiveness": float(row["effectiveness"]),
                    "qualified": bool(row["qualified"]),
                    "selection_role": role,
                    "selection_order": order,
                }
            )
        network_decisions.append(
            {
                "network": str(network),
                "candidate_count": int(len(group)),
                "qualified_candidate_count": int(group["qualified"].sum()),
                "qualified_region_count": int(
                    group.loc[group["qualified"], "region_id"].nunique()
                ),
                "selection_mode": mode,
                "selected_count": len(chosen),
                "qualified_selected_count": qualified_selected,
                "selected_candidates": selected_rows,
            }
        )

    ranking["overall_order"] = ranking["overall_order"].astype("Int64")
    ranking = ranking.sort_values(
        [
            "network",
            "selected_for_revised_final_test",
            "selection_order",
            "overall_order",
        ],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    selected = ranking[
        ranking["selected_for_revised_final_test"]
    ].copy()

    cluster_rows: list[dict[str, Any]] = []
    for (network, region_id), group in ranking.groupby(
        ["network", "region_id"], sort=True
    ):
        selected_ids = group.loc[
            group["selected_for_revised_final_test"], "condition_id"
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
        "status": "amended_candidate_reselection_complete",
        "amended_after_original_results": True,
        "source_candidates_changed": False,
        "source_candidate_count": int(
            ranking[["network", "condition_id"]].drop_duplicates().shape[0]
        ),
        "eligibility_references": list(REQUIRED_REFERENCES),
        "descriptive_references": list(DESCRIPTIVE_REFERENCES),
        "revised_final_candidates_frozen": True,
        "revised_final_test_seeds_used": False,
        "reserved_revised_final_test_seeds": [
            int(seed) for seed in reserved_revised_final_test_seeds
        ],
        "network_decisions": network_decisions,
    }
    return ranking, cluster_summary, selected, decision
