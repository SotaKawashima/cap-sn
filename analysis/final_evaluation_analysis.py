"""Audit and final-effect reporting helpers for Stage 8."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.candidate_validation_analysis import (
    CandidateValidationData,
    build_candidate_effects,
    build_condition_summary,
    build_seed_summary,
    load_candidate_validation,
)
from analysis.optimization_metrics import LEGACY_METRIC_NAME, OBJECTIVE_NAME
from run_stage8_final_evaluation import (
    STAGE,
    read_candidate_rows,
)


FINAL_CANDIDATE_ROLE = "final_candidate"
REFERENCES = ("none", "legacy_balance", "prior_high")


def load_final_evaluation(
    experiment_root: str,
    *,
    expected_specs: list[Any],
    numeric_tolerance: float = 1e-12,
) -> CandidateValidationData:
    """Load Stage 8 and independently reproduce every retained pop result."""

    return load_candidate_validation(
        experiment_root,
        expected_specs=expected_specs,
        numeric_tolerance=numeric_tolerance,
        expected_stage=STAGE,
    )


def candidate_metadata(protocol: dict[str, Any]) -> pd.DataFrame:
    """Return the nine frozen candidates with their pre-test reporting roles."""

    frame = pd.DataFrame(read_candidate_rows(protocol))
    columns = [
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
    ]
    result = frame[columns].copy()
    numeric = [
        column
        for column in columns
        if column
        not in {
            "network",
            "condition_id",
            "reporting_role",
            "selection_mode",
            "qualified",
            "region_id",
            "source_method",
            "candidate_source",
        }
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["qualified"] = result["qualified"].astype(bool)
    if result.duplicated(["network", "condition_id"]).any():
        raise ValueError("frozen Stage 8 candidate metadata is not unique")
    return result.sort_values(
        ["network", "selection_order"], kind="stable"
    ).reset_index(drop=True)


def build_auxiliary_condition_summary(iterations: pd.DataFrame) -> pd.DataFrame:
    """Summarize explanatory timing and peak metrics without changing the objective."""

    keys = ["network", "condition_id"]
    grouped = iterations.groupby(keys, sort=True, dropna=False)
    result = grouped.agg(
        n_iterations_aux=("num_iter", "size"),
        mean_cumulative_selfish_count=("cumulative_selfish_count", "mean"),
        mean_peak_new_selfish_ratio=("peak_new_selfish_ratio", "mean"),
        mean_new_selfish_rate_per_step=(LEGACY_METRIC_NAME, "mean"),
        mean_first_selfish_step=("first_selfish_step", "mean"),
        mean_last_selfish_step=("last_selfish_step", "mean"),
        mean_t50_selfish_step=("t50_selfish_step", "mean"),
        mean_t90_selfish_step=("t90_selfish_step", "mean"),
        mean_selfish_timing_centroid=("selfish_timing_centroid", "mean"),
        mean_selfish_span_steps=("selfish_span_steps", "mean"),
        mean_active_selfish_steps=("active_selfish_steps", "mean"),
    ).reset_index()
    zero_fraction = (
        iterations.assign(
            zero_selfish=iterations["cumulative_selfish_count"].eq(0).astype(float)
        )
        .groupby(keys, sort=True)["zero_selfish"]
        .mean()
        .rename("zero_selfish_iteration_fraction")
        .reset_index()
    )
    return result.merge(zero_fraction, on=keys, validate="one_to_one")


def build_final_condition_summary(
    iterations: pd.DataFrame, *, repetitions: int, seed: int
) -> pd.DataFrame:
    primary = build_condition_summary(
        iterations,
        repetitions=repetitions,
        seed=seed,
    )
    auxiliary = build_auxiliary_condition_summary(iterations)
    duplicate = [
        column
        for column in auxiliary.columns
        if column in primary.columns and column not in {"network", "condition_id"}
    ]
    auxiliary = auxiliary.drop(columns=duplicate)
    return primary.merge(
        auxiliary,
        on=["network", "condition_id"],
        how="left",
        validate="one_to_one",
    )


def build_final_candidate_effects(
    iterations: pd.DataFrame,
    *,
    reference_id: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    return build_candidate_effects(
        iterations,
        reference_id=reference_id,
        repetitions=repetitions,
        seed=seed,
        candidate_role=FINAL_CANDIDATE_ROLE,
    )


def build_candidate_seed_performance(
    iterations: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Report paired candidate effects separately for each final seed block."""

    seed_summary = build_seed_summary(iterations)
    references = (
        seed_summary[seed_summary["condition_id"].isin(REFERENCES)][
            ["network", "simulator_seed", "condition_id", "mean_jcum"]
        ]
        .pivot(
            index=["network", "simulator_seed"],
            columns="condition_id",
            values="mean_jcum",
        )
        .reset_index()
    )
    if any(reference not in references for reference in REFERENCES):
        raise ValueError("one or more Stage 8 reference conditions are missing")
    candidates = seed_summary[
        seed_summary["condition_role"] == FINAL_CANDIDATE_ROLE
    ].copy()
    candidates = candidates.merge(
        references,
        on=["network", "simulator_seed"],
        how="left",
        validate="many_to_one",
    )
    if candidates[list(REFERENCES)].isna().any().any():
        raise ValueError("one or more Stage 8 reference seed means are missing")
    for reference in REFERENCES:
        absolute = f"absolute_suppression_vs_{reference}"
        candidates[absolute] = candidates[reference] - candidates["mean_jcum"]
        candidates[f"relative_suppression_vs_{reference}"] = (
            candidates[absolute] / candidates[reference]
        )
    metadata_columns = [
        "network",
        "condition_id",
        "selection_order",
        "reporting_role",
        "selection_mode",
        "qualified",
        "region_id",
        "stage7_overall_order",
    ]
    candidates = candidates.merge(
        metadata[metadata_columns],
        on=["network", "condition_id"],
        how="left",
        validate="many_to_one",
    )
    if candidates["selection_order"].isna().any():
        raise ValueError("Stage 8 seed table contains an unfrozen candidate")
    return candidates.sort_values(
        ["network", "selection_order", "simulator_seed"], kind="stable"
    ).reset_index(drop=True)


def _effect_projection(effect: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = ["network", "condition_id"]
    columns = [
        "n_blocks",
        "n_per_block",
        "positive_seed_blocks",
        "zero_seed_blocks",
        "negative_seed_blocks",
        "minimum_block_suppression",
        "maximum_block_suppression",
        "reference_estimate",
        "candidate_estimate",
        "absolute_suppression",
        "absolute_ci_low",
        "absolute_ci_high",
        "relative_suppression",
        "relative_ci_low",
        "relative_ci_high",
        "equivalent_agents",
        "equivalent_agents_ci_low",
        "equivalent_agents_ci_high",
        "interpretation",
    ]
    result = effect[keys + columns].copy()
    return result.rename(
        columns={column: f"{prefix}_{column}" for column in columns}
    )


def build_final_candidate_results(
    condition_summary: pd.DataFrame,
    metadata: pd.DataFrame,
    effects_none: pd.DataFrame,
    effects_legacy: pd.DataFrame,
    effects_prior: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all prespecified final estimates without reranking candidates."""

    candidates = condition_summary[
        condition_summary["condition_role"] == FINAL_CANDIDATE_ROLE
    ].copy()
    redundant_metadata = [
        "certainty",
        "effectiveness",
        "candidate_source",
        "source_method",
        "source_optimizer_replicate",
        "source_final_best",
    ]
    frozen = metadata.drop(columns=redundant_metadata)
    result = candidates.merge(
        frozen,
        on=["network", "condition_id"],
        how="left",
        validate="one_to_one",
    )
    for effect, prefix in (
        (effects_none, "none"),
        (effects_legacy, "legacy_balance"),
        (effects_prior, "prior_high"),
    ):
        result = result.merge(
            _effect_projection(effect, prefix),
            on=["network", "condition_id"],
            how="left",
            validate="one_to_one",
        )
    if result["selection_order"].isna().any() or len(result) != len(metadata):
        raise ValueError("Stage 8 results do not match the nine frozen candidates")
    return result.sort_values(
        ["network", "selection_order"], kind="stable"
    ).reset_index(drop=True)


def build_validation_test_comparison(
    final_results: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "network",
        "condition_id",
        "selection_order",
        "reporting_role",
        "selection_mode",
        "qualified",
        "region_id",
        "stage7_validation_mean_jcum",
        "estimate",
        "stage7_none_relative_suppression",
        "none_relative_suppression",
        "stage7_legacy_balance_relative_suppression",
        "legacy_balance_relative_suppression",
        "stage7_prior_high_relative_suppression",
        "prior_high_relative_suppression",
    ]
    result = final_results[columns].copy()
    result = result.rename(columns={"estimate": "stage8_test_mean_jcum"})
    result["test_minus_validation_jcum"] = (
        result["stage8_test_mean_jcum"] - result["stage7_validation_mean_jcum"]
    )
    result["test_to_validation_jcum_ratio"] = (
        result["stage8_test_mean_jcum"] / result["stage7_validation_mean_jcum"]
    )
    for reference in REFERENCES:
        result[f"{reference}_eta_test_minus_validation"] = (
            result[f"{reference}_relative_suppression"]
            - result[f"stage7_{reference}_relative_suppression"]
        )
    result["used_for_reselection"] = False
    return result


def _network_conclusion_code(row: pd.Series) -> str:
    interpretation = str(row["none_interpretation"])
    prefix = "exploratory_fallback" if not bool(row["qualified"]) else "primary_candidate"
    suffix = {
        "reduction": "supported_reduction_vs_none",
        "reduction_tendency_with_uncertainty": "reduction_tendency_vs_none",
        "increase": "supported_increase_vs_none",
        "no_clear_reduction": "no_clear_reduction_vs_none",
    }.get(interpretation, "unclassified_vs_none")
    return f"{prefix}_{suffix}"


def build_network_conclusions(final_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize only the predesignated primary candidate as the main claim."""

    primary = final_results[final_results["selection_order"] == 1].copy()
    if len(primary) != 3 or primary["network"].nunique() != 3:
        raise ValueError("Stage 8 requires one frozen primary candidate per network")
    secondary = final_results[final_results["selection_order"] > 1]
    rows: list[dict[str, Any]] = []
    for _, candidate in primary.sort_values("network").iterrows():
        network = str(candidate["network"])
        other = secondary[secondary["network"] == network].sort_values(
            "selection_order"
        )
        rows.append(
            {
                "network": network,
                "evidence_scope": (
                    "qualified_primary_candidate"
                    if bool(candidate["qualified"])
                    else "exploratory_fallback_no_stage7_qualified_candidate"
                ),
                "primary_condition_id": candidate["condition_id"],
                "primary_certainty": candidate["certainty"],
                "primary_effectiveness": candidate["effectiveness"],
                "primary_region_id": candidate["region_id"],
                "primary_test_mean_jcum": candidate["estimate"],
                "primary_test_jcum_ci_low": candidate["ci_low"],
                "primary_test_jcum_ci_high": candidate["ci_high"],
                "none_relative_suppression": candidate[
                    "none_relative_suppression"
                ],
                "none_relative_ci_low": candidate["none_relative_ci_low"],
                "none_relative_ci_high": candidate["none_relative_ci_high"],
                "none_interpretation": candidate["none_interpretation"],
                "legacy_balance_relative_suppression": candidate[
                    "legacy_balance_relative_suppression"
                ],
                "legacy_balance_relative_ci_low": candidate[
                    "legacy_balance_relative_ci_low"
                ],
                "legacy_balance_relative_ci_high": candidate[
                    "legacy_balance_relative_ci_high"
                ],
                "legacy_balance_interpretation": candidate[
                    "legacy_balance_interpretation"
                ],
                "prior_high_relative_suppression": candidate[
                    "prior_high_relative_suppression"
                ],
                "prior_high_relative_ci_low": candidate[
                    "prior_high_relative_ci_low"
                ],
                "prior_high_relative_ci_high": candidate[
                    "prior_high_relative_ci_high"
                ],
                "prior_high_interpretation": candidate[
                    "prior_high_interpretation"
                ],
                "positive_seed_blocks_vs_none": candidate[
                    "none_positive_seed_blocks"
                ],
                "negative_seed_blocks_vs_none": candidate[
                    "none_negative_seed_blocks"
                ],
                "secondary_candidate_count": int(len(other)),
                "secondary_reduction_count_vs_none": int(
                    other["none_interpretation"].eq("reduction").sum()
                ),
                "secondary_condition_ids": ";".join(
                    other["condition_id"].astype(str)
                ),
                "secondary_none_interpretations": ";".join(
                    other["none_interpretation"].astype(str)
                ),
                "conclusion_code": _network_conclusion_code(candidate),
                "candidate_reselected_after_test": False,
            }
        )
    return pd.DataFrame(rows)


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def build_final_decision(
    final_results: pd.DataFrame,
    network_conclusions: pd.DataFrame,
    *,
    final_test_seeds: list[int],
) -> dict[str, Any]:
    networks: list[dict[str, Any]] = []
    for _, conclusion in network_conclusions.sort_values("network").iterrows():
        network = str(conclusion["network"])
        candidates = final_results[final_results["network"] == network].sort_values(
            "selection_order"
        )
        networks.append(
            {
                "network": network,
                "evidence_scope": conclusion["evidence_scope"],
                "conclusion_code": conclusion["conclusion_code"],
                "primary_condition_id": conclusion["primary_condition_id"],
                "primary_none_effect": {
                    "relative_suppression": _native(
                        conclusion["none_relative_suppression"]
                    ),
                    "ci_low": _native(conclusion["none_relative_ci_low"]),
                    "ci_high": _native(conclusion["none_relative_ci_high"]),
                    "interpretation": conclusion["none_interpretation"],
                },
                "frozen_candidates": [
                    {
                        "condition_id": row["condition_id"],
                        "selection_order": int(row["selection_order"]),
                        "reporting_role": row["reporting_role"],
                        "qualified": bool(row["qualified"]),
                        "region_id": row["region_id"],
                        "certainty": _native(row["certainty"]),
                        "effectiveness": _native(row["effectiveness"]),
                        "test_mean_jcum": _native(row["estimate"]),
                        "none_relative_suppression": _native(
                            row["none_relative_suppression"]
                        ),
                        "none_relative_ci_low": _native(
                            row["none_relative_ci_low"]
                        ),
                        "none_relative_ci_high": _native(
                            row["none_relative_ci_high"]
                        ),
                        "none_interpretation": row["none_interpretation"],
                    }
                    for _, row in candidates.iterrows()
                ],
            }
        )
    return {
        "status": "final_evaluation_complete",
        "candidate_set_matches_stage7": True,
        "candidate_selection_frozen_before_test": True,
        "candidates_reselected": False,
        "test_results_used_for_candidate_selection": False,
        "final_test_seeds": [int(seed) for seed in final_test_seeds],
        "primary_candidate_policy": (
            "selection_order 1 is the predesignated main candidate; orders 2 and "
            "3 remain descriptive region representatives"
        ),
        "facebook_policy": (
            "Facebook results remain exploratory because Stage 7 found no "
            "qualified candidate"
        ),
        "minimum_important_effect_threshold": None,
        "networks": networks,
    }
