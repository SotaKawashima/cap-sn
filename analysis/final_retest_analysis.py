"""Audit and report the revised final evaluation after candidate reselection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.candidate_validation_analysis import (
    CandidateValidationData,
    build_candidate_effects,
    build_seed_summary,
    load_candidate_validation,
)
from analysis.final_evaluation_analysis import build_final_condition_summary
from run_stage8_final_retest import (
    FINAL_CANDIDATE_ROLE,
    REFERENCE_IDS,
    STAGE,
    read_candidate_rows,
)


def load_final_retest(
    experiment_root: str,
    *,
    expected_specs: list[Any],
    numeric_tolerance: float = 1e-12,
) -> CandidateValidationData:
    """Load the revised final runs and reproduce metrics from retained pop data."""

    return load_candidate_validation(
        experiment_root,
        expected_specs=expected_specs,
        numeric_tolerance=numeric_tolerance,
        expected_stage=STAGE,
    )


def candidate_metadata(protocol: dict[str, Any]) -> pd.DataFrame:
    """Return the amended nine-candidate set frozen before revised testing."""

    frame = pd.DataFrame(read_candidate_rows(protocol))
    columns = [
        "network",
        "condition_id",
        "certainty",
        "effectiveness",
        "selection_order",
        "reporting_role",
        "selection_mode",
        "selection_role",
        "qualified",
        "region_id",
        "source_method",
        "source_optimizer_replicate",
        "candidate_source",
        "source_final_best",
        "stage7_validation_mean_jcum",
        "stage7_validation_between_seed_sd",
        "stage7_none_positive_seed_blocks",
        "stage7_none_relative_suppression",
        "stage7_none_relative_ci_low",
        "stage7_none_relative_ci_high",
        "stage7_simple_max_positive_seed_blocks",
        "stage7_simple_max_relative_suppression",
        "stage7_simple_max_relative_ci_low",
        "stage7_simple_max_relative_ci_high",
        "stage7_legacy_balance_relative_suppression",
        "stage7_legacy_balance_relative_ci_low",
        "stage7_legacy_balance_relative_ci_high",
        "stage7_prior_high_relative_suppression",
        "stage7_prior_high_relative_ci_low",
        "stage7_prior_high_relative_ci_high",
        "stage7_overall_order",
    ]
    result = frame[columns].copy()
    nonnumeric = {
        "network",
        "condition_id",
        "reporting_role",
        "selection_mode",
        "selection_role",
        "qualified",
        "region_id",
        "source_method",
        "candidate_source",
    }
    for column in (value for value in columns if value not in nonnumeric):
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["qualified"] = result["qualified"].astype(bool)
    if result.duplicated(["network", "condition_id"]).any():
        raise ValueError("revised final candidate metadata is not unique")
    return result.sort_values(
        ["network", "selection_order"], kind="stable"
    ).reset_index(drop=True)


def build_final_candidate_effects(
    iterations: pd.DataFrame,
    *,
    reference_id: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate a paired hierarchical effect against one frozen reference."""

    if reference_id not in REFERENCE_IDS:
        raise ValueError(f"unsupported revised-final reference: {reference_id}")
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
    """Report candidate-reference effects separately for every test seed block."""

    seed_summary = build_seed_summary(iterations)
    references = (
        seed_summary[seed_summary["condition_id"].isin(REFERENCE_IDS)][
            ["network", "simulator_seed", "condition_id", "mean_jcum"]
        ]
        .pivot(
            index=["network", "simulator_seed"],
            columns="condition_id",
            values="mean_jcum",
        )
        .reset_index()
    )
    if any(reference not in references for reference in REFERENCE_IDS):
        raise ValueError("one or more revised-final references are missing")

    candidates = seed_summary[
        seed_summary["condition_role"] == FINAL_CANDIDATE_ROLE
    ].copy()
    candidates = candidates.merge(
        references,
        on=["network", "simulator_seed"],
        how="left",
        validate="many_to_one",
    )
    if candidates[list(REFERENCE_IDS)].isna().any().any():
        raise ValueError("one or more revised-final reference means are missing")
    for reference in REFERENCE_IDS:
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
        "selection_role",
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
        raise ValueError("revised-final seed table contains an unfrozen candidate")
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
    effects_by_reference: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine all prespecified estimates without reranking candidates."""

    if set(effects_by_reference) != set(REFERENCE_IDS):
        raise ValueError("effects must contain each revised-final reference once")
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
    for reference in REFERENCE_IDS:
        result = result.merge(
            _effect_projection(effects_by_reference[reference], reference),
            on=["network", "condition_id"],
            how="left",
            validate="one_to_one",
        )
    if result["selection_order"].isna().any() or len(result) != len(metadata):
        raise ValueError("revised final results do not match the frozen candidates")
    return result.sort_values(
        ["network", "selection_order"], kind="stable"
    ).reset_index(drop=True)


def build_validation_retest_comparison(
    final_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare validation and revised-test estimates without feeding results back."""

    columns = [
        "network",
        "condition_id",
        "selection_order",
        "reporting_role",
        "selection_mode",
        "selection_role",
        "qualified",
        "region_id",
        "stage7_validation_mean_jcum",
        "estimate",
    ]
    for reference in REFERENCE_IDS:
        columns.extend(
            [
                f"stage7_{reference}_relative_suppression",
                f"{reference}_relative_suppression",
            ]
        )
    result = final_results[columns].copy()
    result = result.rename(columns={"estimate": "revised_test_mean_jcum"})
    result["revised_test_minus_validation_jcum"] = (
        result["revised_test_mean_jcum"]
        - result["stage7_validation_mean_jcum"]
    )
    result["revised_test_to_validation_jcum_ratio"] = (
        result["revised_test_mean_jcum"]
        / result["stage7_validation_mean_jcum"]
    )
    for reference in REFERENCE_IDS:
        result[f"{reference}_eta_test_minus_validation"] = (
            result[f"{reference}_relative_suppression"]
            - result[f"stage7_{reference}_relative_suppression"]
        )
    result["used_for_reselection"] = False
    return result


def _candidate_one_conclusion_code(row: pd.Series) -> str:
    scope = (
        "qualified_candidate_1"
        if row["selection_role"] == "qualified_candidate"
        else "exploratory_candidate_1"
    )
    none = str(row["none_interpretation"])
    simple_max = str(row["simple_max_interpretation"])
    return f"{scope}_{none}_vs_none_{simple_max}_vs_simple_max"


def build_network_conclusions(final_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate 1 and all-candidate directional counts by network."""

    candidate_one = final_results[final_results["selection_order"] == 1].copy()
    if len(candidate_one) != 3 or candidate_one["network"].nunique() != 3:
        raise ValueError("one frozen candidate 1 is required per network")
    rows: list[dict[str, Any]] = []
    for _, candidate in candidate_one.sort_values("network").iterrows():
        network = str(candidate["network"])
        network_candidates = final_results[
            final_results["network"] == network
        ].sort_values("selection_order")
        row: dict[str, Any] = {
            "network": network,
            "selection_mode": candidate["selection_mode"],
            "candidate_1_selection_role": candidate["selection_role"],
            "candidate_1_qualified": bool(candidate["qualified"]),
            "candidate_1_condition_id": candidate["condition_id"],
            "candidate_1_certainty": candidate["certainty"],
            "candidate_1_effectiveness": candidate["effectiveness"],
            "candidate_1_region_id": candidate["region_id"],
            "candidate_1_test_mean_jcum": candidate["estimate"],
            "candidate_1_test_jcum_ci_low": candidate["ci_low"],
            "candidate_1_test_jcum_ci_high": candidate["ci_high"],
            "candidate_count": int(len(network_candidates)),
            "qualified_candidate_count": int(
                network_candidates["qualified"].sum()
            ),
            "exploratory_candidate_count": int(
                (~network_candidates["qualified"]).sum()
            ),
            "candidate_ids_in_frozen_order": ";".join(
                network_candidates["condition_id"].astype(str)
            ),
            "candidate_reselected_after_test": False,
        }
        for reference in REFERENCE_IDS:
            row[f"candidate_1_{reference}_relative_suppression"] = candidate[
                f"{reference}_relative_suppression"
            ]
            row[f"candidate_1_{reference}_relative_ci_low"] = candidate[
                f"{reference}_relative_ci_low"
            ]
            row[f"candidate_1_{reference}_relative_ci_high"] = candidate[
                f"{reference}_relative_ci_high"
            ]
            row[f"candidate_1_{reference}_interpretation"] = candidate[
                f"{reference}_interpretation"
            ]
            row[f"candidate_1_positive_seed_blocks_vs_{reference}"] = candidate[
                f"{reference}_positive_seed_blocks"
            ]
            row[f"all_candidates_reduction_count_vs_{reference}"] = int(
                network_candidates[f"{reference}_interpretation"]
                .eq("reduction")
                .sum()
            )
            row[f"all_candidate_interpretations_vs_{reference}"] = ";".join(
                network_candidates[f"{reference}_interpretation"].astype(str)
            )
        row["conclusion_code"] = _candidate_one_conclusion_code(candidate)
        rows.append(row)
    return pd.DataFrame(rows)


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _effect_record(row: pd.Series, reference: str) -> dict[str, Any]:
    return {
        "reference": reference,
        "reference_estimate": _native(row[f"{reference}_reference_estimate"]),
        "absolute_suppression": _native(
            row[f"{reference}_absolute_suppression"]
        ),
        "absolute_ci_low": _native(row[f"{reference}_absolute_ci_low"]),
        "absolute_ci_high": _native(row[f"{reference}_absolute_ci_high"]),
        "relative_suppression": _native(
            row[f"{reference}_relative_suppression"]
        ),
        "relative_ci_low": _native(row[f"{reference}_relative_ci_low"]),
        "relative_ci_high": _native(row[f"{reference}_relative_ci_high"]),
        "positive_seed_blocks": int(row[f"{reference}_positive_seed_blocks"]),
        "negative_seed_blocks": int(row[f"{reference}_negative_seed_blocks"]),
        "equivalent_agents": _native(row[f"{reference}_equivalent_agents"]),
        "interpretation": str(row[f"{reference}_interpretation"]),
    }


def build_final_decision(
    final_results: pd.DataFrame,
    network_conclusions: pd.DataFrame,
    *,
    final_test_seeds: list[int],
    inspected_original_final_test_seeds: list[int],
) -> dict[str, Any]:
    """Record the frozen analysis decision without creating a new selection."""

    if set(final_test_seeds) & set(inspected_original_final_test_seeds):
        raise ValueError("revised and inspected original final-test seeds overlap")
    networks: list[dict[str, Any]] = []
    for _, conclusion in network_conclusions.sort_values("network").iterrows():
        network = str(conclusion["network"])
        candidates = final_results[final_results["network"] == network].sort_values(
            "selection_order"
        )
        networks.append(
            {
                "network": network,
                "selection_mode": conclusion["selection_mode"],
                "conclusion_code": conclusion["conclusion_code"],
                "candidate_1_condition_id": conclusion[
                    "candidate_1_condition_id"
                ],
                "candidate_1_selection_role": conclusion[
                    "candidate_1_selection_role"
                ],
                "frozen_candidates": [
                    {
                        "condition_id": row["condition_id"],
                        "selection_order": int(row["selection_order"]),
                        "selection_role": row["selection_role"],
                        "qualified": bool(row["qualified"]),
                        "region_id": row["region_id"],
                        "certainty": _native(row["certainty"]),
                        "effectiveness": _native(row["effectiveness"]),
                        "revised_test_mean_jcum": _native(row["estimate"]),
                        "effects": {
                            reference: _effect_record(row, reference)
                            for reference in REFERENCE_IDS
                        },
                    }
                    for _, row in candidates.iterrows()
                ],
            }
        )
    return {
        "status": "revised_final_evaluation_complete",
        "candidate_set_matches_amended_stage7": True,
        "candidate_selection_frozen_before_revised_test": True,
        "candidates_reranked": False,
        "candidates_reselected": False,
        "test_results_used_for_candidate_selection": False,
        "original_final_test_results_were_inspected_before_amendment": True,
        "original_final_test_seeds_reused": False,
        "revised_final_test_seeds": [int(seed) for seed in final_test_seeds],
        "inspected_original_final_test_seeds": [
            int(seed) for seed in inspected_original_final_test_seeds
        ],
        "candidate_1_policy": (
            "Selection order 1 was fixed by the amended Stage 7 rule before "
            "revised testing; orders 2 and 3 remain descriptive representatives."
        ),
        "reference_policy": {
            "none": "primary intervention-effect reference",
            "simple_max": "primary optimization-gain reference",
            "legacy_balance": "descriptive spring-research reference",
            "prior_high": "descriptive external-package benchmark",
        },
        "minimum_important_effect_threshold": None,
        "networks": networks,
    }
