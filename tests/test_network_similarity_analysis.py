from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_network_similarity import FAMILIES, parse_args
from analysis.network_similarity_analysis import (
    build_good_region_summary,
    build_good_region_transfer,
    build_pairwise_similarity,
    build_reference_comparisons,
    build_shared_random_points,
)


class AnalysisCliTests(unittest.TestCase):
    def test_default_scope_includes_all_network_families(self):
        args = parse_args(["--experiment-root", "experiment"])

        self.assertEqual(args.families, list(FAMILIES))
        self.assertEqual(args.analysis_id, "network_similarity_analysis_v01")


def make_common_random_trials(*, reverse_second: bool = False) -> pd.DataFrame:
    rows = []
    for network in ("network_a", "network_b"):
        for index in range(300):
            replicate = index // 50 + 1
            evaluation = index % 50 + 1
            value = float(index)
            if network == "network_b" and reverse_second:
                value = float(299 - index)
            rows.append(
                {
                    "network": network,
                    "method": "random_search",
                    "optimizer_seed": 60000 + replicate,
                    "optimizer_replicate": replicate,
                    "evaluation": evaluation,
                    "applied_certainty": round(0.5 + index / 1000, 4),
                    "applied_effectiveness": round(1.0 - index / 1000, 4),
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


class SharedRandomPointTests(unittest.TestCase):
    def test_identical_rankings_have_complete_good_region_overlap(self):
        shared = build_shared_random_points(
            make_common_random_trials(),
            family_networks={"test_family": ["network_a", "network_b"]},
        )
        self.assertEqual(len(shared), 600)
        self.assertEqual(
            shared.groupby("network")["is_good_point"].sum().to_dict(),
            {"network_a": 30, "network_b": 30},
        )
        pair = build_pairwise_similarity(shared).iloc[0]
        self.assertAlmostEqual(pair["spearman_objective"], 1.0)
        self.assertEqual(pair["good_overlap_count"], 30)
        self.assertAlmostEqual(pair["good_overlap_jaccard"], 1.0)
        self.assertAlmostEqual(pair["chance_expected_overlap_count"], 3.0)

    def test_reversed_rankings_transfer_to_the_worst_decile(self):
        shared = build_shared_random_points(
            make_common_random_trials(reverse_second=True),
            family_networks={"test_family": ["network_a", "network_b"]},
        )
        pair = build_pairwise_similarity(shared).iloc[0]
        self.assertAlmostEqual(pair["spearman_objective"], -1.0)
        self.assertEqual(pair["good_overlap_count"], 0)
        transfers = build_good_region_transfer(shared)
        self.assertTrue((transfers["target_good_overlap_count"] == 0).all())
        self.assertTrue((transfers["target_percentile_median"] > 90.0).all())

    def test_coordinate_mismatch_is_rejected(self):
        trials = make_common_random_trials()
        mismatch = trials.index[
            trials["network"].eq("network_b")
            & trials["evaluation"].eq(1)
            & trials["optimizer_replicate"].eq(1)
        ][0]
        trials.loc[mismatch, "applied_certainty"] += 0.0001
        with self.assertRaisesRegex(ValueError, "coordinates do not align"):
            build_shared_random_points(
                trials,
                family_networks={"test_family": ["network_a", "network_b"]},
            )

    def test_good_region_summary_reports_parameter_quartiles(self):
        shared = build_shared_random_points(
            make_common_random_trials(),
            family_networks={"test_family": ["network_a", "network_b"]},
        )
        summary = build_good_region_summary(shared)
        self.assertEqual(len(summary), 2)
        self.assertTrue((summary["n_good_points"] == 30).all())
        self.assertTrue(np.isfinite(summary["certainty_objective_spearman"]).all())


class ReferenceComparisonTests(unittest.TestCase):
    def test_reference_counts_are_separated_by_subset(self):
        trials = pd.DataFrame(
            {
                "network": ["n1"] * 6,
                "method": ["random_search"] * 3 + ["bo_gp"] * 3,
                "value": [0.8, 1.0, 1.2, 0.7, 0.9, 1.1],
            }
        )
        fixed = pd.DataFrame(
            {
                "network": ["n1", "n1"],
                "condition": ["none", "simple_max"],
                "value": [1.1, 0.95],
            }
        )
        result = build_reference_comparisons(trials, fixed)
        self.assertEqual(set(result["subset"]), {
            "all_adaptive_evaluations",
            "shared_random_search",
        })
        random = result.loc[result["subset"].eq("shared_random_search")].iloc[0]
        self.assertEqual(random["n_evaluations"], 3)
        self.assertEqual(random["count_better_than_none"], 2)
        self.assertEqual(random["count_better_than_simple_max"], 1)


if __name__ == "__main__":
    unittest.main()
