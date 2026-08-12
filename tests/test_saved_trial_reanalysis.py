from __future__ import annotations

import unittest

import pandas as pd

from analysis.saved_trial_reanalysis import (
    build_analysis_tables,
    discover_source_runs,
    reconstruct_applied_parameter,
    spearman_with_bootstrap,
    summarize_info_data,
)


class SavedTrialReanalysisTests(unittest.TestCase):
    def test_protocol_discovers_36_source_runs(self) -> None:
        runs = discover_source_runs()
        self.assertEqual(len(runs), 36)
        self.assertEqual(len({run.run_id for run in runs}), 36)

    def test_applied_parameter_reproduces_four_decimal_rounding(self) -> None:
        self.assertEqual(reconstruct_applied_parameter(0.556952299), 0.557)
        self.assertEqual(reconstruct_applied_parameter(0.904845901), 0.9048)
        with self.assertRaises(ValueError):
            reconstruct_applied_parameter(0.49)

    def test_information_summary_and_ratio(self) -> None:
        rows = []
        for num_iter in range(2):
            for label in range(4):
                rows.append(
                    {
                        "num_iter": num_iter,
                        "t": label,
                        "info_label": label,
                        "num_posted": 1,
                        "num_received": 2,
                        "num_shared": 10 if label == 0 else 20,
                        "num_viewed": 30 if label == 0 else 60,
                        "num_fst_viewed": 5 if label == 0 else 15,
                    }
                )
        summary, wide = summarize_info_data(
            pd.DataFrame(rows),
            num_agents=10,
            expected_iterations=2,
        )
        self.assertEqual(len(summary), 4)
        self.assertEqual(wide["misinformation_shared_sum"], 20)
        self.assertEqual(wide["corrective_shared_sum"], 40)
        self.assertEqual(wide["corrective_to_misinformation_shared_ratio"], 2)
        self.assertEqual(wide["corrective_to_misinformation_fst_viewed_ratio"], 3)

    def test_bootstrap_spearman_reports_monotonic_relation(self) -> None:
        result = spearman_with_bootstrap(
            range(20),
            range(20),
            n_bootstrap=50,
            seed=1,
        )
        self.assertEqual(result["rho"], 1.0)
        self.assertAlmostEqual(result["ci_low"], 1.0)
        self.assertAlmostEqual(result["ci_high"], 1.0)

    def test_analysis_tables_use_random_trials_for_parameter_response(self) -> None:
        rows = []
        for network_index, network in enumerate(
            ("ba1000", "facebook", "wiki_vote")
        ):
            for optseed in (4, 5, 6):
                for trial in range(10):
                    effectiveness = 0.5 + trial / 20
                    j_cum = 1.1 - effectiveness + network_index * 0.01
                    rows.append(
                        {
                            "trial_id": f"{network}:{optseed}:{trial}",
                            "network": network,
                            "method": "random",
                            "optseed": optseed,
                            "trial": trial,
                            "proposed_certainty": 0.5 + trial / 20,
                            "proposed_effectiveness": effectiveness,
                            "applied_certainty": 0.5 + trial / 20,
                            "applied_effectiveness": effectiveness,
                            "j_old_recalculated": j_cum / 10,
                            "j_cum": j_cum,
                            "j_peak": j_cum,
                            "mean_first_selfish_step": 1,
                            "mean_last_selfish_step": 5,
                            "mean_t50_selfish_step": 2,
                            "mean_t90_selfish_step": 4,
                            "mean_selfish_timing_centroid": 3,
                            "mean_selfish_span_steps": 5,
                            "mean_active_selfish_steps": 4,
                            "mean_recorded_steps": 10,
                            "corrective_to_misinformation_fst_viewed_ratio": 2,
                            "behavior_guiding_fst_viewed_per_agent_iteration": 1,
                            "corrective_fst_viewed_per_agent_iteration": 2,
                        }
                    )
        tables = build_analysis_tables(
            pd.DataFrame(rows),
            n_bootstrap=50,
            bootstrap_seed=1,
        )
        correlations = tables["parameter_correlations"]
        effectiveness = correlations[
            correlations["parameter"] == "applied_effectiveness"
        ]
        self.assertTrue((effectiveness["rho"] < -0.99).all())
        self.assertTrue(
            tables["random_pairing_audit"]["applied_points_equal"].all()
        )
        self.assertEqual(
            int(tables["random_duplicate_audit"]["applied_duplicate_count"].sum()),
            0,
        )
        ranked = tables["trial_rank_comparison"]
        self.assertTrue(
            (ranked.groupby("network")["old_low_10pct"].sum() == 3).all()
        )
        self.assertTrue(
            (ranked.groupby("network")["new_low_10pct"].sum() == 3).all()
        )


if __name__ == "__main__":
    unittest.main()
