from __future__ import annotations

import unittest

import pandas as pd

from analysis.fixed_confirmation_analysis import (
    FixedConfirmationData,
    _paired_bootstrap_effect,
    _summarize_stage4_info,
    build_comparison_table,
    build_factorial_tables,
    replace_fixed_confirmation_condition,
)
from analysis.optimization_metrics import OBJECTIVE_NAME
from run_stage4_fixed_confirmation import load_protocol


class Stage4ConfirmationAnalysisTests(unittest.TestCase):
    def test_condition_replacement_preserves_other_conditions(self):
        def make_data(
            conditions: list[tuple[str, float]],
        ) -> FixedConfirmationData:
            common_rows = []
            iteration_rows = []
            info_long_rows = []
            pop_rows = []
            for condition_id, value in conditions:
                run_key = f"ba1000:{condition_id}:simseed1"
                common = {
                    "run_key": run_key,
                    "network": "ba1000",
                    "condition_id": condition_id,
                    "simulator_seed": 1,
                }
                common_rows.append({**common, OBJECTIVE_NAME: value})
                iteration_rows.append(
                    {**common, "num_iter": 0, OBJECTIVE_NAME: value}
                )
                info_long_rows.append(
                    {**common, "info_label": 0, "count": value}
                )
                pop_rows.append(
                    {**common, "num_iter": 0, "t": 0, "num_selfish": value}
                )

            runs = pd.DataFrame(common_rows)
            return FixedConfirmationData(
                iterations=pd.DataFrame(iteration_rows),
                runs=runs,
                audit=runs.assign(valid=True),
                info_runs=runs.rename(columns={OBJECTIVE_NAME: "info_total"}),
                info_long=pd.DataFrame(info_long_rows),
                pop_events=pd.DataFrame(pop_rows),
            )

        original = make_data([("none", 0.40), ("prior_high", 0.30)])
        correction = make_data([("prior_high", 0.20)])

        merged = replace_fixed_confirmation_condition(
            original,
            correction,
            condition_id="prior_high",
        )

        values = merged.runs.set_index("condition_id")[OBJECTIVE_NAME]
        self.assertAlmostEqual(values["none"], 0.40)
        self.assertAlmostEqual(values["prior_high"], 0.20)
        self.assertEqual(len(merged.runs), len(original.runs))
        self.assertEqual(len(merged.iterations), len(original.iterations))

    def test_no_intervention_accepts_absent_behavior_guiding_label(self):
        rows = []
        for iteration in range(2):
            for label in (0, 1, 2):
                rows.append(
                    {
                        "num_iter": iteration,
                        "t": 0,
                        "info_label": label,
                        "num_posted": 1,
                        "num_received": 2,
                        "num_shared": 3,
                        "num_viewed": 4,
                        "num_fst_viewed": 5,
                    }
                )

        summary, wide = _summarize_stage4_info(
            pd.DataFrame(rows),
            intervention_enabled=False,
            num_agents=10,
            expected_iterations=2,
        )

        behavior = summary[summary["info_name"] == "behavior_guiding"].iloc[0]
        self.assertEqual(behavior["shared_sum"], 0)
        self.assertEqual(wide["behavior_guiding_fst_viewed_sum"], 0.0)

    def test_paired_bootstrap_preserves_exact_iteration_difference(self):
        rows = []
        for seed, baseline in ((1, 0.40), (2, 0.50)):
            for iteration in range(6):
                rows.append(
                    {
                        "simulator_seed": seed,
                        "reference": baseline + iteration * 0.001,
                        "candidate": baseline - 0.10 + iteration * 0.001,
                    }
                )
        result = _paired_bootstrap_effect(
            pd.DataFrame(rows),
            reference_column="reference",
            candidate_column="candidate",
            block_column="simulator_seed",
            repetitions=500,
            seed=10,
        )

        self.assertAlmostEqual(result["absolute_suppression"], 0.10)
        self.assertAlmostEqual(result["absolute_ci_low"], 0.10)
        self.assertAlmostEqual(result["absolute_ci_high"], 0.10)
        self.assertEqual(result["interpretation"], "reduction")

    def test_comparison_pairs_same_seed_and_iteration(self):
        rows = []
        for seed in (1, 2):
            for iteration in range(5):
                for condition, value in (("none", 0.30), ("candidate", 0.25)):
                    rows.append(
                        {
                            "network": "ba1000",
                            "condition_id": condition,
                            "simulator_seed": seed,
                            "num_iter": iteration,
                            OBJECTIVE_NAME: value + seed * 0.001,
                        }
                    )
        result = build_comparison_table(
            pd.DataFrame(rows),
            comparisons=[
                {
                    "id": "candidate_vs_none",
                    "reference": "none",
                    "candidate": "candidate",
                }
            ],
            family="test",
            repetitions=500,
            seed=20,
        ).iloc[0]

        self.assertAlmostEqual(result["absolute_suppression"], 0.05)
        self.assertAlmostEqual(result["equivalent_agents"], 50.0)
        self.assertAlmostEqual(
            result["relative_suppression"],
            0.05 / 0.3015,
        )

    def test_corner_interaction_matches_registered_definition(self):
        protocol = load_protocol()
        values = {
            "grid_c050_e050": 0.40,
            "grid_c050_e075": 0.35,
            "grid_c050_e100": 0.30,
            "grid_c075_e050": 0.38,
            "grid_c075_e075": 0.31,
            "grid_c075_e100": 0.26,
            "grid_c100_e050": 0.35,
            "grid_c100_e075": 0.27,
            "grid_c100_e100": 0.20,
        }
        rows = []
        for seed in (1, 2):
            for iteration in range(5):
                for condition, value in values.items():
                    rows.append(
                        {
                            "network": "ba1000",
                            "condition_id": condition,
                            "simulator_seed": seed,
                            "num_iter": iteration,
                            OBJECTIVE_NAME: value,
                        }
                    )

        contrasts, interaction = build_factorial_tables(
            pd.DataFrame(rows),
            protocol,
            repetitions=500,
            seed=30,
        )

        self.assertEqual(len(contrasts), 6)
        self.assertAlmostEqual(interaction.iloc[0]["estimate"], 0.05)
        self.assertEqual(
            interaction.iloc[0]["interpretation"], "interaction_detected"
        )


if __name__ == "__main__":
    unittest.main()
