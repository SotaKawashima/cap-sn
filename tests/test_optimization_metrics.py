from __future__ import annotations

import unittest

import pandas as pd

from analysis.optimization_metrics import (
    LEGACY_METRIC_NAME,
    MetricValidationError,
    OBJECTIVE_NAME,
    compute_selfish_metrics,
    validate_pop_agent_consistency,
)


def pop_frame(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["num_iter", "t", "num_selfish"])


class ComputeSelfishMetricsTests(unittest.TestCase):
    def test_same_cumulative_count_has_same_objective_despite_duration(self):
        short = pop_frame([(0, 0, 0), (0, 1, 6), (0, 2, 4)])
        long = pop_frame(
            [(0, 0, 0), (0, 1, 2), (0, 2, 2), (0, 3, 2), (0, 4, 2), (0, 5, 2)]
        )

        short_result = compute_selfish_metrics(
            short, num_agents=100, expected_iterations=1
        )
        long_result = compute_selfish_metrics(
            long, num_agents=100, expected_iterations=1
        )

        self.assertAlmostEqual(short_result.objective_value, 0.10)
        self.assertAlmostEqual(long_result.objective_value, 0.10)
        self.assertNotEqual(
            short_result.summary[LEGACY_METRIC_NAME],
            long_result.summary[LEGACY_METRIC_NAME],
        )

    def test_fewer_selfish_agents_has_lower_objective(self):
        fewer = pop_frame([(0, 0, 0), (0, 1, 5)])
        more = pop_frame([(0, 0, 0), (0, 1, 8)])

        fewer_result = compute_selfish_metrics(
            fewer, num_agents=100, expected_iterations=1
        )
        more_result = compute_selfish_metrics(
            more, num_agents=100, expected_iterations=1
        )

        self.assertLess(fewer_result.objective_value, more_result.objective_value)

    def test_objective_matches_manual_iteration_average_and_keeps_zero_iteration(self):
        frame = pop_frame(
            [
                (2, 1, 0),
                (0, 1, 10),
                (1, 0, 0),
                (0, 0, 0),
                (2, 0, 0),
                (1, 1, 20),
            ]
        )

        result = compute_selfish_metrics(
            frame, num_agents=100, expected_iterations=3
        )

        self.assertAlmostEqual(result.objective_value, (0.10 + 0.20 + 0.00) / 3)
        self.assertEqual(result.summary["n_iterations"], 3)
        self.assertEqual(result.summary["n_zero_selfish_iterations"], 1)
        self.assertEqual(result.per_iteration["num_iter"].tolist(), [0, 1, 2])

    def test_all_zero_counts_produce_zero_objective_and_undefined_timing(self):
        frame = pop_frame([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])

        result = compute_selfish_metrics(
            frame, num_agents=100, expected_iterations=2
        )

        self.assertEqual(result.objective_value, 0.0)
        self.assertEqual(result.summary["peak_new_selfish_ratio"], 0.0)
        self.assertIsNone(result.summary["mean_first_selfish_step"])
        self.assertTrue(result.per_iteration["first_selfish_step"].isna().all())

    def test_timing_metrics_are_computed_from_cumulative_events(self):
        frame = pop_frame(
            [(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 1), (0, 4, 0)]
        )

        result = compute_selfish_metrics(
            frame, num_agents=100, expected_iterations=1
        )
        row = result.per_iteration.iloc[0]

        self.assertEqual(row["first_selfish_step"], 1)
        self.assertEqual(row["last_selfish_step"], 3)
        self.assertEqual(row["t50_selfish_step"], 2)
        self.assertEqual(row["t90_selfish_step"], 3)
        self.assertAlmostEqual(row["selfish_timing_centroid"], 2.0)
        self.assertEqual(row["selfish_span_steps"], 3)
        self.assertEqual(row["active_selfish_steps"], 3)

    def test_cumulative_count_over_num_agents_is_rejected(self):
        frame = pop_frame([(0, 0, 60), (0, 1, 50)])

        with self.assertRaisesRegex(
            MetricValidationError, "cumulative selfish count exceeds num_agents"
        ):
            compute_selfish_metrics(frame, num_agents=100, expected_iterations=1)

    def test_missing_iteration_is_rejected(self):
        frame = pop_frame([(0, 0, 0), (0, 1, 5)])

        with self.assertRaisesRegex(MetricValidationError, "missing=\\[1\\]"):
            compute_selfish_metrics(frame, num_agents=100, expected_iterations=2)

    def test_missing_column_empty_data_and_duplicate_step_are_rejected(self):
        with self.assertRaisesRegex(MetricValidationError, "missing required columns"):
            compute_selfish_metrics(
                pd.DataFrame({"num_iter": [0], "t": [0]}),
                num_agents=100,
                expected_iterations=1,
            )

        with self.assertRaisesRegex(MetricValidationError, "pop data is empty"):
            compute_selfish_metrics(
                pop_frame([]), num_agents=100, expected_iterations=1
            )

        duplicate = pop_frame([(0, 0, 0), (0, 0, 1)])
        with self.assertRaisesRegex(MetricValidationError, "duplicate"):
            compute_selfish_metrics(
                duplicate, num_agents=100, expected_iterations=1
            )

    def test_non_integer_negative_and_missing_counts_are_rejected(self):
        invalid_values = [1.5, -1, None]
        for value in invalid_values:
            with self.subTest(value=value):
                frame = pop_frame([(0, 0, value)])
                with self.assertRaises(MetricValidationError):
                    compute_selfish_metrics(
                        frame, num_agents=100, expected_iterations=1
                    )

    def test_summary_uses_named_primary_objective(self):
        result = compute_selfish_metrics(
            pop_frame([(0, 0, 2)]),
            num_agents=10,
            expected_iterations=1,
        )

        self.assertEqual(result.summary["objective_name"], OBJECTIVE_NAME)
        self.assertEqual(result.summary[OBJECTIVE_NAME], 0.2)

    def test_pop_and_agent_counts_match_including_zero_event_iteration(self):
        pop = pop_frame(
            [(0, 0, 0), (0, 1, 2), (1, 0, 0), (1, 1, 0)]
        )
        agents = pd.DataFrame(
            [(0, 1, 3), (0, 1, 8)],
            columns=["num_iter", "t", "agent_idx"],
        )

        comparison = validate_pop_agent_consistency(
            pop,
            agents,
            num_agents=10,
            expected_iterations=2,
        )

        self.assertEqual(
            comparison["pop_cumulative_selfish_count"].tolist(), [2, 0]
        )
        self.assertEqual(comparison["agent_unique_selfish_count"].tolist(), [2, 0])
        self.assertEqual(comparison["difference"].tolist(), [0, 0])

    def test_pop_agent_mismatch_and_duplicate_actor_are_rejected(self):
        pop = pop_frame([(0, 0, 0), (0, 1, 2)])
        mismatch = pd.DataFrame(
            [(0, 1)], columns=["num_iter", "agent_idx"]
        )
        with self.assertRaisesRegex(MetricValidationError, "counts disagree"):
            validate_pop_agent_consistency(
                pop, mismatch, num_agents=10, expected_iterations=1
            )

        duplicate = pd.DataFrame(
            [(0, 1), (0, 1)], columns=["num_iter", "agent_idx"]
        )
        with self.assertRaisesRegex(MetricValidationError, "more than once"):
            validate_pop_agent_consistency(
                pop, duplicate, num_agents=10, expected_iterations=1
            )


if __name__ == "__main__":
    unittest.main()
