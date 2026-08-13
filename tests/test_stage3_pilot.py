from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from analysis.pilot_analysis import (
    _fixed_condition_mismatches,
    _manifest_mismatches,
    _optimization_mismatches,
    assess_optimization_budget,
    assess_optimization_budget_by_tolerance,
    build_best_so_far,
    build_paired_differences,
    build_prefix_estimates,
    build_rank_stability,
    build_selection_regret,
    recommend_iteration_count,
    recommend_iteration_count_by_selection_regret,
)
from run_stage3_pilot import (
    build_optimization_specs,
    build_precision_specs,
    command_for_spec,
    load_protocol,
)


class Stage3ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_precision_protocol_contains_90_fixed_runs(self):
        specs = build_precision_specs(self.protocol)

        self.assertEqual(len(specs), 3 * 6 * 5)
        self.assertEqual(len({spec.key for spec in specs}), len(specs))
        self.assertTrue(all(spec.iterations == 200 for spec in specs))
        self.assertTrue(all(spec.raw_level == "all" for spec in specs))

    def test_amended_protocol_fixes_selection_and_budget_rules(self):
        self.assertEqual(self.protocol["protocol_id"], "stage3_pilot_v3")
        self.assertEqual(
            self.protocol["amendment"]["parent_protocol_id"],
            "stage3_pilot_v2",
        )
        precision_rule = self.protocol["precision_phase"]["decision_rule"]
        self.assertEqual(precision_rule["selected_iterations"], 100)
        self.assertEqual(precision_rule["selection_regret_tolerance"], 0.005)
        budget = self.protocol["optimization_budget_phase"]
        self.assertEqual(
            budget["networks"], ["ba1000", "facebook", "wiki_vote"]
        )
        self.assertEqual(budget["iterations"], 100)
        self.assertEqual(
            budget["decision_rule"]["absolute_improvement_tolerance"],
            0.005,
        )
        self.assertEqual(
            budget["decision_rule"][
                "minimum_passing_replicates_per_network_method"
            ],
            2,
        )

    def test_stage2_good_is_network_specific(self):
        specs = [
            spec
            for spec in build_precision_specs(self.protocol)
            if spec.condition_id == "stage2_good" and spec.simulator_seed == 10001
        ]
        values = {
            spec.network: (spec.certainty, spec.effectiveness) for spec in specs
        }

        self.assertEqual(values["ba1000"], (0.85, 0.95))
        self.assertEqual(values["facebook"], (0.85, 0.85))
        self.assertEqual(values["wiki_vote"], (0.95, 0.75))

    def test_no_intervention_command_has_no_design_variables(self):
        spec = next(
            spec
            for spec in build_precision_specs(self.protocol)
            if spec.condition_id == "none"
        )
        command = command_for_spec(
            spec, experiment_id="20260812_120000_pilot_precision_v01"
        )

        self.assertIn("--no-intervention", command)
        self.assertNotIn("--certainty", command)
        self.assertNotIn("--effectiveness", command)

    def test_child_command_receives_selected_output_root(self):
        spec = build_precision_specs(self.protocol)[0]
        output_root = Path("/tmp/stage3-custom-output")
        command = command_for_spec(
            spec,
            experiment_id="20260812_120000_pilot_precision_v01",
            output_root=output_root,
        )

        position = command.index("--output-root")
        self.assertEqual(command[position + 1], output_root.resolve().as_posix())

    def test_optimization_protocol_has_three_networks_methods_and_replicates(self):
        specs = build_optimization_specs(self.protocol, iterations=100)

        self.assertEqual(len(specs), 27)
        self.assertEqual(
            {spec.network for spec in specs},
            {"ba1000", "facebook", "wiki_vote"},
        )
        self.assertEqual(
            {spec.method for spec in specs},
            {"bo_gp", "cma_es", "random_search"},
        )
        self.assertEqual(len({spec.key for spec in specs}), 27)
        self.assertEqual(len({spec.optimizer_seed for spec in specs}), 9)
        self.assertTrue(
            all(
                len(
                    {
                        spec.optimizer_replicate
                        for spec in specs
                        if spec.network == network and spec.method == method
                    }
                )
                == 3
                for network in {spec.network for spec in specs}
                for method in {spec.method for spec in specs}
            )
        )
        self.assertTrue(all(spec.trials == 100 for spec in specs))

    def test_optimization_iterations_must_come_from_precision_prefix(self):
        with self.assertRaisesRegex(ValueError, "precision prefixes"):
            build_optimization_specs(self.protocol, iterations=30)


class Stage3AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_fixed_manifest_protocol_validation(self):
        spec = next(
            spec
            for spec in build_precision_specs(self.protocol)
            if spec.condition_id == "high_effectiveness"
        )
        manifest = {
            "stage": "stage3_pilot",
            "run_type": "fixed_condition",
            "network": {"id": spec.network},
            "runtime": {
                "simulator_seed": spec.simulator_seed,
                "iteration_count": spec.iterations,
            },
            "objective": {
                "name": "cumulative_selfish_fraction",
                "definition_version": "cumulative_selfish_fraction_v1",
            },
            "intervention": {
                "condition_id": spec.condition_id,
                "enabled": True,
                "applied_parameters": {
                    "certainty": spec.certainty,
                    "effectiveness": spec.effectiveness,
                },
            },
        }

        self.assertEqual(
            _manifest_mismatches(
                manifest, spec, expected_run_type="fixed_condition"
            ),
            [],
        )
        self.assertEqual(_fixed_condition_mismatches(manifest, spec), [])
        manifest["runtime"]["simulator_seed"] = 99999
        self.assertTrue(
            _manifest_mismatches(
                manifest, spec, expected_run_type="fixed_condition"
            )
        )

    def test_optimization_manifest_protocol_validation(self):
        spec = build_optimization_specs(self.protocol, iterations=100)[0]
        manifest = {
            "stage": "stage3_pilot",
            "run_type": "single_objective_optimization",
            "network": {"id": spec.network},
            "runtime": {
                "simulator_seed": spec.simulator_seed,
                "iteration_count": spec.iterations,
            },
            "objective": {
                "name": "cumulative_selfish_fraction",
                "definition_version": "cumulative_selfish_fraction_v1",
            },
            "optimization": {
                "method": spec.method,
                "optimizer_replicate": spec.optimizer_replicate,
                "optimizer_seed": spec.optimizer_seed,
                "n_trials_requested": spec.trials,
                "startup_trials": spec.startup_trials,
            },
        }

        self.assertEqual(
            _manifest_mismatches(
                manifest,
                spec,
                expected_run_type="single_objective_optimization",
            ),
            [],
        )
        self.assertEqual(_optimization_mismatches(manifest, spec), [])
        manifest["optimization"]["method"] = "random_search"
        self.assertTrue(_optimization_mismatches(manifest, spec))

    @staticmethod
    def _iteration_data() -> pd.DataFrame:
        rows = []
        condition_base = {"reference": 0.30, "candidate": 0.28, "third": 0.32}
        for seed_index, simulator_seed in enumerate(range(10001, 10006)):
            seed_shift = seed_index * 0.0002
            for condition, base in condition_base.items():
                for num_iter in range(4):
                    rows.append(
                        {
                            "network": "ba1000",
                            "condition_id": condition,
                            "simulator_seed": simulator_seed,
                            "num_iter": num_iter,
                            "cumulative_selfish_fraction": (
                                base + seed_shift + num_iter * 0.0001
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    def test_prefix_and_paired_difference_tables(self):
        data = self._iteration_data()
        prefix = build_prefix_estimates(data, prefixes=[2, 4])
        differences = build_paired_differences(
            data,
            prefixes=[2, 4],
            comparisons=[
                {
                    "id": "candidate_vs_reference",
                    "reference": "reference",
                    "candidate": "candidate",
                }
            ],
            repetitions=200,
            seed=7,
        )

        self.assertEqual(len(prefix), 2 * 3 * 5)
        self.assertTrue((differences["estimate"].round(8) == 0.02).all())
        self.assertTrue((differences["ci_half_width"] >= 0).all())

    def test_iteration_recommendation_uses_precision_and_rank_rules(self):
        data = self._iteration_data()
        differences = build_paired_differences(
            data,
            prefixes=[2, 4],
            comparisons=[
                {
                    "id": "candidate_vs_reference",
                    "reference": "reference",
                    "candidate": "candidate",
                }
            ],
            repetitions=200,
            seed=8,
        )
        _, rank_summary = build_rank_stability(data, prefixes=[2, 4])
        decision = recommend_iteration_count(
            differences,
            rank_summary,
            prefixes=[2, 4],
            delta_min=0.01,
            ci_fraction=0.5,
            minimum_median_rank_spearman=0.9,
            minimum_best_condition_agreement=0.8,
        )

        self.assertEqual(decision["status"], "recommended")
        self.assertEqual(decision["recommended_iterations"], 2)

    def test_selection_regret_recommends_smallest_passing_prefix(self):
        detail, summary = build_selection_regret(
            self._iteration_data(), prefixes=[2, 4], tolerance=0.005
        )
        decision = recommend_iteration_count_by_selection_regret(
            summary,
            prefixes=[2, 4],
            tolerance=0.005,
            minimum_block_pass_rate_per_network=1.0,
        )

        self.assertTrue((detail["selection_regret"] == 0).all())
        self.assertTrue((summary["block_pass_rate"] == 1).all())
        self.assertEqual(decision["status"], "recommended")
        self.assertEqual(decision["recommended_iterations"], 2)

    def test_selection_regret_rule_rejects_an_excessive_prefix(self):
        summary = pd.DataFrame(
            [
                {
                    "network": "ba1000",
                    "prefix_iterations": 50,
                    "max_regret": 0.006,
                    "block_pass_rate": 0.8,
                },
                {
                    "network": "facebook",
                    "prefix_iterations": 50,
                    "max_regret": 0.0,
                    "block_pass_rate": 1.0,
                },
                {
                    "network": "ba1000",
                    "prefix_iterations": 100,
                    "max_regret": 0.004,
                    "block_pass_rate": 1.0,
                },
                {
                    "network": "facebook",
                    "prefix_iterations": 100,
                    "max_regret": 0.003,
                    "block_pass_rate": 1.0,
                },
            ]
        )
        decision = recommend_iteration_count_by_selection_regret(
            summary,
            prefixes=[50, 100],
            tolerance=0.005,
            minimum_block_pass_rate_per_network=1.0,
        )

        self.assertEqual(decision["recommended_iterations"], 100)

    def test_best_so_far_and_budget_assessment(self):
        rows = []
        for method in ("bo_gp", "cma_es", "random_search"):
            for replicate in (1, 2, 3):
                for trial in range(100):
                    rows.append(
                        {
                            "run_key": f"{method}:{replicate}",
                            "network": "ba1000",
                            "method": method,
                            "optimizer_replicate": replicate,
                            "trial": trial,
                            "value": 0.30 - min(trial, 49) * 0.0001,
                        }
                    )
        trials = pd.DataFrame(rows)
        best, checkpoints, improvements = build_best_so_far(
            trials, checkpoints=[50, 75, 100]
        )
        decision = assess_optimization_budget(
            checkpoints,
            checkpoints=[50, 75, 100],
            delta_min=0.002,
            late_improvement_fraction=0.5,
        )

        self.assertEqual(len(best), 900)
        self.assertEqual(len(checkpoints), 27)
        self.assertTrue((improvements["median"] == 0).all())
        self.assertEqual(decision["recommended_evaluations"], 50)

        amended_decision = assess_optimization_budget_by_tolerance(
            checkpoints,
            checkpoints=[50, 75, 100],
            improvement_tolerance=0.005,
            minimum_passing_replicates_per_network_method=2,
        )
        self.assertEqual(amended_decision["status"], "recommended")
        self.assertEqual(amended_decision["recommended_evaluations"], 50)

    def test_budget_assessment_requires_extension_when_two_runs_do_not_pass(self):
        rows = []
        for method in ("bo_gp", "cma_es", "random_search"):
            for replicate, improvement in zip((1, 2, 3), (0.001, 0.010, 0.020)):
                for checkpoint in (50, 75, 100):
                    rows.append(
                        {
                            "run_key": f"{method}:{replicate}",
                            "network": "ba1000",
                            "method": method,
                            "optimizer_replicate": replicate,
                            "checkpoint": checkpoint,
                            "best_so_far": (
                                0.20 + improvement if checkpoint < 100 else 0.20
                            ),
                        }
                    )
        decision = assess_optimization_budget_by_tolerance(
            pd.DataFrame(rows),
            checkpoints=[50, 75, 100],
            improvement_tolerance=0.005,
            minimum_passing_replicates_per_network_method=2,
        )

        self.assertEqual(decision["status"], "extend_beyond_tested_maximum")
        self.assertIsNone(decision["recommended_evaluations"])

    def test_budget_assessment_requires_every_network_method_to_pass(self):
        rows = []
        networks = ("ba1000", "facebook", "wiki_vote")
        methods = ("bo_gp", "cma_es", "random_search")
        for network in networks:
            for method in methods:
                improvements = (0.001, 0.002, 0.003)
                if network == "facebook" and method == "random_search":
                    improvements = (0.001, 0.010, 0.020)
                for replicate, improvement in zip((1, 2, 3), improvements):
                    for checkpoint in (50, 75, 100):
                        rows.append(
                            {
                                "run_key": f"{network}:{method}:{replicate}",
                                "network": network,
                                "method": method,
                                "optimizer_replicate": replicate,
                                "checkpoint": checkpoint,
                                "best_so_far": (
                                    0.20 + improvement
                                    if checkpoint < 100
                                    else 0.20
                                ),
                            }
                        )

        decision = assess_optimization_budget_by_tolerance(
            pd.DataFrame(rows),
            checkpoints=[50, 75, 100],
            improvement_tolerance=0.005,
            minimum_passing_replicates_per_network_method=2,
        )

        self.assertEqual(decision["status"], "extend_beyond_tested_maximum")
        failed = [
            row
            for row in decision["diagnostics"][0]["network_methods"]
            if not row["passes"]
        ]
        self.assertEqual(
            [(row["network"], row["method"]) for row in failed],
            [("facebook", "random_search")],
        )


if __name__ == "__main__":
    unittest.main()
