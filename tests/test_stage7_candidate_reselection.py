from __future__ import annotations

import copy
import io
import json
import unittest
from collections import Counter
from contextlib import redirect_stdout

import pandas as pd

from analysis.candidate_reselection_analysis import (
    ALL_REFERENCES,
    build_candidate_block_performance,
    build_candidate_selection,
)
from analysis.candidate_validation_analysis import build_candidate_effects
from experiment_runtime import ExperimentConfigurationError
from run_stage7_candidate_reselection import (
    build_specs,
    command_for_spec,
    load_protocol,
    parse_args,
    run_candidate_reselection,
)


EXPERIMENT_ID = "20260915_120000_candidate_reselection_v01"


class CandidateReselectionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.specs = build_specs(cls.protocol)

    def test_protocol_builds_only_nine_simple_max_runs(self):
        self.assertEqual(len(self.specs), 9)
        self.assertEqual(len({spec.key for spec in self.specs}), 9)
        self.assertEqual(
            Counter(spec.network for spec in self.specs),
            Counter({"ba1000": 3, "facebook": 3, "wiki_vote": 3}),
        )
        self.assertEqual(
            Counter(spec.simulator_seed for spec in self.specs),
            Counter({40001: 3, 40002: 3, 40003: 3}),
        )
        self.assertTrue(all(spec.condition_id == "simple_max" for spec in self.specs))
        self.assertTrue(all(spec.certainty == 1.0 for spec in self.specs))
        self.assertTrue(all(spec.effectiveness == 1.0 for spec in self.specs))
        self.assertTrue(all(spec.iterations == 100 for spec in self.specs))
        self.assertTrue(all(spec.raw_level == "pop" for spec in self.specs))

    def test_simple_max_uses_generated_design_variables(self):
        command = command_for_spec(self.specs[0], experiment_id=EXPERIMENT_ID)
        self.assertNotIn("--intervention-opinion-csv", command)
        self.assertNotIn("--no-intervention", command)
        self.assertEqual(command[command.index("--certainty") + 1], "1.0")
        self.assertEqual(command[command.index("--effectiveness") + 1], "1.0")

    def test_seed_groups_are_disjoint_and_revised_test_is_reserved(self):
        groups = [
            set(self.protocol["seed_policy"][name])
            for name in (
                "development_and_fixed_confirmation",
                "stage6_exploration",
                "candidate_validation",
                "inspected_original_final_test",
                "stage9_structure_confirmation",
                "reserved_revised_final_test",
            )
        ]
        for index, group in enumerate(groups):
            for other in groups[index + 1 :]:
                self.assertTrue(group.isdisjoint(other))
        self.assertEqual(groups[-1], {70001, 70002, 70003, 70004, 70005})

    def test_changed_source_candidate_hash_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["source_validation"]["candidate_source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ExperimentConfigurationError, "candidate source SHA-256 mismatch"
        ):
            build_specs(protocol)

    def test_dry_run_reports_nine_commands(self):
        args = parse_args(["--experiment-id", EXPERIMENT_ID, "--dry-run"])
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_candidate_reselection(args)
        data = json.loads(output.getvalue())
        self.assertIsNone(result)
        self.assertEqual(data["stage"], "stage7_candidate_reselection")
        self.assertEqual(data["full_run_count"], 9)
        self.assertEqual(data["selected_run_count"], 9)


class CandidateReselectionAnalysisTests(unittest.TestCase):
    @staticmethod
    def _iterations() -> pd.DataFrame:
        rows = []
        values = {
            "none": [0.30, 0.31, 0.29],
            "simple_max": [0.27, 0.28, 0.26],
            "legacy_balance": [0.20, 0.21, 0.19],
            "prior_high": [0.15, 0.16, 0.14],
            "cand_a": [0.25, 0.26, 0.24],
            "cand_b": [0.251, 0.261, 0.241],
            "cand_c": [0.26, 0.27, 0.25],
            "cand_d": [0.28, 0.29, 0.27],
        }
        parameters = {
            "cand_a": (0.60, 0.90),
            "cand_b": (0.62, 0.90),
            "cand_c": (0.85, 0.55),
            "cand_d": (0.95, 0.95),
        }
        roles = {
            "none": "no_intervention_baseline",
            "simple_max": "simple_design_variable_maximum_reference",
            "legacy_balance": "spring_research_reference",
            "prior_high": "prior_study_package_benchmark",
        }
        for condition_id, block_values in values.items():
            is_candidate = condition_id.startswith("cand_")
            certainty, effectiveness = parameters.get(condition_id, (0.8, 0.8))
            if condition_id == "none":
                certainty, effectiveness = (None, None)
            elif condition_id == "simple_max":
                certainty, effectiveness = (1.0, 1.0)
            for offset, value in enumerate(block_values):
                seed = 40001 + offset
                for num_iter, jitter in enumerate((-0.001, 0.001)):
                    rows.append(
                        {
                            "run_key": f"ba1000:{condition_id}:simseed{seed}",
                            "network": "ba1000",
                            "condition_id": condition_id,
                            "condition_role": (
                                "stage6_candidate"
                                if is_candidate
                                else roles[condition_id]
                            ),
                            "intervention_enabled": condition_id != "none",
                            "certainty": certainty,
                            "effectiveness": effectiveness,
                            "candidate_source": (
                                f"ba1000:bo_gp:{condition_id}"
                                if is_candidate
                                else None
                            ),
                            "source_method": "bo_gp" if is_candidate else None,
                            "source_optimizer_replicate": (
                                1 if is_candidate else None
                            ),
                            "source_final_best": 0.20 if is_candidate else None,
                            "simulator_seed": seed,
                            "num_agents": 1000,
                            "num_iter": num_iter,
                            "cumulative_selfish_fraction": value + jitter,
                            "peak_new_selfish_ratio": 0.1,
                        }
                    )
        return pd.DataFrame(rows)

    def test_selection_requires_none_and_simple_max_not_legacy(self):
        iterations = self._iterations()
        blocks = build_candidate_block_performance(iterations)
        effects = {
            reference: build_candidate_effects(
                iterations,
                reference_id=reference,
                repetitions=200,
                seed=100 + offset,
            )
            for offset, reference in enumerate(ALL_REFERENCES)
        }
        ranking, clusters, selected, decision = build_candidate_selection(
            blocks,
            effects,
            maximum_cluster_distance=0.05,
            target_candidates_per_network=3,
            minimum_positive_seed_blocks=2,
            reserved_revised_final_test_seeds=[
                70001, 70002, 70003, 70004, 70005
            ],
        )
        by_id = ranking.set_index("condition_id")
        self.assertTrue(bool(by_id.loc["cand_a", "qualified"]))
        self.assertLess(
            float(by_id.loc["cand_a", "legacy_balance_absolute_suppression"]),
            0.0,
        )
        self.assertFalse(bool(by_id.loc["cand_d", "qualified"]))
        self.assertEqual(set(selected["condition_id"]), {"cand_a", "cand_c", "cand_d"})
        selected_by_id = selected.set_index("condition_id")
        self.assertEqual(
            selected_by_id.loc["cand_d", "selection_role"],
            "exploratory_fallback",
        )
        self.assertEqual(len(clusters), 3)
        network = decision["network_decisions"][0]
        self.assertEqual(
            network["selection_mode"],
            "qualified_candidates_with_exploratory_fill",
        )
        self.assertEqual(network["qualified_selected_count"], 2)
        self.assertFalse(decision["revised_final_test_seeds_used"])


if __name__ == "__main__":
    unittest.main()
