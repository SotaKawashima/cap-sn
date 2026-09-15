from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_fixed_condition as fixed_module
import run_stage8_final_retest as retest_module
from analysis.final_evaluation_analysis import build_final_condition_summary
from analysis.final_retest_analysis import (
    REFERENCE_IDS,
    build_candidate_seed_performance,
    build_final_candidate_effects,
    build_final_candidate_results,
    build_final_decision,
    build_network_conclusions,
    build_validation_retest_comparison,
    candidate_metadata,
    load_final_retest,
)
from experiment_runtime import (
    ExperimentConfigurationError,
    SimulationRunResult,
)
from run_stage8_final_retest import (
    build_specs,
    command_for_spec,
    load_protocol,
    parse_args,
    run_final_retest,
)


EXPERIMENT_ID = "20260915_190000_final_retest_v01"


def fake_successful_simulator(**kwargs) -> SimulationRunResult:
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pop_path = output_dir / "pop.arrow"
    info_path = output_dir / "info.arrow"
    agent_path = output_dir / "agent.arrow"
    stdout_path = Path(kwargs["stdout_path"])
    stderr_path = Path(kwargs["stderr_path"])
    pd.DataFrame(
        [(0, 1, 5), (0, 2, 0), (1, 1, 0), (1, 2, 0)],
        columns=["num_iter", "t", "num_selfish"],
    ).to_feather(pop_path)
    pd.DataFrame([], columns=["num_iter", "t", "information_idx"]).to_feather(
        info_path
    )
    pd.DataFrame([], columns=["num_iter", "agent_idx"]).to_feather(agent_path)
    stdout_path.write_text("fake simulator output\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return SimulationRunResult(
        command=["fake-simulator"],
        elapsed_sec=0.01,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        arrow_paths={"pop": pop_path, "info": info_path, "agent": agent_path},
    )


class Stage8FinalRetestProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.specs = build_specs(cls.protocol)

    def test_protocol_builds_all_candidates_and_four_references(self):
        self.assertEqual(len(self.specs), 105)
        self.assertEqual(len({spec.key for spec in self.specs}), 105)
        self.assertEqual(
            Counter(spec.network for spec in self.specs),
            Counter({"ba1000": 35, "facebook": 35, "wiki_vote": 35}),
        )
        self.assertEqual(
            Counter(spec.condition_role for spec in self.specs),
            Counter(
                {
                    "final_candidate": 45,
                    "no_intervention_baseline": 15,
                    "simple_design_variable_maximum_reference": 15,
                    "spring_research_reference": 15,
                    "prior_study_package_benchmark": 15,
                }
            ),
        )
        self.assertTrue(all(spec.iterations == 100 for spec in self.specs))
        self.assertTrue(all(spec.raw_level == "pop" for spec in self.specs))

    def test_amended_candidate_set_and_individual_roles_are_frozen(self):
        candidates = [
            spec
            for spec in self.specs
            if spec.condition_role == "final_candidate"
            and spec.simulator_seed == 70001
        ]
        observed = {
            (spec.network, spec.condition_id): (
                spec.certainty,
                spec.effectiveness,
                spec.selection_order,
                spec.selection_role,
                spec.qualified,
            )
            for spec in candidates
        }
        expected = {
            ("ba1000", "cand_cma_es_r03"): (
                0.7947,
                0.8157,
                1,
                "qualified_candidate",
                True,
            ),
            ("ba1000", "cand_bo_gp_r02"): (
                0.5034,
                0.9927,
                2,
                "qualified_candidate",
                True,
            ),
            ("ba1000", "cand_bo_gp_r04"): (
                0.6969,
                0.9928,
                3,
                "exploratory_fallback",
                False,
            ),
            ("facebook", "cand_bo_gp_r06"): (
                0.7109,
                0.8575,
                1,
                "exploratory_fallback",
                False,
            ),
            ("facebook", "cand_bo_gp_r01"): (
                0.6614,
                0.8815,
                2,
                "exploratory_fallback",
                False,
            ),
            ("facebook", "cand_bo_gp_r03"): (
                0.6211,
                0.5428,
                3,
                "exploratory_fallback",
                False,
            ),
            ("wiki_vote", "cand_cma_es_r04"): (
                0.7442,
                0.7482,
                1,
                "qualified_candidate",
                True,
            ),
            ("wiki_vote", "cand_random_search_r02"): (
                0.5542,
                0.7779,
                2,
                "qualified_candidate",
                True,
            ),
            ("wiki_vote", "cand_random_search_r05"): (
                0.8804,
                0.5085,
                3,
                "qualified_candidate",
                True,
            ),
        }
        self.assertEqual(observed, expected)

    def test_commands_distinguish_all_intervention_modes(self):
        by_condition = {
            spec.condition_id: spec
            for spec in self.specs
            if spec.network == "ba1000" and spec.simulator_seed == 70001
        }
        commands = {
            condition: command_for_spec(spec, experiment_id=EXPERIMENT_ID)
            for condition, spec in by_condition.items()
        }
        self.assertIn("--no-intervention", commands["none"])
        self.assertNotIn("--certainty", commands["none"])
        self.assertIn("--intervention-opinion-csv", commands["prior_high"])
        self.assertNotIn("--certainty", commands["prior_high"])
        for condition, expected in {
            "simple_max": ("1.0", "1.0"),
            "legacy_balance": ("0.8", "0.8"),
            "cand_cma_es_r03": ("0.7947", "0.8157"),
        }.items():
            command = commands[condition]
            self.assertNotIn("--intervention-opinion-csv", command)
            self.assertEqual(
                command[command.index("--certainty") + 1], expected[0]
            )
            self.assertEqual(
                command[command.index("--effectiveness") + 1], expected[1]
            )

    def test_revised_test_seeds_are_new_and_disjoint(self):
        used = {spec.simulator_seed for spec in self.specs}
        policy = self.protocol["seed_policy"]
        previous = set()
        for name, values in policy.items():
            excluded = {"revised_final_test", "disjointness_required"}
            if name not in excluded and isinstance(values, list):
                previous.update(values)
        self.assertEqual(used, {70001, 70002, 70003, 70004, 70005})
        self.assertTrue(used.isdisjoint(previous))

    def test_changed_candidate_source_hash_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["candidate_source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ExperimentConfigurationError, "candidate source SHA-256"
        ):
            build_specs(protocol)

    def test_dry_run_reports_all_105_commands(self):
        args = parse_args(["--experiment-id", EXPERIMENT_ID, "--dry-run"])
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_final_retest(args)
        data = json.loads(output.getvalue())
        self.assertIsNone(result)
        self.assertEqual(data["stage"], "stage8_final_retest")
        self.assertEqual(data["full_run_count"], 105)
        self.assertEqual(data["selected_run_count"], 105)

    def test_real_execution_requires_clean_worktree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_binary = Path(temp_dir) / "v2"
            fake_binary.touch()
            args = parse_args(
                [
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--output-root",
                    temp_dir,
                ]
            )
            with patch.object(
                retest_module, "RUST_BINARY", fake_binary
            ), patch.object(
                retest_module,
                "git_state",
                return_value={"commit": "test", "dirty": True},
            ):
                with self.assertRaisesRegex(
                    ExperimentConfigurationError, "clean Git worktree"
                ):
                    run_final_retest(args)


class Stage8FinalRetestAnalysisTests(unittest.TestCase):
    def _create_small_run(self, temp_dir: str):
        base_spec = next(
            spec
            for spec in build_specs(load_protocol())
            if spec.network == "ba1000"
            and spec.condition_id == "cand_cma_es_r03"
            and spec.simulator_seed == 70001
        )
        spec = replace(base_spec, iterations=2)
        with patch.object(
            fixed_module,
            "run_simulator",
            side_effect=fake_successful_simulator,
        ):
            args = fixed_module.parse_args(
                [
                    "--stage",
                    "stage8_final_retest",
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network",
                    "ba1000",
                    "--condition-id",
                    "cand_cma_es_r03",
                    "--certainty",
                    "0.7947",
                    "--effectiveness",
                    "0.8157",
                    "--simulator-seed",
                    "70001",
                    "--iterations",
                    "2",
                    "--raw-level",
                    "pop",
                    "--output-root",
                    temp_dir,
                ]
            )
            fixed_module.run_fixed_condition(args)
        root = Path(temp_dir) / "stage8_final_retest" / EXPERIMENT_ID
        return root, spec

    def test_loader_recalculates_pop_only_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = self._create_small_run(temp_dir)
            data = load_final_retest(str(root), expected_specs=[spec])
            self.assertEqual(len(data.runs), 1)
            self.assertEqual(len(data.iterations), 2)
            self.assertTrue(data.audit["valid"].all())
            self.assertAlmostEqual(
                float(data.runs.iloc[0]["cumulative_selfish_fraction"]),
                0.0025,
            )
            run_dir = root / spec.relative_run_dir
            self.assertTrue((run_dir / "pop.arrow").is_file())
            self.assertFalse((run_dir / "info.arrow").exists())
            self.assertFalse((run_dir / "agent.arrow").exists())

    def _synthetic_iterations(self) -> pd.DataFrame:
        metadata = candidate_metadata(load_protocol())
        roles = {
            "none": "no_intervention_baseline",
            "simple_max": "simple_design_variable_maximum_reference",
            "legacy_balance": "spring_research_reference",
            "prior_high": "prior_study_package_benchmark",
        }
        reference_values = {
            "none": 0.300,
            "simple_max": 0.265,
            "legacy_balance": 0.280,
            "prior_high": 0.150,
        }
        rows = []
        for network, network_candidates in metadata.groupby("network", sort=True):
            conditions = [row.to_dict() for _, row in network_candidates.iterrows()]
            for reference, value in reference_values.items():
                conditions.append(
                    {
                        "condition_id": reference,
                        "condition_role": roles[reference],
                        "certainty": None if reference == "none" else 1.0,
                        "effectiveness": None if reference == "none" else 1.0,
                        "candidate_source": None,
                        "source_method": None,
                        "source_optimizer_replicate": None,
                        "source_final_best": None,
                        "fixed_value": value,
                    }
                )
            for condition in conditions:
                selection_order = condition.get("selection_order")
                value = condition.get("fixed_value")
                if value is None:
                    value = 0.240 + 0.010 * int(selection_order)
                for seed_offset, simulator_seed in enumerate(
                    (70001, 70002, 70003)
                ):
                    for num_iter, jitter in enumerate(
                        (-0.0015, -0.0005, 0.0005, 0.0015)
                    ):
                        selfish = float(value) + jitter + seed_offset * 0.0005
                        rows.append(
                            {
                                "run_key": (
                                    f"{network}:{condition['condition_id']}:"
                                    f"simseed{simulator_seed}"
                                ),
                                "network": network,
                                "condition_id": condition["condition_id"],
                                "condition_role": condition.get(
                                    "condition_role", "final_candidate"
                                ),
                                "intervention_enabled": (
                                    condition["condition_id"] != "none"
                                ),
                                "certainty": condition.get("certainty"),
                                "effectiveness": condition.get("effectiveness"),
                                "candidate_source": condition.get(
                                    "candidate_source"
                                ),
                                "source_method": condition.get("source_method"),
                                "source_optimizer_replicate": condition.get(
                                    "source_optimizer_replicate"
                                ),
                                "source_final_best": condition.get(
                                    "source_final_best"
                                ),
                                "simulator_seed": simulator_seed,
                                "num_agents": 1000,
                                "num_iter": num_iter,
                                "n_steps": 10,
                                "cumulative_selfish_count": int(selfish * 1000),
                                "cumulative_selfish_fraction": selfish,
                                "peak_new_selfish_ratio": selfish / 3,
                                "mean_new_selfish_rate_per_step": selfish / 10,
                                "first_selfish_step": 1,
                                "last_selfish_step": 9,
                                "t50_selfish_step": 4,
                                "t90_selfish_step": 8,
                                "selfish_timing_centroid": 4.5,
                                "selfish_span_steps": 9,
                                "active_selfish_steps": 6,
                            }
                        )
        return pd.DataFrame(rows)

    def test_analysis_reports_four_references_without_reselection(self):
        protocol = load_protocol()
        metadata = candidate_metadata(protocol)
        iterations = self._synthetic_iterations()
        condition_summary = build_final_condition_summary(
            iterations, repetitions=200, seed=10
        )
        candidate_seed = build_candidate_seed_performance(iterations, metadata)
        effects = {
            reference: build_final_candidate_effects(
                iterations,
                reference_id=reference,
                repetitions=200,
                seed=20 + index * 10,
            )
            for index, reference in enumerate(REFERENCE_IDS)
        }
        final = build_final_candidate_results(
            condition_summary, metadata, effects
        )
        comparison = build_validation_retest_comparison(final)
        conclusions = build_network_conclusions(final)
        decision = build_final_decision(
            final,
            conclusions,
            final_test_seeds=[70001, 70002, 70003],
            inspected_original_final_test_seeds=[50001, 50002, 50003],
        )

        self.assertEqual(len(candidate_seed), 27)
        self.assertEqual(len(final), 9)
        self.assertEqual(len(comparison), 9)
        self.assertEqual(len(conclusions), 3)
        self.assertTrue((comparison["used_for_reselection"] == False).all())
        self.assertTrue(final["none_interpretation"].eq("reduction").all())
        self.assertIn("simple_max_relative_suppression", final)
        self.assertIn("prior_high_relative_suppression", final)
        self.assertEqual(decision["status"], "revised_final_evaluation_complete")
        self.assertFalse(decision["candidates_reranked"])
        self.assertFalse(decision["candidates_reselected"])
        self.assertFalse(decision["original_final_test_seeds_reused"])

        ba1000 = next(
            row for row in decision["networks"] if row["network"] == "ba1000"
        )
        self.assertEqual(
            [row["selection_role"] for row in ba1000["frozen_candidates"]],
            [
                "qualified_candidate",
                "qualified_candidate",
                "exploratory_fallback",
            ],
        )
        self.assertEqual(
            set(ba1000["frozen_candidates"][0]["effects"]),
            set(REFERENCE_IDS),
        )


if __name__ == "__main__":
    unittest.main()
