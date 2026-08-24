from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_fixed_condition as fixed_module
import run_stage7_candidate_validation as stage7_module
from analysis.candidate_validation_analysis import (
    build_candidate_block_performance,
    build_candidate_clusters,
    build_candidate_effects,
    build_candidate_selection,
    load_candidate_validation,
)
from experiment_runtime import (
    ExperimentConfigurationError,
    SimulationRunResult,
)
from run_stage7_candidate_validation import (
    CandidateValidationRunSpec,
    build_specs,
    command_for_spec,
    load_protocol,
    parse_args,
    run_candidate_validation,
)


EXPERIMENT_ID = "20260822_120000_candidate_validation_v01"


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


class Stage7ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.specs = build_specs(cls.protocol)

    def test_protocol_builds_all_candidates_and_comparators(self):
        self.assertEqual(len(self.specs), 189)
        self.assertEqual(len({spec.key for spec in self.specs}), 189)
        self.assertEqual(
            Counter(spec.network for spec in self.specs),
            Counter({"ba1000": 63, "facebook": 63, "wiki_vote": 63}),
        )
        self.assertEqual(
            Counter(spec.condition_role for spec in self.specs),
            Counter(
                {
                    "stage6_candidate": 162,
                    "no_intervention_baseline": 9,
                    "spring_research_reference": 9,
                    "prior_study_package_benchmark": 9,
                }
            ),
        )
        self.assertEqual(
            Counter(spec.simulator_seed for spec in self.specs),
            Counter({40001: 63, 40002: 63, 40003: 63}),
        )
        self.assertTrue(all(spec.iterations == 100 for spec in self.specs))
        self.assertTrue(all(spec.raw_level == "pop" for spec in self.specs))

    def test_all_54_stage6_candidates_are_preserved(self):
        candidates = [
            spec
            for spec in self.specs
            if spec.condition_role == "stage6_candidate"
            and spec.simulator_seed == 40001
        ]
        self.assertEqual(len(candidates), 54)
        self.assertEqual(
            Counter(spec.network for spec in candidates),
            Counter({"ba1000": 18, "facebook": 18, "wiki_vote": 18}),
        )
        self.assertTrue(all(spec.candidate_source for spec in candidates))
        self.assertTrue(all(spec.source_simulator_seed == 30001 for spec in candidates))

    def test_commands_distinguish_all_three_comparator_modes(self):
        none = next(spec for spec in self.specs if spec.condition_id == "none")
        prior = next(spec for spec in self.specs if spec.condition_id == "prior_high")
        candidate = next(
            spec
            for spec in self.specs
            if spec.network == "ba1000" and spec.condition_id == "cand_bo_gp_r04"
        )
        none_command = command_for_spec(none, experiment_id=EXPERIMENT_ID)
        prior_command = command_for_spec(prior, experiment_id=EXPERIMENT_ID)
        candidate_command = command_for_spec(candidate, experiment_id=EXPERIMENT_ID)
        self.assertIn("--no-intervention", none_command)
        self.assertNotIn("--certainty", none_command)
        self.assertIn("--intervention-opinion-csv", prior_command)
        self.assertNotIn("--certainty", prior_command)
        self.assertEqual(
            candidate_command[candidate_command.index("--certainty") + 1],
            "0.6969",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--effectiveness") + 1],
            "0.9928",
        )

    def test_final_test_seeds_are_disjoint_and_never_executed(self):
        used = {spec.simulator_seed for spec in self.specs}
        final = set(self.protocol["seed_policy"]["reserved_final_test"])
        self.assertTrue(used.isdisjoint(final))
        self.assertEqual(final, {50001, 50002, 50003, 50004, 50005})

    def test_changed_candidate_source_hash_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["candidate_source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ExperimentConfigurationError, "candidate source SHA-256"
        ):
            build_specs(protocol)

    def test_dry_run_reports_all_189_commands(self):
        args = parse_args(["--experiment-id", EXPERIMENT_ID, "--dry-run"])
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_candidate_validation(args)
        data = json.loads(output.getvalue())
        self.assertIsNone(result)
        self.assertEqual(data["stage"], "stage7_candidate_validation")
        self.assertEqual(data["full_run_count"], 189)
        self.assertEqual(data["selected_run_count"], 189)

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
            with patch.object(stage7_module, "RUST_BINARY", fake_binary), patch.object(
                stage7_module,
                "git_state",
                return_value={"commit": "test", "dirty": True},
            ):
                with self.assertRaisesRegex(
                    ExperimentConfigurationError, "clean Git worktree"
                ):
                    run_candidate_validation(args)


class Stage7AnalysisTests(unittest.TestCase):
    def _create_small_run(self, temp_dir: str) -> tuple[Path, CandidateValidationRunSpec]:
        with patch.object(
            fixed_module,
            "run_simulator",
            side_effect=fake_successful_simulator,
        ):
            args = fixed_module.parse_args(
                [
                    "--stage",
                    "stage7_candidate_validation",
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network",
                    "ba1000",
                    "--condition-id",
                    "cand_random_search_r01",
                    "--certainty",
                    "0.75",
                    "--effectiveness",
                    "0.9",
                    "--simulator-seed",
                    "40001",
                    "--iterations",
                    "2",
                    "--raw-level",
                    "pop",
                    "--output-root",
                    temp_dir,
                ]
            )
            fixed_module.run_fixed_condition(args)
        root = Path(temp_dir) / "stage7_candidate_validation" / EXPERIMENT_ID
        spec = CandidateValidationRunSpec(
            key="ba1000:cand_random_search_r01:simseed40001",
            relative_run_dir="ba1000/cand_random_search_r01/simseed_40001",
            network="ba1000",
            condition_id="cand_random_search_r01",
            condition_role="stage6_candidate",
            intervention_enabled=True,
            certainty=0.75,
            effectiveness=0.9,
            opinion_csv=None,
            opinion_sha256=None,
            simulator_seed=40001,
            iterations=2,
            raw_level="pop",
            candidate_source="ba1000:random_search:optseed1",
            source_method="random_search",
            source_optimizer_replicate=1,
            source_optimizer_seed=60103,
            source_simulator_seed=30001,
            source_final_best=0.27,
        )
        return root, spec

    def test_loader_recalculates_pop_only_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = self._create_small_run(temp_dir)
            data = load_candidate_validation(root, expected_specs=[spec])
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

    def test_complete_linkage_does_not_chain_distant_endpoints(self):
        metadata = pd.DataFrame(
            [
                ("ba1000", "a", 0.50, 0.50),
                ("ba1000", "b", 0.54, 0.50),
                ("ba1000", "c", 0.58, 0.50),
            ],
            columns=["network", "condition_id", "certainty", "effectiveness"],
        )
        result = build_candidate_clusters(metadata, maximum_distance=0.05)
        regions = result.set_index("condition_id")["region_id"]
        self.assertEqual(regions["a"], regions["b"])
        self.assertNotEqual(regions["a"], regions["c"])

    def test_selection_uses_robust_effects_and_one_representative_per_region(self):
        rows = []
        values = {
            "none": [0.30, 0.31, 0.29],
            "legacy_balance": [0.28, 0.29, 0.27],
            "prior_high": [0.15, 0.16, 0.14],
            "cand_a": [0.25, 0.26, 0.24],
            "cand_b": [0.251, 0.261, 0.241],
            "cand_c": [0.26, 0.27, 0.25],
            "cand_d": [0.31, 0.30, 0.32],
        }
        parameters = {
            "cand_a": (0.60, 0.90),
            "cand_b": (0.62, 0.90),
            "cand_c": (0.85, 0.55),
            "cand_d": (0.95, 0.95),
        }
        roles = {
            "none": "no_intervention_baseline",
            "legacy_balance": "spring_research_reference",
            "prior_high": "prior_study_package_benchmark",
        }
        for condition_id, block_values in values.items():
            is_candidate = condition_id.startswith("cand_")
            certainty, effectiveness = parameters.get(condition_id, (0.8, 0.8))
            if condition_id == "none":
                certainty, effectiveness = (None, None)
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
                                f"ba1000:bo_gp:{condition_id}" if is_candidate else None
                            ),
                            "source_method": "bo_gp" if is_candidate else None,
                            "source_optimizer_replicate": 1 if is_candidate else None,
                            "source_final_best": 0.20 if is_candidate else None,
                            "simulator_seed": seed,
                            "num_agents": 1000,
                            "num_iter": num_iter,
                            "cumulative_selfish_fraction": value + jitter,
                            "peak_new_selfish_ratio": 0.1,
                        }
                    )
        iterations = pd.DataFrame(rows)
        blocks = build_candidate_block_performance(iterations)
        effects_none = build_candidate_effects(
            iterations, reference_id="none", repetitions=200, seed=1
        )
        effects_legacy = build_candidate_effects(
            iterations, reference_id="legacy_balance", repetitions=200, seed=2
        )
        effects_prior = build_candidate_effects(
            iterations, reference_id="prior_high", repetitions=200, seed=3
        )
        ranking, clusters, selected, decision = build_candidate_selection(
            blocks,
            effects_none,
            effects_legacy,
            effects_prior,
            maximum_cluster_distance=0.05,
            maximum_regions_per_network=3,
            minimum_positive_seed_blocks=2,
        )
        self.assertEqual(set(selected["condition_id"]), {"cand_a", "cand_c"})
        self.assertTrue(selected["qualified"].all())
        self.assertEqual(len(clusters), 3)
        self.assertEqual(decision["status"], "candidate_selection_complete")
        self.assertFalse(decision["effect_claim_allowed"])
        self.assertFalse(decision["final_test_seeds_used"])
        rows_by_id = ranking.set_index("condition_id")
        self.assertTrue(bool(rows_by_id.loc["cand_a", "qualified"]))
        self.assertFalse(bool(rows_by_id.loc["cand_d", "qualified"]))


if __name__ == "__main__":
    unittest.main()
