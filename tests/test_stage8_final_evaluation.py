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
import run_stage8_final_evaluation as stage8_module
from analysis.final_evaluation_analysis import (
    build_final_candidate_effects,
    build_final_candidate_results,
    build_final_condition_summary,
    build_final_decision,
    build_network_conclusions,
    build_validation_test_comparison,
    candidate_metadata,
    load_final_evaluation,
)
from experiment_runtime import (
    ExperimentConfigurationError,
    SimulationRunResult,
)
from run_stage8_final_evaluation import (
    FinalEvaluationRunSpec,
    build_specs,
    command_for_spec,
    load_protocol,
    parse_args,
    run_final_evaluation,
)


EXPERIMENT_ID = "20260824_150000_final_evaluation_v01"


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


class Stage8ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.specs = build_specs(cls.protocol)

    def test_protocol_builds_frozen_candidates_and_comparators(self):
        self.assertEqual(len(self.specs), 90)
        self.assertEqual(len({spec.key for spec in self.specs}), 90)
        self.assertEqual(
            Counter(spec.network for spec in self.specs),
            Counter({"ba1000": 30, "facebook": 30, "wiki_vote": 30}),
        )
        self.assertEqual(
            Counter(spec.condition_role for spec in self.specs),
            Counter(
                {
                    "final_candidate": 45,
                    "no_intervention_baseline": 15,
                    "spring_research_reference": 15,
                    "prior_study_package_benchmark": 15,
                }
            ),
        )
        self.assertTrue(all(spec.iterations == 100 for spec in self.specs))
        self.assertTrue(all(spec.raw_level == "pop" for spec in self.specs))

    def test_candidate_set_and_reporting_roles_are_frozen(self):
        candidates = [
            spec
            for spec in self.specs
            if spec.condition_role == "final_candidate"
            and spec.simulator_seed == 50001
        ]
        self.assertEqual(len(candidates), 9)
        observed = {
            (spec.network, spec.condition_id): (
                spec.certainty,
                spec.effectiveness,
                spec.selection_order,
                spec.reporting_role,
                spec.qualified,
            )
            for spec in candidates
        }
        self.assertEqual(
            observed[("ba1000", "cand_cma_es_r03")],
            (0.7947, 0.8157, 1, "primary_candidate", True),
        )
        self.assertEqual(
            observed[("facebook", "cand_bo_gp_r06")],
            (0.7109, 0.8575, 1, "primary_exploratory_fallback", False),
        )
        self.assertEqual(
            observed[("wiki_vote", "cand_cma_es_r04")],
            (0.7442, 0.7482, 1, "primary_candidate", True),
        )
        for network in ("ba1000", "facebook", "wiki_vote"):
            network_rows = [row for row in candidates if row.network == network]
            self.assertEqual(
                {row.selection_order for row in network_rows}, {1, 2, 3}
            )
            self.assertEqual(len({row.region_id for row in network_rows}), 3)

    def test_commands_distinguish_all_intervention_modes(self):
        none = next(spec for spec in self.specs if spec.condition_id == "none")
        prior = next(
            spec for spec in self.specs if spec.condition_id == "prior_high"
        )
        candidate = next(
            spec
            for spec in self.specs
            if spec.network == "ba1000"
            and spec.condition_id == "cand_cma_es_r03"
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
            "0.7947",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--effectiveness") + 1],
            "0.8157",
        )

    def test_final_test_seeds_are_disjoint_from_all_previous_groups(self):
        used = {spec.simulator_seed for spec in self.specs}
        policy = self.protocol["seed_policy"]
        previous = set(policy["development_and_fixed_confirmation"])
        previous |= set(policy["stage6_exploration"])
        previous |= set(policy["stage7_candidate_validation"])
        self.assertEqual(used, {50001, 50002, 50003, 50004, 50005})
        self.assertTrue(used.isdisjoint(previous))

    def test_changed_candidate_source_hash_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["candidate_source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ExperimentConfigurationError, "candidate source SHA-256"
        ):
            build_specs(protocol)

    def test_dry_run_reports_all_90_commands(self):
        args = parse_args(["--experiment-id", EXPERIMENT_ID, "--dry-run"])
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_final_evaluation(args)
        data = json.loads(output.getvalue())
        self.assertIsNone(result)
        self.assertEqual(data["stage"], "stage8_final_evaluation")
        self.assertEqual(data["full_run_count"], 90)
        self.assertEqual(data["selected_run_count"], 90)

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
            with patch.object(stage8_module, "RUST_BINARY", fake_binary), patch.object(
                stage8_module,
                "git_state",
                return_value={"commit": "test", "dirty": True},
            ):
                with self.assertRaisesRegex(
                    ExperimentConfigurationError, "clean Git worktree"
                ):
                    run_final_evaluation(args)


class Stage8AnalysisTests(unittest.TestCase):
    def _create_small_run(self, temp_dir: str) -> tuple[Path, FinalEvaluationRunSpec]:
        with patch.object(
            fixed_module,
            "run_simulator",
            side_effect=fake_successful_simulator,
        ):
            args = fixed_module.parse_args(
                [
                    "--stage",
                    "stage8_final_evaluation",
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
                    "50001",
                    "--iterations",
                    "2",
                    "--raw-level",
                    "pop",
                    "--output-root",
                    temp_dir,
                ]
            )
            fixed_module.run_fixed_condition(args)
        root = Path(temp_dir) / "stage8_final_evaluation" / EXPERIMENT_ID
        spec = FinalEvaluationRunSpec(
            key="ba1000:cand_cma_es_r03:simseed50001",
            relative_run_dir="ba1000/cand_cma_es_r03/simseed_50001",
            network="ba1000",
            condition_id="cand_cma_es_r03",
            condition_role="final_candidate",
            intervention_enabled=True,
            certainty=0.7947,
            effectiveness=0.8157,
            opinion_csv=None,
            opinion_sha256=None,
            simulator_seed=50001,
            iterations=2,
            raw_level="pop",
            candidate_source="ba1000:cma_es:optseed3",
            source_method="cma_es",
            source_optimizer_replicate=3,
            source_optimizer_seed=None,
            source_simulator_seed=None,
            source_final_best=0.2667,
            selection_order=1,
            reporting_role="primary_candidate",
            selection_mode="qualified_candidates_selected",
            qualified=True,
            region_id="region_07",
            stage7_validation_mean_jcum=0.2735,
            stage7_validation_between_seed_sd=0.0038,
            stage7_none_relative_suppression=0.03,
            stage7_legacy_balance_relative_suppression=0.01,
            stage7_prior_high_relative_suppression=-0.79,
        )
        return root, spec

    def test_loader_recalculates_pop_only_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = self._create_small_run(temp_dir)
            data = load_final_evaluation(
                str(root), expected_specs=[spec]
            )
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
        rows = []
        roles = {
            "none": "no_intervention_baseline",
            "legacy_balance": "spring_research_reference",
            "prior_high": "prior_study_package_benchmark",
        }
        base = {"none": 0.30, "legacy_balance": 0.28, "prior_high": 0.15}
        for network, network_candidates in metadata.groupby("network", sort=True):
            conditions = []
            for _, candidate in network_candidates.iterrows():
                conditions.append(candidate.to_dict())
            for reference, value in base.items():
                conditions.append(
                    {
                        "condition_id": reference,
                        "condition_role": roles[reference],
                        "certainty": None if reference == "none" else 0.8,
                        "effectiveness": None if reference == "none" else 0.8,
                        "candidate_source": None,
                        "source_method": None,
                        "source_optimizer_replicate": None,
                        "source_final_best": None,
                        "fixed_value": value,
                    }
                )
            for condition in conditions:
                candidate_order = condition.get("selection_order")
                value = condition.get("fixed_value")
                if value is None:
                    value = 0.24 + 0.01 * int(candidate_order)
                for seed_offset, simulator_seed in enumerate((50001, 50002, 50003)):
                    for num_iter, jitter in enumerate((-0.001, 0.001, 0.0)):
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
                                "candidate_source": condition.get("candidate_source"),
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

    def test_final_analysis_reports_all_candidates_without_reselection(self):
        protocol = load_protocol()
        metadata = candidate_metadata(protocol)
        iterations = self._synthetic_iterations()
        condition_summary = build_final_condition_summary(
            iterations, repetitions=200, seed=10
        )
        none = build_final_candidate_effects(
            iterations, reference_id="none", repetitions=200, seed=20
        )
        legacy = build_final_candidate_effects(
            iterations,
            reference_id="legacy_balance",
            repetitions=200,
            seed=30,
        )
        prior = build_final_candidate_effects(
            iterations, reference_id="prior_high", repetitions=200, seed=40
        )
        final = build_final_candidate_results(
            condition_summary, metadata, none, legacy, prior
        )
        comparison = build_validation_test_comparison(final)
        conclusions = build_network_conclusions(final)
        decision = build_final_decision(
            final,
            conclusions,
            final_test_seeds=[50001, 50002, 50003],
        )

        self.assertEqual(len(final), 9)
        self.assertEqual(len(comparison), 9)
        self.assertEqual(len(conclusions), 3)
        self.assertTrue((comparison["used_for_reselection"] == False).all())
        self.assertTrue(final["none_interpretation"].eq("reduction").all())
        self.assertEqual(decision["status"], "final_evaluation_complete")
        self.assertFalse(decision["candidates_reselected"])
        self.assertFalse(decision["test_results_used_for_candidate_selection"])
        facebook = next(
            row for row in decision["networks"] if row["network"] == "facebook"
        )
        self.assertEqual(
            facebook["evidence_scope"],
            "exploratory_fallback_no_stage7_qualified_candidate",
        )
        self.assertEqual(len(facebook["frozen_candidates"]), 3)


if __name__ == "__main__":
    unittest.main()
