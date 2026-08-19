from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import optimize_single_objective as optimization_module
import run_stage6_reoptimization as stage6_module
from analysis.reoptimization_analysis import (
    build_best_so_far,
    build_candidate_pool,
    build_convergence_summary,
    build_method_summary,
    build_run_final_summary,
    build_stage6_decision,
    build_timing_summary,
    build_unique_candidate_pool,
    load_reoptimization,
)
from experiment_runtime import (
    ExperimentConfigurationError,
    SimulationRunResult,
)
from run_stage6_reoptimization import (
    ReoptimizationRunSpec,
    build_run_specs,
    command_for_spec,
    inspect_run_status,
    load_protocol,
    parse_args,
    run_reoptimization,
)


EXPERIMENT_ID = "20260819_120000_stage6_reoptimization_v01"


def fake_successful_simulator(**kwargs) -> SimulationRunResult:
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pop_path = output_dir / "pop.arrow"
    info_path = output_dir / "info.arrow"
    agent_path = output_dir / "agent.arrow"
    stdout_path = Path(kwargs["stdout_path"])
    stderr_path = Path(kwargs["stderr_path"])

    pd.DataFrame(
        [(1, 1, 0), (0, 1, 5), (1, 0, 0), (0, 0, 0)],
        columns=["num_iter", "t", "num_selfish"],
    ).to_feather(pop_path)
    pd.DataFrame([], columns=["num_iter", "t", "information_idx"]).to_feather(
        info_path
    )
    pd.DataFrame(
        [(0, 1, index, True) for index in range(5)],
        columns=["num_iter", "t", "agent_idx", "selfish"],
    ).to_feather(agent_path)
    stdout_path.write_text("fake simulator output\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return SimulationRunResult(
        command=["fake-simulator"],
        elapsed_sec=0.01,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        arrow_paths={"pop": pop_path, "info": info_path, "agent": agent_path},
    )


class Stage6ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.specs = build_run_specs(cls.protocol)

    def test_protocol_builds_the_frozen_54_run_design(self):
        self.assertEqual(len(self.specs), 54)
        self.assertEqual(len({spec.key for spec in self.specs}), 54)
        self.assertEqual(
            Counter(spec.network for spec in self.specs),
            Counter({"ba1000": 18, "facebook": 18, "wiki_vote": 18}),
        )
        self.assertEqual(
            Counter(spec.method for spec in self.specs),
            Counter({"bo_gp": 18, "cma_es": 18, "random_search": 18}),
        )
        self.assertEqual(len({spec.optimizer_seed for spec in self.specs}), 18)
        self.assertTrue(all(spec.iterations == 100 for spec in self.specs))
        self.assertTrue(all(spec.trials == 50 for spec in self.specs))
        self.assertTrue(all(spec.simulator_seed == 30001 for spec in self.specs))
        self.assertTrue(all(spec.raw_level == "pop" for spec in self.specs))

    def test_each_network_method_has_six_replicates(self):
        counts = Counter((spec.network, spec.method) for spec in self.specs)
        self.assertEqual(len(counts), 9)
        self.assertTrue(all(count == 6 for count in counts.values()))

    def test_child_command_contains_the_frozen_budget_and_seed(self):
        spec = next(spec for spec in self.specs if spec.key == "ba1000:bo_gp:optseed1")
        command = command_for_spec(spec, experiment_id=EXPERIMENT_ID)

        self.assertEqual(command[command.index("--stage") + 1], "stage6_reoptimization")
        self.assertEqual(command[command.index("--iterations") + 1], "100")
        self.assertEqual(command[command.index("--trials") + 1], "50")
        self.assertEqual(command[command.index("--simulator-seed") + 1], "30001")
        self.assertEqual(command[command.index("--raw-level") + 1], "pop")

    def test_dry_run_reports_all_54_commands(self):
        args = parse_args(["--experiment-id", EXPERIMENT_ID, "--dry-run"])
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_reoptimization(args)

        data = json.loads(output.getvalue())
        self.assertIsNone(result)
        self.assertEqual(data["stage"], "stage6_reoptimization")
        self.assertEqual(data["full_run_count"], 54)
        self.assertEqual(data["selected_run_count"], 54)
        self.assertEqual(len(data["commands"]), 54)

    def test_dry_run_filters_do_not_change_the_full_design(self):
        args = parse_args(
            [
                "--experiment-id",
                EXPERIMENT_ID,
                "--networks",
                "facebook",
                "--methods",
                "random_search",
                "--optimizer-replicates",
                "2",
                "--dry-run",
            ]
        )
        output = io.StringIO()
        with redirect_stdout(output):
            run_reoptimization(args)
        data = json.loads(output.getvalue())

        self.assertEqual(data["full_run_count"], 54)
        self.assertEqual(data["selected_run_count"], 1)

    def test_completed_status_requires_the_full_clean_budget(self):
        spec = self.specs[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "counts": {"complete": 49, "failed": 1, "pruned": 0},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                inspect_run_status(run_dir, spec),
                "invalid_completed_budget",
            )

    def test_real_execution_requires_a_clean_worktree(self):
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
            with patch.object(stage6_module, "RUST_BINARY", fake_binary), patch.object(
                stage6_module,
                "git_state",
                return_value={"commit": "test", "dirty": True},
            ):
                with self.assertRaisesRegex(
                    ExperimentConfigurationError, "clean Git worktree"
                ):
                    run_reoptimization(args)


class Stage6AnalysisTests(unittest.TestCase):
    def _create_small_run(self, temp_dir: str) -> tuple[Path, ReoptimizationRunSpec]:
        with patch.object(
            optimization_module,
            "run_simulator",
            side_effect=fake_successful_simulator,
        ):
            args = optimization_module.parse_args(
                [
                    "--stage",
                    "stage6_reoptimization",
                    "--purpose",
                    "stage6_reoptimization",
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network",
                    "ba1000",
                    "--method",
                    "random_search",
                    "--optimizer-replicate",
                    "1",
                    "--optimizer-seed",
                    "60103",
                    "--simulator-seed",
                    "30001",
                    "--iterations",
                    "2",
                    "--trials",
                    "2",
                    "--startup-trials",
                    "1",
                    "--raw-level",
                    "pop",
                    "--output-root",
                    temp_dir,
                ]
            )
            optimization_module.run_optimization(args)
        root = Path(temp_dir) / "stage6_reoptimization" / EXPERIMENT_ID
        spec = ReoptimizationRunSpec(
            key="ba1000:random_search:optseed1",
            relative_run_dir="ba1000/random_search/optseed_1",
            network="ba1000",
            method="random_search",
            optimizer_replicate=1,
            optimizer_seed=60103,
            simulator_seed=30001,
            iterations=2,
            trials=2,
            startup_trials=1,
            raw_level="pop",
        )
        return root, spec

    def test_formal_loader_recalculates_every_trial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = self._create_small_run(temp_dir)
            data = load_reoptimization(root, expected_specs=[spec])

            self.assertEqual(len(data.trials), 2)
            self.assertEqual(len(data.iterations), 4)
            self.assertEqual(len(data.runs), 1)
            self.assertTrue(data.audit["valid"].all())
            self.assertTrue((data.trials["value"] == 0.0025).all())
            trial_dir = root / spec.relative_run_dir / "raw" / "trial_0000"
            self.assertTrue((trial_dir / "pop.arrow").is_file())
            self.assertFalse((trial_dir / "info.arrow").exists())
            self.assertFalse((trial_dir / "agent.arrow").exists())

    def test_formal_loader_rejects_a_changed_objective(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = self._create_small_run(temp_dir)
            trials_path = root / spec.relative_run_dir / "trials.csv"
            trials = pd.read_csv(trials_path)
            trials.loc[0, "value"] = 0.99
            trials.to_csv(trials_path, index=False)

            with self.assertRaisesRegex(ValueError, "raw objective"):
                load_reoptimization(root, expected_specs=[spec])

    def test_aggregation_builds_a_run_candidate_without_claiming_effect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = self._create_small_run(temp_dir)
            data = load_reoptimization(root, expected_specs=[spec])
            best = build_best_so_far(data.trials)
            convergence = build_convergence_summary(best)
            run_final = build_run_final_summary(data.trials, data.runs)
            methods = build_method_summary(run_final)
            timing = build_timing_summary(data.runs)
            candidates = build_candidate_pool(run_final)
            unique = build_unique_candidate_pool(candidates)
            decision = build_stage6_decision(run_final, candidates, unique)

            self.assertEqual(len(best), 2)
            self.assertEqual(len(convergence), 2)
            self.assertEqual(len(run_final), 1)
            self.assertEqual(len(methods), 1)
            self.assertEqual(len(timing), 1)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(len(unique), 1)
            self.assertEqual(decision["status"], "candidate_pool_ready")
            self.assertFalse(decision["candidate_selection_complete"])
            self.assertFalse(decision["effect_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
