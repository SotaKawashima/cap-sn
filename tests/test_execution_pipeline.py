from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import optimize_single_objective as optimization_module
import run_fixed_condition as fixed_module
from analysis.optimization_metrics import compute_selfish_metrics_from_arrow
from experiment_runtime import (
    SimulationExecutionError,
    SimulationRunResult,
    sha256_file,
)


EXPERIMENT_ID = "20260810_000000_unit_test_v01"


def fake_successful_simulator(**kwargs) -> SimulationRunResult:
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pop_path = output_dir / "pop.arrow"
    info_path = output_dir / "info.arrow"
    agent_path = output_dir / "agent.arrow"
    stdout_path = Path(kwargs["stdout_path"])
    stderr_path = Path(kwargs["stderr_path"])

    pd.DataFrame(
        [
            (1, 1, 0),
            (0, 1, 5),
            (1, 0, 0),
            (0, 0, 0),
        ],
        columns=["num_iter", "t", "num_selfish"],
    ).to_feather(pop_path)
    pd.DataFrame(
        [], columns=["num_iter", "t", "information_idx"]
    ).to_feather(info_path)
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


class ExecutionPipelineTests(unittest.TestCase):
    def test_fixed_runner_preserves_existing_opinion_csv_and_manifest_hashes(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "v2"
            / "test_2"
            / "strategy"
            / "inhibition_opinion"
            / "98_98.csv"
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            fixed_module, "run_simulator", side_effect=fake_successful_simulator
        ):
            args = fixed_module.parse_args(
                [
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network",
                    "ba1000",
                    "--condition-id",
                    "prior_high",
                    "--intervention-opinion-csv",
                    str(source),
                    "--simulator-seed",
                    "1",
                    "--iterations",
                    "2",
                    "--output-root",
                    temp_dir,
                ]
            )
            run_dir = fixed_module.run_fixed_condition(args)

            with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            intervention = manifest["intervention"]
            copied = run_dir / "inhibition_opinion.csv"

            self.assertEqual(intervention["opinion_mode"], "existing_csv")
            self.assertEqual(
                intervention["opinion_source"]["sha256"], sha256_file(source)
            )
            self.assertEqual(
                intervention["opinion_csv"]["sha256"], sha256_file(source)
            )
            self.assertEqual(sha256_file(copied), sha256_file(source))
            self.assertEqual(intervention["opinion_values"]["phi_a0"], 0.02)
            self.assertEqual(
                intervention["applied_parameters"],
                {"certainty": 1.0, "effectiveness": 0.98},
            )

    def test_fixed_runner_online_value_matches_offline_recalculation(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            fixed_module, "run_simulator", side_effect=fake_successful_simulator
        ):
            args = fixed_module.parse_args(
                [
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network",
                    "ba1000",
                    "--condition-id",
                    "unit_condition",
                    "--certainty",
                    "0.8",
                    "--effectiveness",
                    "0.9",
                    "--simulator-seed",
                    "1",
                    "--iterations",
                    "2",
                    "--output-root",
                    temp_dir,
                ]
            )
            run_dir = fixed_module.run_fixed_condition(args)

            with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            offline = compute_selfish_metrics_from_arrow(
                run_dir / "pop.arrow",
                num_agents=1000,
                expected_iterations=2,
            )

            self.assertEqual(manifest["status"], "completed")
            self.assertAlmostEqual(
                manifest["objective"]["value"], offline.objective_value
            )
            self.assertAlmostEqual(offline.objective_value, 0.0025)
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "runtime.toml").exists())

    def test_optimization_runner_uses_shared_objective(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            optimization_module,
            "run_simulator",
            side_effect=fake_successful_simulator,
        ):
            args = optimization_module.parse_args(
                [
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network",
                    "ba1000",
                    "--method",
                    "random_search",
                    "--optimizer-replicate",
                    "1",
                    "--optimizer-seed",
                    "11",
                    "--simulator-seed",
                    "12",
                    "--iterations",
                    "2",
                    "--trials",
                    "2",
                    "--raw-level",
                    "all",
                    "--output-root",
                    temp_dir,
                ]
            )
            run_dir = optimization_module.run_optimization(args)
            trials = pd.read_csv(run_dir / "trials.csv")

            self.assertEqual(trials["state"].tolist(), ["COMPLETE", "COMPLETE"])
            self.assertTrue((trials["value"] == 0.0025).all())
            self.assertTrue(
                (trials["cumulative_selfish_fraction"] == trials["value"]).all()
            )
            self.assertTrue((run_dir / "study.db").exists())
            self.assertTrue((run_dir / "summary.json").exists())

    def test_failed_simulation_is_optuna_fail_not_objective_one(self):
        def fail_simulator(**kwargs):
            raise SimulationExecutionError(
                "intentional test failure", stage="simulation", exit_code=7
            )

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            optimization_module, "run_simulator", side_effect=fail_simulator
        ):
            args = optimization_module.parse_args(
                [
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network",
                    "ba1000",
                    "--method",
                    "random_search",
                    "--optimizer-replicate",
                    "1",
                    "--optimizer-seed",
                    "21",
                    "--simulator-seed",
                    "22",
                    "--iterations",
                    "2",
                    "--trials",
                    "1",
                    "--output-root",
                    temp_dir,
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "no complete trials"):
                optimization_module.run_optimization(args)

            run_dir = (
                Path(temp_dir)
                / "stage1_metric_validation"
                / EXPERIMENT_ID
                / "ba1000"
                / "random_search"
                / "optseed_1"
            )
            trials = pd.read_csv(run_dir / "trials.csv")
            with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(trials.loc[0, "state"], "FAIL")
            self.assertTrue(pd.isna(trials.loc[0, "value"]))
            self.assertNotEqual(trials.loc[0, "value"], 1.0)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["counts"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
