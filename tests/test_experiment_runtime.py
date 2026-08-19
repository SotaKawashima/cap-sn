from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import toml

from experiment_runtime import (
    AGENT_CONFIG,
    NETWORKS,
    ExperimentConfigurationError,
    build_simulator_command,
    copy_intervention_opinion_csv,
    create_unique_run_directory,
    make_experiment_id,
    parameter_condition_id,
    read_intervention_opinion_csv,
    sha256_file,
    validate_experiment_id,
    write_intervention_opinion_csv,
    write_runtime_config,
    write_strategy_config,
)
from run_fixed_condition import parse_args, resolve_condition


class ExperimentRuntimeTests(unittest.TestCase):
    def test_experiment_id_matches_protocol(self):
        experiment_id = make_experiment_id("metric_smoke", version=2)

        self.assertRegex(
            experiment_id, r"^\d{8}_\d{6}_metric_smoke_v02$"
        )
        self.assertEqual(validate_experiment_id(experiment_id), experiment_id)

    def test_run_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            create_unique_run_directory(run_dir)

            with self.assertRaises(FileExistsError):
                create_unique_run_directory(run_dir)

    def test_runtime_config_records_seed_and_iteration_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_path = write_runtime_config(
                Path(temp_dir) / "runtime.toml",
                simulator_seed=0,
                iteration_count=7,
            )

            runtime = toml.load(runtime_path)
            self.assertEqual(runtime, {"seed_state": 0, "iteration_count": 7})

    def test_intervention_csv_rounds_applied_values_to_four_decimals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "opinion.csv"
            applied = write_intervention_opinion_csv(
                csv_path, certainty=0.54321, effectiveness=0.98765
            )
            frame = pd.read_csv(csv_path)

            self.assertEqual(applied, {"certainty": 0.5432, "effectiveness": 0.9877})
            self.assertAlmostEqual(frame.loc[0, "phi_b1"], 0.5432)
            self.assertAlmostEqual(frame.loc[0, "phi_u"], 0.4568)
            self.assertAlmostEqual(frame.loc[0, "psi1_b0"], 0.9877)
            self.assertAlmostEqual(frame.loc[0, "psi1_u"], 0.0123)

    def test_strategy_config_uses_absolute_paths_and_generated_opinion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            opinion_path = temp_path / "opinion.csv"
            write_intervention_opinion_csv(
                opinion_path, certainty=0.8, effectiveness=0.9
            )
            strategy_path = write_strategy_config(
                temp_path / "strategy.toml",
                intervention_opinion_csv=opinion_path,
            )
            strategy = toml.load(strategy_path)

            self.assertTrue(Path(strategy["informing"]).is_absolute())
            for path in strategy["information"].values():
                self.assertTrue(Path(path).is_absolute())
            self.assertEqual(
                Path(strategy["information"]["inhibition"]), opinion_path.resolve()
            )

    def test_existing_intervention_csv_is_copied_without_regeneration(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "v2"
            / "test_2"
            / "strategy"
            / "inhibition_opinion"
            / "98_98.csv"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "opinion.csv"
            values, applied = copy_intervention_opinion_csv(source, destination)

            self.assertEqual(sha256_file(destination), sha256_file(source))
            self.assertEqual(applied, {"certainty": 1.0, "effectiveness": 0.98})
            self.assertEqual(values["phi_a0"], 0.02)
            self.assertEqual(values["phi_a1"], 0.98)

    def test_existing_intervention_csv_resolves_design_parameters(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "v2"
            / "test_2"
            / "strategy"
            / "inhibition_opinion"
            / "98_98.csv"
        )
        args = parse_args(
            [
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
            ]
        )

        condition = resolve_condition(args)

        self.assertTrue(condition["enabled"])
        self.assertEqual(condition["condition_id"], "prior_high")
        self.assertEqual(
            condition["proposed_parameters"],
            {"certainty": 1.0, "effectiveness": 0.98},
        )
        self.assertEqual(condition["opinion_mode"], "existing_csv")

    def test_existing_intervention_csv_schema_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Path(temp_dir) / "invalid.csv"
            pd.DataFrame([{"phi_b1": 1.0}]).to_csv(invalid, index=False)

            with self.assertRaises(ExperimentConfigurationError):
                read_intervention_opinion_csv(invalid)

    def test_simulator_command_only_enables_intervention_when_requested(self):
        common = {
            "identifier": "simulation",
            "output_dir": Path("/tmp/output"),
            "runtime_path": Path("/tmp/runtime.toml"),
            "network_path": NETWORKS["ba1000"].config_path,
            "agent_path": AGENT_CONFIG,
            "strategy_path": Path("/tmp/strategy.toml"),
        }

        enabled = build_simulator_command(**common, intervention_enabled=True)
        disabled = build_simulator_command(**common, intervention_enabled=False)

        self.assertIn("-e", enabled)
        self.assertNotIn("-e", disabled)
        self.assertEqual(enabled[-2:], ["-d", "0"])
        self.assertEqual(disabled[-2:], ["-d", "0"])

    def test_no_intervention_has_null_parameters_and_condition_none(self):
        args = parse_args(
            [
                "--network",
                "ba1000",
                "--no-intervention",
                "--simulator-seed",
                "1",
                "--iterations",
                "2",
            ]
        )

        condition = resolve_condition(args)

        self.assertFalse(condition["enabled"])
        self.assertEqual(condition["condition_id"], "none")
        self.assertIsNone(condition["proposed_parameters"])

    def test_no_intervention_rejects_design_variables(self):
        args = parse_args(
            [
                "--network",
                "ba1000",
                "--no-intervention",
                "--certainty",
                "0.8",
                "--effectiveness",
                "0.8",
                "--simulator-seed",
                "1",
                "--iterations",
                "2",
            ]
        )

        with self.assertRaises(ExperimentConfigurationError):
            resolve_condition(args)

    def test_parameter_condition_id_uses_four_decimal_places(self):
        self.assertEqual(
            parameter_condition_id(0.5, 0.75), "c0p5000_e0p7500"
        )


if __name__ == "__main__":
    unittest.main()
