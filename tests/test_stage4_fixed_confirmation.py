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

import run_stage4_fixed_confirmation as stage4_module
from experiment_runtime import ExperimentConfigurationError
from run_stage4_fixed_confirmation import (
    build_specs,
    command_for_spec,
    load_protocol,
    parse_args,
    run_fixed_confirmation,
)


EXPERIMENT_ID = "20260817_120000_fixed_confirmation_v01"


class Stage4ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.specs = build_specs(cls.protocol)

    def test_protocol_builds_complete_fixed_design(self):
        self.assertEqual(len(self.specs), 210)
        self.assertEqual(
            Counter(spec.network for spec in self.specs),
            Counter({"ba1000": 70, "facebook": 70, "wiki_vote": 70}),
        )
        self.assertEqual(
            set(Counter(spec.condition_id for spec in self.specs).values()),
            {15},
        )
        self.assertEqual(
            Counter(spec.simulator_seed for spec in self.specs),
            Counter(
                {
                    20001: 42,
                    20002: 42,
                    20003: 42,
                    20004: 42,
                    20005: 42,
                }
            ),
        )
        self.assertTrue(all(spec.iterations == 100 for spec in self.specs))
        self.assertTrue(all(spec.raw_level == "all" for spec in self.specs))
        self.assertEqual(len({spec.key for spec in self.specs}), 210)

    def test_stage4_design_matches_final_stage3_plan(self):
        stage3_path = (
            Path(__file__).resolve().parents[1]
            / "experiment_protocols"
            / "stage3_pilot_v4.json"
        )
        with stage3_path.open(encoding="utf-8") as handle:
            stage3 = json.load(handle)

        source = stage3["finalized_experiment_plan"][
            "stage4_fixed_confirmation"
        ]
        design = self.protocol["design"]
        self.assertEqual(design["networks"], source["networks"])
        self.assertEqual(design["simulator_seeds"], source["simulator_seeds"])
        self.assertEqual(design["raw_level"], source["raw_level"])
        for stage4_condition, source_condition in zip(
            design["conditions"], source["conditions"], strict=True
        ):
            self.assertEqual(
                {
                    key: stage4_condition[key]
                    for key in (
                        "id",
                        "enabled",
                        "certainty",
                        "effectiveness",
                    )
                },
                source_condition,
            )

    def test_no_intervention_command_has_no_design_variables(self):
        spec = next(
            spec for spec in self.specs if spec.condition_id == "none"
        )
        command = command_for_spec(spec, experiment_id=EXPERIMENT_ID)

        self.assertIn("--no-intervention", command)
        self.assertNotIn("--certainty", command)
        self.assertNotIn("--effectiveness", command)
        self.assertEqual(command[command.index("--condition-id") + 1], "none")

    def test_intervention_command_has_fixed_parameters(self):
        spec = next(
            spec
            for spec in self.specs
            if spec.network == "ba1000"
            and spec.condition_id == "grid_c075_e100"
        )
        command = command_for_spec(spec, experiment_id=EXPERIMENT_ID)

        self.assertNotIn("--no-intervention", command)
        self.assertEqual(command[command.index("--certainty") + 1], "0.75")
        self.assertEqual(command[command.index("--effectiveness") + 1], "1.0")
        self.assertEqual(
            command[command.index("--stage") + 1],
            "stage4_fixed_confirmation",
        )

    def test_stage4_and_future_seed_groups_are_disjoint(self):
        stage4_seeds = set(self.protocol["design"]["simulator_seeds"])
        groups = [
            set(values)
            for values in self.protocol["seed_policy"][
                "reserved_future_groups"
            ].values()
        ]
        self.assertTrue(all(stage4_seeds.isdisjoint(group) for group in groups))
        for index, group in enumerate(groups):
            for other in groups[index + 1 :]:
                self.assertTrue(group.isdisjoint(other))

    def test_duplicate_intervention_parameters_are_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["design"]["conditions"][1]["certainty"] = 0.5
        protocol["design"]["conditions"][1]["effectiveness"] = 0.5

        with self.assertRaisesRegex(
            ExperimentConfigurationError, "duplicate Stage 4 intervention"
        ):
            build_specs(protocol)

    def test_reserved_seed_overlap_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["seed_policy"]["reserved_future_groups"][
            "candidate_validation"
        ][0] = 20001

        with self.assertRaisesRegex(
            ExperimentConfigurationError, "overlap with reserved group"
        ):
            build_specs(protocol)

    def test_unknown_condition_filter_is_rejected(self):
        args = parse_args(
            [
                "--experiment-id",
                EXPERIMENT_ID,
                "--conditions",
                "unknown_condition",
                "--dry-run",
            ]
        )

        with self.assertRaisesRegex(
            ExperimentConfigurationError, "condition is not in protocol"
        ):
            run_fixed_confirmation(args)

    def test_dry_run_reports_all_210_commands(self):
        args = parse_args(
            ["--experiment-id", EXPERIMENT_ID, "--dry-run"]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = run_fixed_confirmation(args)

        data = json.loads(output.getvalue())
        self.assertIsNone(result)
        self.assertEqual(data["stage"], "stage4_fixed_confirmation")
        self.assertEqual(data["full_run_count"], 210)
        self.assertEqual(data["selected_run_count"], 210)
        self.assertEqual(len(data["commands"]), 210)

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

            with patch.object(stage4_module, "RUST_BINARY", fake_binary), patch.object(
                stage4_module,
                "git_state",
                return_value={"commit": "test", "dirty": True},
            ):
                with self.assertRaisesRegex(
                    ExperimentConfigurationError, "clean Git worktree"
                ):
                    run_fixed_confirmation(args)


if __name__ == "__main__":
    unittest.main()
