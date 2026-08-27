from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_custom_fixed_condition as custom_module
import run_fixed_condition as fixed_module
from analysis.structure_confirmation_analysis import (
    build_intervention_effects,
    build_simulator_seed_effects,
    build_structure_effect_contrasts,
    build_structure_level_summary,
)
from experiment_runtime import SimulationRunResult
from run_stage9_structure_confirmation import (
    build_specs,
    command_for_spec,
    load_protocol,
)


EXPERIMENT_ID = "20260824_180000_structure_confirmation_v01"


def fake_successful_simulator(**kwargs) -> SimulationRunResult:
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pop_path = output_dir / "pop.arrow"
    info_path = output_dir / "info.arrow"
    agent_path = output_dir / "agent.arrow"
    stdout_path = Path(kwargs["stdout_path"])
    stderr_path = Path(kwargs["stderr_path"])
    pd.DataFrame(
        [(0, 0, 0), (0, 1, 4), (1, 0, 0), (1, 1, 1)],
        columns=["num_iter", "t", "num_selfish"],
    ).to_feather(pop_path)
    pd.DataFrame([], columns=["num_iter", "t"]).to_feather(info_path)
    pd.DataFrame([], columns=["num_iter", "t"]).to_feather(agent_path)
    stdout_path.write_text("fake\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return SimulationRunResult(
        command=["fake-simulator"],
        elapsed_sec=0.01,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        arrow_paths={"pop": pop_path, "info": info_path, "agent": agent_path},
    )


class Stage9StructureConfirmationProtocolTests(unittest.TestCase):
    def test_frozen_protocol_builds_195_custom_network_runs(self):
        protocol = load_protocol()
        specs = build_specs(protocol)

        self.assertEqual(len(specs), 195)
        self.assertEqual(len({spec.network for spec in specs}), 13)
        self.assertEqual(len({spec.simulator_seed for spec in specs}), 5)
        self.assertEqual(
            {spec.condition_id for spec in specs},
            {"none", "legacy_balance", "prior_high"},
        )
        command = command_for_spec(
            next(spec for spec in specs if spec.condition_id == "prior_high"),
            experiment_id=EXPERIMENT_ID,
        )
        self.assertTrue(command[1].endswith("run_custom_fixed_condition.py"))
        self.assertIn("--network-config", command)
        self.assertIn("--intervention-opinion-csv", command)

    def test_custom_fixed_runner_records_network_provenance(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "v2/test_2/network/network-lfr_strong_seed1.toml"
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            fixed_module,
            "run_simulator",
            side_effect=fake_successful_simulator,
        ):
            args = custom_module.parse_args(
                [
                    "--stage",
                    "stage9_structure_confirmation",
                    "--experiment-id",
                    EXPERIMENT_ID,
                    "--network-id",
                    "lfr_strong_seed1",
                    "--network-config",
                    str(config),
                    "--network-seed",
                    "20261602",
                    "--num-agents",
                    "1000",
                    "--condition-id",
                    "none",
                    "--no-intervention",
                    "--simulator-seed",
                    "60001",
                    "--iterations",
                    "2",
                    "--raw-level",
                    "pop",
                    "--output-root",
                    temp_dir,
                ]
            )
            network = custom_module.resolve_custom_network(args)
            run_dir = fixed_module.run_fixed_condition_for_network(
                args,
                network=network,
                network_seed=args.network_seed,
            )
            with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(manifest["network"]["id"], "lfr_strong_seed1")
        self.assertEqual(manifest["network"]["network_seed"], 20261602)
        self.assertEqual(manifest["network"]["num_agents"], 1000)
        self.assertEqual(manifest["status"], "completed")
        self.assertIn("pop_arrow", manifest["outputs"])
        self.assertNotIn("info_arrow", manifest["outputs"])


class Stage9StructureConfirmationAnalysisTests(unittest.TestCase):
    def _iterations(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        definitions = [
            ("lfr_community", "strong", 0.05, 3),
            ("lfr_community", "middle", 0.2, 3),
            ("facebook_degree_rewire", "original", 0.0, 1),
            ("facebook_degree_rewire", "rewire_0p1", 0.1, 1),
        ]
        for family, level, value, network_count in definitions:
            for network_index in range(network_count):
                network = f"{family}_{level}_{network_index + 1}"
                for simulator_seed in (1, 2):
                    for iteration in range(4):
                        none = 0.4 + 0.01 * network_index + 0.001 * iteration
                        balance_delta = (
                            0.04 if level in {"strong", "original"} else 0.02
                        )
                        for condition_id, delta in (
                            ("none", 0.0),
                            ("legacy_balance", balance_delta),
                            ("prior_high", 0.08),
                        ):
                            rows.append(
                                {
                                    "family": family,
                                    "network": network,
                                    "structure_level": level,
                                    "structure_value": value,
                                    "network_seed_index": (
                                        network_index + 1
                                        if family == "lfr_community"
                                        else None
                                    ),
                                    "network_generation_seed": (
                                        network_index + 10
                                        if family == "lfr_community"
                                        else None
                                    ),
                                    "simulator_seed": simulator_seed,
                                    "num_iter": iteration,
                                    "condition_id": condition_id,
                                    "condition_role": condition_id,
                                    "intervention_enabled": condition_id != "none",
                                    "certainty": (
                                        None if condition_id == "none" else 0.8
                                    ),
                                    "effectiveness": (
                                        None if condition_id == "none" else 0.8
                                    ),
                                    "num_agents": 1000,
                                    "avg_degree": 40.0,
                                    "avg_clustering": 0.2,
                                    "modularity": 0.5,
                                    "internal_edge_ratio": 0.7,
                                    "cumulative_selfish_fraction": none - delta,
                                    "peak_new_selfish_ratio": 0.1 - delta / 2,
                                }
                            )
        return pd.DataFrame(rows)

    def test_summaries_separate_baseline_effect_and_structure_difference(self):
        iterations = self._iterations()
        summary = build_structure_level_summary(
            iterations, repetitions=200, seed=1
        )
        effects = build_intervention_effects(
            iterations, repetitions=200, seed=2
        )
        protocol = {
            "design": {
                "network_families": [
                    {"id": "lfr_community", "reference_level": "strong"},
                    {
                        "id": "facebook_degree_rewire",
                        "reference_level": "original",
                    },
                ]
            }
        }
        contrasts = build_structure_effect_contrasts(
            iterations,
            protocol,
            repetitions=200,
            seed=3,
        )

        self.assertEqual(len(summary), 12)
        self.assertEqual(len(effects), 8)
        self.assertEqual(len(contrasts), 4)
        row = effects[
            (effects["family"] == "lfr_community")
            & (effects["structure_level"] == "strong")
            & (effects["candidate"] == "legacy_balance")
        ].iloc[0]
        self.assertAlmostEqual(float(row["absolute_suppression"]), 0.04)
        contrast = contrasts[
            (contrasts["family"] == "lfr_community")
            & (contrasts["condition_id"] == "legacy_balance")
        ].iloc[0]
        self.assertLess(
            float(
                contrast[
                    "relative_suppression_difference_candidate_minus_reference"
                ]
            ),
            0,
        )

    def test_seed_effect_table_keeps_observed_graph_rows(self):
        result = build_simulator_seed_effects(self._iterations())
        observed = result[result["family"] == "facebook_degree_rewire"]

        self.assertEqual(len(observed), 8)
        self.assertTrue(observed["network_seed_index"].isna().all())
        self.assertTrue((result["relative_suppression"] > 0).all())


if __name__ == "__main__":
    unittest.main()
