from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.optimization_metrics import OBJECTIVE_NAME
from analysis.structure_reanalysis import (
    build_level_summary,
    build_paired_structure_contrasts,
    build_strategy_contrasts,
    discover_legacy_runs,
    load_legacy_structure_data,
    load_protocol,
)


class Stage9StructureReanalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_root = self.root / "legacy_runs"
        self.metadata_path = self.root / "generation_summary.csv"
        self.protocol_path = self.root / "protocol.json"
        self._write_fixture()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_arrow_run(
        self, network: str, strategy: str, counts: list[int]
    ) -> None:
        result = self.run_root / network / strategy / "result"
        result.mkdir(parents=True, exist_ok=True)
        prefix = f"{network}_{strategy}"
        pop = pd.DataFrame(
            {
                "num_iter": range(len(counts)),
                "t": [0] * len(counts),
                "num_selfish": counts,
            }
        )
        agent_rows = []
        for num_iter, count in enumerate(counts):
            for agent_idx in range(count):
                agent_rows.append(
                    {
                        "num_iter": num_iter,
                        "agent_idx": agent_idx,
                        "selfish": True,
                    }
                )
        agent = pd.DataFrame(
            agent_rows, columns=["num_iter", "agent_idx", "selfish"]
        )
        pop.to_feather(result / f"{prefix}_pop.arrow")
        agent.to_feather(result / f"{prefix}_agent.arrow")

    def _write_fixture(self) -> None:
        rows = []
        values = {
            "original": {
                "balance": [20, 20, 20, 20],
                "effective_high": [15, 15, 15, 15],
                "certainty_high": [25, 25, 25, 25],
            },
            "target": {
                "balance": [10, 10, 10, 10],
                "effective_high": [8, 8, 8, 8],
                "certainty_high": [12, 12, 12, 12],
            },
        }
        for seed_index in (1, 2):
            for method, strategies in values.items():
                network = f"pool_{method}_seed{seed_index}"
                rows.append(
                    {
                        "network": network,
                        "seed_index": seed_index,
                        "num_nodes": 100,
                        "comm_method": method,
                        "avg_degree": 20 + seed_index,
                    }
                )
                for strategy, counts in strategies.items():
                    shifted = [value + seed_index - 1 for value in counts]
                    self._write_arrow_run(network, strategy, shifted)
        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)
        protocol = {
            "schema_version": 1,
            "stage": "stage9_structure_reanalysis",
            "objective": {
                "name": "cumulative_selfish_fraction",
                "definition_version": "cumulative_selfish_fraction_v1",
                "secondary_metric": "peak_new_selfish_ratio",
            },
            "expected_iterations_per_run": 4,
            "expected_raw_run_count": 12,
            "bootstrap": {"repetitions": 200, "seed": 7},
            "strategies": {
                "balance": {"certainty": 0.8, "effectiveness": 0.8},
                "effective_high": {"certainty": 0.5, "effectiveness": 0.9},
                "certainty_high": {"certainty": 0.9, "effectiveness": 0.5},
            },
            "strategy_comparisons": [
                {
                    "id": "effective_high_vs_balance",
                    "reference": "balance",
                    "candidate": "effective_high",
                }
            ],
            "datasets": [
                {
                    "id": "pool",
                    "run_root": str(self.run_root),
                    "metadata_csv": str(self.metadata_path),
                    "path_layout": "standard",
                    "structure_level_template": "{comm_method}",
                    "block_kind": "network_seed",
                    "default_simulator_seed": 0,
                    "paired_structure_reference": "original",
                    "evidence_scope": "test_scope",
                    "analysis_note": "fixture",
                }
            ],
        }
        self.protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    def test_discovers_and_audits_every_legacy_run(self) -> None:
        protocol = load_protocol(self.protocol_path)
        specs = discover_legacy_runs(protocol, repo_root=self.root)
        self.assertEqual(len(specs), 12)
        data = load_legacy_structure_data(specs, expected_iterations=4)
        self.assertEqual(len(data.runs), 12)
        self.assertEqual(len(data.iterations), 48)
        self.assertTrue(data.audit["valid"].all())
        self.assertEqual(set(data.runs["block_id"]), {"network_seed_1", "network_seed_2"})

    def test_current_objective_and_strategy_contrast_are_recomputed(self) -> None:
        protocol = load_protocol(self.protocol_path)
        data = load_legacy_structure_data(
            discover_legacy_runs(protocol, repo_root=self.root),
            expected_iterations=4,
        )
        original_balance = data.runs[
            (data.runs["structure_level"] == "original")
            & (data.runs["strategy"] == "balance")
            & (data.runs["network_seed_index"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(float(original_balance["jcum"]), 0.2)

        summary = build_level_summary(data.iterations, repetitions=200, seed=10)
        row = summary[
            (summary["structure_level"] == "original")
            & (summary["strategy"] == "balance")
        ].iloc[0]
        self.assertAlmostEqual(float(row["jcum_estimate"]), 0.205)

        contrasts = build_strategy_contrasts(
            data.iterations,
            protocol["strategy_comparisons"],
            repetitions=200,
            seed=20,
        )
        row = contrasts[
            (contrasts["structure_level"] == "original")
            & (contrasts["metric"] == OBJECTIVE_NAME)
        ].iloc[0]
        self.assertGreater(float(row["reference_minus_candidate"]), 0)
        self.assertEqual(int(row["positive_blocks"]), 2)

    def test_paired_structure_contrast_uses_matching_network_seed(self) -> None:
        protocol = load_protocol(self.protocol_path)
        data = load_legacy_structure_data(
            discover_legacy_runs(protocol, repo_root=self.root),
            expected_iterations=4,
        )
        contrasts = build_paired_structure_contrasts(
            data.iterations,
            protocol,
            repetitions=200,
            seed=30,
        )
        balance = contrasts[
            (contrasts["strategy"] == "balance")
            & (contrasts["metric"] == OBJECTIVE_NAME)
        ].iloc[0]
        self.assertEqual(balance["reference_structure"], "original")
        self.assertEqual(balance["candidate_structure"], "target")
        self.assertAlmostEqual(float(balance["reference_minus_candidate"]), 0.1)
        self.assertEqual(int(balance["positive_blocks"]), 2)


if __name__ == "__main__":
    unittest.main()
