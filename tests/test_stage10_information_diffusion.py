from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.information_diffusion_analysis import (
    _metrics_match,
    _partial_spearman,
    add_bh_q_values,
    build_fixed_run_summary,
    hierarchical_paired_difference,
    load_stage10_raw,
    summarize_info_by_iteration,
)
from run_stage10_information_diffusion import (
    DEFAULT_PROTOCOL_PATH,
    build_specs,
    command_for_spec,
    load_protocol,
    main,
)


class Stage10RunnerTests(unittest.TestCase):
    def test_frozen_protocol_selects_one_candidate_per_network(self):
        specs = build_specs(load_protocol())

        self.assertEqual(len(specs), 15)
        self.assertEqual({spec.network for spec in specs}, {
            "ba1000",
            "facebook",
            "wiki_vote",
        })
        self.assertEqual({spec.simulator_seed for spec in specs}, {
            20001,
            20002,
            20003,
            20004,
            20005,
        })
        self.assertEqual({spec.raw_level for spec in specs}, {"info_pop"})
        self.assertEqual(
            {spec.condition_id for spec in specs if spec.network == "ba1000"},
            {"cand_cma_es_r03"},
        )
        self.assertEqual(
            {spec.condition_id for spec in specs if spec.network == "facebook"},
            {"cand_bo_gp_r06"},
        )
        self.assertTrue(
            all(not spec.qualified for spec in specs if spec.network == "facebook")
        )

    def test_commands_keep_fixed_parameters_and_info_pop(self):
        spec = build_specs(load_protocol())[0]
        with tempfile.TemporaryDirectory() as directory:
            command = command_for_spec(
                spec,
                experiment_id="20260827_170000_information_diffusion_v01",
                output_root=directory,
            )

        self.assertEqual(command[command.index("--network") + 1], "ba1000")
        self.assertEqual(
            command[command.index("--condition-id") + 1], "cand_cma_es_r03"
        )
        self.assertEqual(command[command.index("--certainty") + 1], "0.7947")
        self.assertEqual(command[command.index("--effectiveness") + 1], "0.8157")
        self.assertEqual(command[command.index("--raw-level") + 1], "info_pop")

    def test_dry_run_reports_fifteen_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry_run.json"
            with output.open("w", encoding="utf-8") as handle:
                from contextlib import redirect_stdout

                with redirect_stdout(handle):
                    result = main(
                        [
                            "--protocol",
                            str(DEFAULT_PROTOCOL_PATH),
                            "--experiment-id",
                            "20260827_170000_information_diffusion_v01",
                            "--dry-run",
                        ]
                    )
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(data["full_run_count"], 15)
        self.assertEqual(data["selected_run_count"], 15)


class Stage10InformationMetricTests(unittest.TestCase):
    @staticmethod
    def _info_frame(*, intervention: bool) -> pd.DataFrame:
        labels = range(4) if intervention else range(3)
        rows = []
        for num_iter in range(2):
            for label in labels:
                rows.append(
                    {
                        "num_iter": num_iter,
                        "t": 0,
                        "info_label": label,
                        "num_shared": 1 + label,
                        "num_viewed": 2 + label,
                        "num_fst_viewed": (0 if num_iter == 0 else 3 + label),
                    }
                )
        return pd.DataFrame(rows)

    def test_no_intervention_completes_behavior_guiding_with_zero(self):
        summary, _ = summarize_info_by_iteration(
            self._info_frame(intervention=False),
            num_agents=10,
            expected_iterations=2,
            intervention_enabled=False,
        )

        self.assertTrue(
            (summary["behavior_guiding_fst_viewed_count"] == 0).all()
        )
        self.assertTrue(
            summary.loc[0, "misinformation_fst_viewed_denominator_zero"]
        )
        self.assertTrue(
            math.isnan(
                summary.loc[
                    0, "corrective_to_misinformation_fst_viewed_ratio"
                ]
            )
        )
        self.assertAlmostEqual(
            summary.loc[1, "corrective_to_misinformation_fst_viewed_ratio"],
            4 / 3,
        )

    def test_partial_spearman_removes_shared_design_trend(self):
        rng = np.random.default_rng(42)
        x = np.arange(1, 101, dtype=float)
        frame = pd.DataFrame(
            {
                "indicator": x + rng.normal(0, 12, size=len(x)),
                "outcome": x + rng.normal(0, 12, size=len(x)),
                "certainty": x,
                "effectiveness": rng.uniform(0.5, 1.0, size=len(x)),
            }
        )

        result = _partial_spearman(
            frame,
            "indicator",
            "outcome",
            ["certainty", "effectiveness"],
        )

        self.assertEqual(result["n"], 100)
        self.assertLess(abs(result["rho"]), 0.25)

    def test_bh_q_values_are_monotonic_after_sorting(self):
        frame = pd.DataFrame(
            {
                "family": ["a"] * 4,
                "p_value": [0.04, 0.01, 0.20, 0.03],
            }
        )
        result = add_bh_q_values(frame, group_columns=["family"])
        ordered = result.sort_values("p_value")

        self.assertTrue((np.diff(ordered["q_value_bh"]) >= -1e-12).all())
        self.assertTrue((result["q_value_bh"] >= result["p_value"]).all())

    def test_saved_metric_comparison_rejects_missing_columns(self):
        recalculated = pd.DataFrame(
            {"num_iter": [0, 1], "cumulative_selfish_fraction": [0.1, 0.2]}
        )
        saved = recalculated[["num_iter"]].copy()

        self.assertFalse(_metrics_match(saved, recalculated))


class Stage10RawPlanTests(unittest.TestCase):
    def test_analysis_rejects_execution_plan_from_another_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "synthetic_information_diffusion_v01"
            root.mkdir()
            plan = {
                "experiment_id": root.name,
                "stage": "stage10_information_diffusion",
                "protocol": {"sha256": "0" * 64},
                "counts": {
                    "total": 15,
                    "completed": 15,
                    "pending": 0,
                    "other": 0,
                },
            }
            (root / "information_diffusion_execution_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "protocol hash"):
                load_stage10_raw(
                    root,
                    load_protocol(),
                    expected_protocol_sha256="1" * 64,
                )


class Stage10PairedComparisonTests(unittest.TestCase):
    @staticmethod
    def _iterations() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        condition_values = {
            "none": 0.30,
            "legacy_balance": 0.28,
            "prior_high": 0.20,
            "final_candidate": 0.25,
        }
        for seed in range(5):
            for num_iter in range(100):
                common = seed * 0.001 + num_iter * 0.00001
                for condition, value in condition_values.items():
                    misinformation = 10 + (num_iter % 3)
                    corrective = misinformation * (
                        2 if condition in {"prior_high", "final_candidate"} else 1
                    )
                    rows.append(
                        {
                            "run_key": f"ba1000:{condition}:simseed{seed}",
                            "source": "synthetic",
                            "network": "ba1000",
                            "condition_id": condition,
                            "condition_group": condition,
                            "condition_role": condition,
                            "intervention_enabled": condition != "none",
                            "certainty": None if condition == "none" else 0.8,
                            "effectiveness": None if condition == "none" else 0.8,
                            "simulator_seed": seed,
                            "num_agents": 1000,
                            "iterations": 100,
                            "num_iter": num_iter,
                            "cumulative_selfish_fraction": value + common,
                            "peak_new_selfish_ratio": value / 2 + common,
                            "misinformation_fst_viewed_count": misinformation,
                            "corrective_fst_viewed_count": corrective,
                            "misinformation_fst_viewed_per_agent_iteration": misinformation
                            / 1000,
                            "corrective_fst_viewed_per_agent_iteration": corrective
                            / 1000,
                            "corrective_to_misinformation_fst_viewed_ratio": corrective
                            / misinformation,
                        }
                    )
        return pd.DataFrame(rows)

    def test_hierarchical_pairing_recovers_outcome_reduction(self):
        result = hierarchical_paired_difference(
            self._iterations(),
            network="ba1000",
            reference="none",
            candidate="final_candidate",
            metric="cumulative_selfish_fraction",
            repetitions=200,
            seed=1,
        )

        self.assertAlmostEqual(result["candidate_minus_reference"], -0.05)
        self.assertLess(result["ci_high"], 0)

    def test_ratio_uses_ratio_of_resampled_sums(self):
        result = hierarchical_paired_difference(
            self._iterations(),
            network="ba1000",
            reference="none",
            candidate="final_candidate",
            metric="corrective_to_misinformation_fst_viewed_ratio",
            repetitions=200,
            seed=2,
        )

        self.assertAlmostEqual(result["reference_mean"], 1.0)
        self.assertAlmostEqual(result["candidate_mean"], 2.0)
        self.assertAlmostEqual(result["candidate_minus_reference"], 1.0)
        self.assertGreater(result["ci_low"], 0)

    def test_run_summary_uses_ratio_of_sums(self):
        data = self._iterations()
        for info_name in (
            "misinformation",
            "corrective",
            "observational",
            "behavior_guiding",
        ):
            for metric in ("shared", "viewed", "fst_viewed"):
                count_column = f"{info_name}_{metric}_count"
                rate_column = f"{info_name}_{metric}_per_agent_iteration"
                if count_column not in data:
                    data[count_column] = 1.0
                if rate_column not in data:
                    data[rate_column] = data[count_column] / 1000
        for metric in ("shared", "viewed"):
            data[f"corrective_to_misinformation_{metric}_ratio"] = 1.0
        summary = build_fixed_run_summary(data)
        candidate = summary[
            summary["condition_group"] == "final_candidate"
        ].iloc[0]

        self.assertEqual(len(summary), 20)
        self.assertAlmostEqual(
            candidate["corrective_to_misinformation_fst_viewed_ratio"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
