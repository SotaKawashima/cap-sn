from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import optimize_single_objective as optimization
import run_network_similarity_optimization as similarity
from experiment_runtime import ExperimentConfigurationError, REPO_ROOT, sha256_file


EXPERIMENT_ID = "20260922_120000_network_similarity_optimization_v01"


class NetworkSimilarityProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = similarity.load_protocol()
        cls.specs = similarity.build_run_specs(cls.protocol)

    def test_frozen_design_counts_and_seeds(self):
        self.assertEqual(len(self.specs), 146)
        self.assertEqual(len({spec.key for spec in self.specs}), 146)
        self.assertEqual(Counter(spec.run_type for spec in self.specs),
                         {"optimization": 126, "fixed": 20})
        self.assertEqual(Counter(spec.network for spec in self.specs),
                         {name: 20 for name in similarity.ADDED_IDS}
                         | {name: 2 for name in ("ba1000", "facebook", "wiki_vote")})
        self.assertEqual(Counter(spec.condition for spec in self.specs if spec.run_type == "fixed"),
                         {"none": 10, "simple_max": 10})
        self.assertTrue(all(spec.simulator_seed == 30001 and spec.iterations == 100
                            and spec.raw_level == "pop" for spec in self.specs))
        self.assertTrue(all(spec.trials == 50 for spec in self.specs if spec.run_type == "optimization"))
        self.assertEqual(len({(spec.method, spec.optimizer_seed)
                              for spec in self.specs if spec.run_type == "optimization"}), 18)

    def test_commands_have_separate_optimization_and_fixed_designs(self):
        opt = next(spec for spec in self.specs if spec.run_type == "optimization")
        fixed = next(spec for spec in self.specs if spec.condition == "simple_max")
        none = next(spec for spec in self.specs if spec.condition == "none")
        output_root = Path(tempfile.gettempdir())
        opt_command = similarity.command_for_spec(opt, experiment_id=EXPERIMENT_ID,
                                                  output_root=output_root)
        fixed_command = similarity.command_for_spec(fixed, experiment_id=EXPERIMENT_ID,
                                                    output_root=output_root)
        none_command = similarity.command_for_spec(none, experiment_id=EXPERIMENT_ID,
                                                   output_root=output_root)
        self.assertIn("--network-config", opt_command)
        self.assertEqual(opt_command[opt_command.index("--optimizer-seed") + 1], "60101")
        self.assertEqual(opt_command[opt_command.index("--trials") + 1], "50")
        self.assertEqual(fixed_command[fixed_command.index("--certainty") + 1], "1.0")
        self.assertEqual(fixed_command[fixed_command.index("--effectiveness") + 1], "1.0")
        self.assertNotIn("--trials", fixed_command)
        self.assertIn("--no-intervention", none_command)

    def test_dry_run_and_filters(self):
        args = similarity.parse_args(["--experiment-id", EXPERIMENT_ID, "--dry-run"])
        with redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(similarity.run_experiment(args))
        data = json.loads(output.getvalue())
        self.assertEqual((data["full_run_count"], data["selected_run_count"]), (146, 146))
        args = similarity.parse_args(["--experiment-id", EXPERIMENT_ID,
                                      "--networks", "facebook_brandeis99",
                                      "--kinds", "optimization", "--methods", "bo_gp",
                                      "--optimizer-replicates", "1", "--dry-run"])
        with redirect_stdout(io.StringIO()) as output:
            similarity.run_experiment(args)
        self.assertEqual(json.loads(output.getvalue())["selected_run_count"], 1)

    def test_changed_input_hash_is_rejected(self):
        with similarity.DEFAULT_PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
        protocol["added_networks"][0]["edge_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ExperimentConfigurationError, "SHA-256 mismatch"):
                similarity.load_protocol(path)

    def test_resume_status_checks_design_and_outputs(self):
        opt = self.specs[0]
        fixed = next(spec for spec in self.specs if spec.condition == "simple_max")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            manifest = {
                "status": "completed",
                "stage": similarity.STAGE,
                "run_type": "single_objective_optimization",
                "network": {"id": opt.network, "num_agents": opt.num_agents,
                            "sha256": sha256_file(REPO_ROOT / opt.network_config)},
                "runtime": {"simulator_seed": 30001, "iteration_count": 100},
                "optimization": {"method": opt.method, "optimizer_seed": opt.optimizer_seed},
                "counts": {"complete": 50, "failed": 0, "pruned": 0},
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(similarity.inspect_run_status(run_dir, opt), "invalid_completed_outputs")
            (run_dir / "trials.csv").touch()
            self.assertEqual(similarity.inspect_run_status(run_dir, opt), "completed")
            manifest["runtime"]["simulator_seed"] = 999
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(similarity.inspect_run_status(run_dir, opt), "invalid_completed_design")
            manifest["runtime"]["simulator_seed"] = 30001
            manifest["network"]["id"] = fixed.network
            manifest["network"]["num_agents"] = fixed.num_agents
            manifest["network"]["sha256"] = sha256_file(REPO_ROOT / fixed.network_config)
            manifest["run_type"] = "fixed_condition"
            manifest["intervention"] = {"condition_id": "simple_max"}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(similarity.inspect_run_status(run_dir, fixed), "invalid_completed_outputs")
            (run_dir / "pop.arrow").touch()
            self.assertEqual(similarity.inspect_run_status(run_dir, fixed), "completed")


class CustomOptimizationNetworkTests(unittest.TestCase):
    def test_standard_network_path_remains_available(self):
        args = optimization.parse_args(["--network", "ba1000", "--method", "random_search",
                                        "--optimizer-replicate", "1", "--optimizer-seed", "1",
                                        "--simulator-seed", "30001", "--iterations", "100",
                                        "--trials", "50"])
        self.assertEqual(optimization.resolve_network(args).id, "ba1000")

    def test_custom_network_resolution_and_incomplete_pair(self):
        args = optimization.parse_args(["--network", "facebook_brandeis99",
                                        "--network-config",
                                        "experiments/summer_2026/observed_network_inputs/20260922_v01/facebook_brandeis99/network.toml",
                                        "--network-num-agents", "3898", "--method", "random_search",
                                        "--optimizer-replicate", "1", "--optimizer-seed", "1",
                                        "--simulator-seed", "30001", "--iterations", "100",
                                        "--trials", "50"])
        network = optimization.resolve_network(args)
        self.assertEqual(network.id, "facebook_brandeis99")
        self.assertEqual(network.num_agents, 3898)
        args.network_num_agents = None
        with self.assertRaisesRegex(ExperimentConfigurationError, "supplied together"):
            optimization.resolve_network(args)

    def test_real_run_requires_clean_worktree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_binary = Path(temp_dir) / "simulator"
            fake_binary.touch()
            args = similarity.parse_args(["--experiment-id", EXPERIMENT_ID,
                                          "--output-root", temp_dir])
            with patch.object(similarity, "RUST_BINARY", fake_binary), patch.object(
                similarity, "git_state", return_value={"dirty": True, "commit": "test"}
            ):
                with self.assertRaisesRegex(ExperimentConfigurationError, "clean Git worktree"):
                    similarity.run_experiment(args)


if __name__ == "__main__":
    unittest.main()
