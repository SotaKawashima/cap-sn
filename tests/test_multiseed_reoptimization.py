from __future__ import annotations

import copy
import io
import json
import pickle
import subprocess
import sys
import tempfile
import tomllib
import unittest
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import optuna

import analyze_multiseed_reoptimization as analysis_cli
import multiseed_runtime as runtime
import optimize_multiseed_objective as optimizer
import run_multiseed_reoptimization as runner
from analysis import multiseed_reoptimization_analysis as analysis
from analysis.candidate_reselection_analysis import (
    build_candidate_block_performance, build_candidate_selection,
)
from analysis.candidate_validation_analysis import build_candidate_effects
from experiment_runtime import (
    ExperimentConfigurationError, SimulationRunResult,
    read_intervention_opinion_csv, sha256_file, write_json,
)
from multiseed_protocol import (
    DEFAULT_PROTOCOL_PATH, STAGE, block_spec, collect_input_files, digest_json,
    load_protocol, optimization_specs, read_json, repo_path,
)
from optimize_single_objective import create_sampler
import test_stage7_candidate_reselection as reselection_tests

EXPERIMENT_ID = "20261006_120000_test_multiseed_v01"
CONTEXT = {"protocol": {"sha256": sha256_file(DEFAULT_PROTOCOL_PATH)}, "fixture": True}


def fake_simulator(**kwargs) -> SimulationRunResult:
    directory = Path(kwargs["output_dir"])
    with Path(kwargs["runtime_path"]).open("rb") as handle:
        config = tomllib.load(handle)
    opinion = directory / "inhibition_opinion.csv"
    if not kwargs["intervention_enabled"]:
        count = 300
    else:
        _, parameters = read_intervention_opinion_csv(opinion)
        count = int(130 + 60 * parameters["certainty"] + 10 * parameters["effectiveness"])
        if parameters == {"certainty": 1.0, "effectiveness": 1.0}:
            count = 260
    count += config["seed_state"] % 3
    rows = [(m, t, count + m % 3 if t == 1 else 0)
            for m in range(config["iteration_count"]) for t in (1, 2)]
    paths = {name: directory / f"{name}.arrow" for name in ("pop", "info", "agent")}
    pd.DataFrame(rows, columns=["num_iter", "t", "num_selfish"]).to_feather(paths["pop"])
    for name in ("info", "agent"):
        pd.DataFrame({"placeholder": []}).to_feather(paths[name])
    for name in ("stdout_path", "stderr_path"):
        Path(kwargs[name]).write_text("", encoding="utf-8")
    return SimulationRunResult(command=["fixture"], elapsed_sec=0.01,
                               stdout_path=kwargs["stdout_path"], stderr_path=kwargs["stderr_path"],
                               arrow_paths=paths)


def write_frozen_table(root: Path, name: str, rows: list[dict]) -> None:
    folder = root / f"{name}_fixture_analysis"
    table = folder / "tables" / f"{name}.csv"
    table.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(table, index=False)
    write_json(folder / "analysis_manifest.json", {
        "status": "completed", "context_sha256": digest_json(CONTEXT),
        "outputs": {name: {"sha256": sha256_file(table)}},
    })
    analysis_cli.freeze_candidate_rows(root, name, folder, CONTEXT)


def pool_fixture(protocol: dict) -> list[dict]:
    rows = []
    for spec in optimization_specs(protocol):
        method_index = protocol["execution"]["methods"].index(spec.method)
        rows.append({
            "network": spec.network,
            "condition_id": f"cand_{spec.method}_r{spec.optimizer_replicate:02d}",
            "certainty": round(0.5 + (spec.optimizer_replicate - 1) * 0.08, 4),
            "effectiveness": round(0.5 + method_index * 0.2, 4),
            "source_method": spec.method, "source_optimizer_replicate": spec.optimizer_replicate,
            "source_optimizer_seed": spec.optimizer_seed, "candidate_source": spec.key,
            "source_final_best": 0.15, "source_best_trial": 0,
        })
    return rows


class MultiseedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol()

    def test_inventory_and_graph_checks_are_frozen(self):
        inventory = read_json(repo_path(self.protocol["input_inventory"]["path"]))
        self.assertEqual(collect_input_files(), inventory["files"])
        self.assertEqual(len(inventory["files"]), 25)
        self.assertEqual([(row["nodes"], row["edges"]) for row in inventory["graph_checks"]],
                         [(1000, 9900), (4039, 88234), (7115, 103689)])

    def test_exploration_budget_and_disjoint_seeds(self):
        specs = runner.phase_specs(self.protocol, "exploration", Path("unused"))
        self.assertEqual(len(specs), 84)
        self.assertEqual(len(optimization_specs(self.protocol)), 54)
        self.assertEqual(Counter(spec.network for spec in specs), {network: 28 for network in runner.NETWORKS})
        self.assertEqual(Counter(spec.condition_id for spec in specs if not hasattr(spec, "method")),
                         {"none": 15, "simple_max": 15})
        groups = [set(self.protocol["seed_policy"][key]) for key in ("exploration", "validation", "final_test")]
        self.assertEqual(len(set.union(*groups)), 13)
        self.assertTrue(all(spec.simulator_seeds == tuple(sorted(groups[0]))
                            for spec in optimization_specs(self.protocol)))

    def test_changed_frozen_specification_is_rejected(self):
        for keys, value in [
            (("execution", "evaluations_per_run"), 51),
            (("objective", "aggregation"), "median"),
            (("objective", "variance_penalty"), True),
            (("quality_and_resume_policy", "required_complete_blocks_per_evaluation"), 4),
            (("inference", "bootstrap", "confidence_level"), 0.9),
            (("seed_policy", "validation"), [80001, 81002, 81003]),
        ]:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as temp:
                protocol = copy.deepcopy(self.protocol)
                target = protocol
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                path = Path(temp) / "protocol.json"
                write_json(path, protocol)
                with self.assertRaises(ExperimentConfigurationError):
                    load_protocol(path)

    def test_dry_run_does_not_create_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            args = runner.parse_args(["--experiment-id", EXPERIMENT_ID, "--output-root", temp, "--dry-run"])
            with redirect_stdout(output):
                runner.run_experiment(args)
            data = json.loads(output.getvalue())
            self.assertEqual((data["full_run_count"], data["optimization_run_count"], data["fixed_block_count"]), (84, 54, 30))
            self.assertEqual(data["full_phase_block_count"], 13530)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_pilot_filter_preserves_full_budget(self):
        output = io.StringIO()
        args = runner.parse_args(["--experiment-id", EXPERIMENT_ID, "--dry-run", "--networks", "ba1000",
                                  "--kinds", "optimization", "--optimizer-replicates", "1", "--max-evaluations", "2"])
        with redirect_stdout(output):
            runner.run_experiment(args)
        data = json.loads(output.getvalue())
        self.assertEqual(data["selected_run_count"], 3)
        self.assertTrue(all(row["trials"] == 50 for row in data["runs"]))
        self.assertTrue(all("--max-evaluations" in command for command in data["commands"]))

    def test_validation_and_final_test_require_audited_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for phase in ("validation", "final_test"):
                with self.assertRaises(FileNotFoundError):
                    runner.phase_specs(self.protocol, phase, root)

    def test_unsafe_condition_paths_are_rejected(self):
        with self.assertRaises(ExperimentConfigurationError):
            block_spec(phase="validation", network="ba1000", condition_id="../escape", simulator_seed=81001)

    def test_execution_plan_records_partial_pilot_and_verified_resumption(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "v2"
            binary.touch()
            base = ["--experiment-id", EXPERIMENT_ID, "--output-root", temp,
                    "--networks", "ba1000", "--methods", "random_search",
                    "--optimizer-replicates", "1", "--kinds", "optimization"]
            with patch.object(runner, "RUST_BINARY", binary), patch.object(runner, "git_state", return_value={"dirty": False}), \
                    patch.object(runner, "initialize_experiment", return_value=CONTEXT), \
                    patch.object(runtime, "run_simulator", side_effect=fake_simulator) as call, redirect_stdout(io.StringIO()):
                root = runner.run_experiment(runner.parse_args(base + ["--max-evaluations", "1"]))
                plan = read_json(root / runner.PLAN_NAME)
                self.assertEqual(plan["counts"], {"total": 84, "completed": 0, "pending": 83, "other": 1})
                self.assertEqual(call.call_count, 5)
                runner.run_experiment(runner.parse_args(base + ["--max-evaluations", "2", "--resume"]))
                self.assertEqual(call.call_count, 10)
                trial_csv = root / "optimization/ba1000/random_search/optseed_1/trials.csv"
                self.assertEqual(len(pd.read_csv(trial_csv)), 2)

    def test_real_execution_rejects_dirty_worktree_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "v2"
            binary.touch()
            args = runner.parse_args(["--experiment-id", EXPERIMENT_ID, "--output-root", temp])
            with patch.object(runner, "RUST_BINARY", binary), patch.object(runner, "git_state", return_value={"dirty": True}):
                with self.assertRaisesRegex(ExperimentConfigurationError, "clean Git"):
                    runner.run_experiment(args)
            self.assertFalse((Path(temp) / STAGE).exists())


class MultiseedBlockTests(unittest.TestCase):
    def setUp(self):
        self.spec = block_spec(phase="validation", network="ba1000", condition_id="cand_test",
                               simulator_seed=81001, certainty=0.75, effectiveness=0.9)

    def test_raw_metrics_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "run_simulator", side_effect=fake_simulator) as call:
            root = Path(temp)
            with runtime.execution_lock(root) as fd:
                directory = root / self.spec.relative_run_dir
                value = runtime.run_block(directory, self.spec, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=False, lock_fd=fd)
                self.assertAlmostEqual(value, runtime.verify_block(directory, self.spec, CONTEXT)[1].objective_value)
                again = runtime.run_block(directory, self.spec, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=True, lock_fd=fd)
                self.assertEqual(again, value)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.kwargs["pass_fds"], (fd,))
                self.assertFalse((directory / "info.arrow").exists())
                self.assertFalse((directory / "agent.arrow").exists())
                self.assertEqual(len(pd.read_csv(directory / "metrics.csv")), 100)

    def test_tampered_output_or_coordinates_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "run_simulator", side_effect=fake_simulator):
            root = Path(temp)
            with runtime.execution_lock(root) as fd:
                directory = root / self.spec.relative_run_dir
                runtime.run_block(directory, self.spec, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=False, lock_fd=fd)
                with self.assertRaisesRegex(ExperimentConfigurationError, "different settings"):
                    runtime.run_block(directory, replace(self.spec, certainty=0.8), context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=True, lock_fd=fd)
                (directory / "pop.arrow").write_bytes(b"damaged")
                with self.assertRaisesRegex(ExperimentConfigurationError, "hash mismatch"):
                    runtime.run_block(directory, self.spec, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=True, lock_fd=fd)

    def test_failed_attempt_is_preserved_and_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with runtime.execution_lock(root) as fd:
                directory = root / self.spec.relative_run_dir
                with patch.object(runtime, "run_simulator", side_effect=RuntimeError("fixture failure")):
                    with self.assertRaises(RuntimeError):
                        runtime.run_block(directory, self.spec, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=False, lock_fd=fd)
                self.assertEqual(read_json(directory / "manifest.json")["status"], "failed")
                with patch.object(runtime, "run_simulator", side_effect=fake_simulator):
                    runtime.run_block(directory, self.spec, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=True, lock_fd=fd)
                archives = list((directory.parent / "_failed_attempts").glob("*/manifest.json"))
                self.assertEqual(len(archives), 1)
                self.assertEqual(read_json(archives[0])["status"], "failed")

    def test_exclusive_lock_rejects_concurrent_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with runtime.execution_lock(root):
                with self.assertRaisesRegex(ExperimentConfigurationError, "active process"):
                    with runtime.execution_lock(root):
                        self.fail("second writer acquired the lock")

    def test_orphaned_child_holds_the_execution_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with runtime.execution_lock(root) as fd:
                child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                                         stdin=subprocess.PIPE, pass_fds=(fd,))
            try:
                with self.assertRaisesRegex(ExperimentConfigurationError, "active process"):
                    with runtime.execution_lock(root):
                        self.fail("child lock was lost")
            finally:
                child.communicate(input=b"", timeout=10)
            with runtime.execution_lock(root):
                pass

    def test_experiment_resume_requires_unchanged_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            binary = Path(temp) / "v2"
            binary.write_bytes(b"fixture")
            with patch.object(runtime, "RUST_BINARY", binary), \
                    patch.object(runtime, "software_versions", return_value={"fixture": "v1"}), \
                    patch.object(runtime, "git_state", return_value={"commit": "fixture", "dirty": False}):
                with runtime.execution_lock(root):
                    runtime.initialize_experiment(root, protocol_path=DEFAULT_PROTOCOL_PATH, input_files=collect_input_files(), resume=False)
                    runtime.initialize_experiment(root, protocol_path=DEFAULT_PROTOCOL_PATH, input_files=collect_input_files(), resume=True)
                    with patch.object(runtime, "software_versions", return_value={"fixture": "v2"}):
                        with self.assertRaisesRegex(ExperimentConfigurationError, "software changed"):
                            runtime.initialize_experiment(root, protocol_path=DEFAULT_PROTOCOL_PATH, input_files=collect_input_files(), resume=True)


class MultiseedOptimizerTests(unittest.TestCase):
    def _run(self, root, spec, *, resume=False, limit=None):
        with runtime.execution_lock(root) as fd, redirect_stdout(io.StringIO()):
            return optimizer.run_optimization(root, spec, context=CONTEXT, resume=resume, lock_fd=fd,
                                              max_evaluations=limit or spec.trials)

    def test_partial_resume_preserves_proposals_for_all_three_methods(self):
        for original in optimization_specs(load_protocol())[:18:6]:
            with self.subTest(method=original.method), tempfile.TemporaryDirectory() as temp, patch.object(runtime, "run_simulator", side_effect=fake_simulator):
                spec = replace(original, trials=4)
                root = Path(temp)
                direct = self._run(root / "direct", spec)
                self._run(root / "resumed", spec, limit=2)
                resumed = self._run(root / "resumed", spec, resume=True)
                columns = ["trial", "value", "proposed_certainty", "proposed_effectiveness", "applied_certainty", "applied_effectiveness", "block_count"]
                pd.testing.assert_frame_equal(pd.read_csv(direct / "trials.csv")[columns], pd.read_csv(resumed / "trials.csv")[columns])
                self.assertEqual(read_json(resumed / "manifest.json")["counts"], {"complete": 4, "failed": 0, "pruned": 0})

    def test_interrupted_evaluation_reuses_same_candidate_and_completed_blocks(self):
        spec = replace(optimization_specs(load_protocol())[12], trials=1)
        calls = []

        def interrupted(**kwargs):
            calls.append(Path(kwargs["output_dir"]).name)
            if len(calls) == 3:
                raise KeyboardInterrupt()
            return fake_simulator(**kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(runtime, "run_simulator", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
                self._run(root, spec)
            directory = root / spec.relative_run_dir
            checkpoint = runtime.load_checkpoint(directory)
            self.assertEqual(checkpoint["study"].trials[0].state.name, "RUNNING")
            original_parameters = checkpoint["active_trial"].params
            with patch.object(runtime, "run_simulator", side_effect=fake_simulator) as call:
                self._run(root, spec, resume=True)
                self.assertEqual(call.call_count, 3)
            state = runtime.load_checkpoint(directory)
            self.assertEqual(state["study"].best_trial.params, original_parameters)
            values = state["study"].best_trial.user_attrs["block_values"]
            self.assertEqual(len(values), 5)
            self.assertAlmostEqual(state["study"].best_value, sum(values) / 5)

    def test_missing_or_corrupt_checkpoint_never_resamples(self):
        spec = replace(optimization_specs(load_protocol())[12], trials=2)
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "run_simulator", side_effect=fake_simulator):
            root = Path(temp)
            directory = self._run(root, spec, limit=1)
            pointer = read_json(directory / "checkpoint.json")
            (directory / pointer["path"]).write_bytes(b"damaged")
            with self.assertRaisesRegex(ExperimentConfigurationError, "checkpoint is invalid"):
                self._run(root, spec, resume=True)
            (directory / "checkpoint.json").unlink()
            with self.assertRaisesRegex(ExperimentConfigurationError, "refusing to resample"):
                self._run(root, spec, resume=True)

    def test_bo_gp_checkpoint_after_model_based_proposals(self):
        study = optuna.create_study(direction="minimize", sampler=create_sampler("bo_gp", seed=60101, startup_trials=2))
        for _ in range(3):
            trial = study.ask()
            x = trial.suggest_float("certainty", 0.5, 1.0)
            y = trial.suggest_float("effectiveness", 0.5, 1.0)
            study.tell(trial, (x - 0.75) ** 2 + (y - 0.85) ** 2)
        restored = pickle.loads(pickle.dumps(study))
        proposals = []
        for item in (study, restored):
            trial = item.ask()
            proposals.append((trial.suggest_float("certainty", 0.5, 1.0), trial.suggest_float("effectiveness", 0.5, 1.0)))
        self.assertEqual(proposals[0], proposals[1])

    def test_all_fifty_five_block_evaluations_are_audited(self):
        protocol = load_protocol()
        spec = optimization_specs(protocol)[12]
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "run_simulator", side_effect=fake_simulator):
            root = Path(temp)
            write_json(root / "execution_manifest.json", {"context": CONTEXT})
            self._run(root, spec)
            references = [item for item in runner.phase_specs(protocol, "exploration", root) if not hasattr(item, "method")]
            with runtime.execution_lock(root) as fd:
                for item in references:
                    runtime.run_block(root / item.relative_run_dir, item, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=False, lock_fd=fd)
            with patch.object(analysis, "optimization_specs", return_value=[spec]):
                tables, decision = analysis.audit_optimization(root, protocol)
            self.assertEqual(decision["evaluation_count"], 50)
            self.assertEqual(decision["optimization_block_count"], 250)
            self.assertEqual(len(tables["exploration_reference_summary"]), 30)
            self.assertTrue(tables["trial_inventory"]["block_count"].eq(5).all())


class MultiseedSelectionTests(unittest.TestCase):
    def test_two_reference_rule_preserves_qualified_and_exploratory_selection(self):
        data = reselection_tests.CandidateReselectionAnalysisTests._iterations()
        data = data[~data.condition_id.isin(["legacy_balance", "prior_high"])].copy()
        performance = build_candidate_block_performance(data, reference_ids=analysis.REFERENCES)
        effects = {ref: build_candidate_effects(data, reference_id=ref, repetitions=100, seed=1)
                   for ref in analysis.REFERENCES}
        effects["simple_max"]["absolute_ci_low"] = -1
        _, clusters, selected, _ = build_candidate_selection(
            performance, effects, reference_ids=analysis.REFERENCES, maximum_cluster_distance=0.05,
            target_candidates_per_network=3, minimum_positive_seed_blocks=2,
            reserved_revised_final_test_seeds=[82001, 82002, 82003, 82004, 82005],
        )
        self.assertEqual(selected.condition_id.tolist(), ["cand_a", "cand_c", "cand_d"])
        self.assertEqual(selected.selection_role.tolist(), ["qualified_candidate", "qualified_candidate", "exploratory_fallback"])
        self.assertEqual(len(clusters), 3)

    def test_validation_freezing_and_final_analysis_end_to_end(self):
        protocol = copy.deepcopy(load_protocol())
        protocol["inference"]["bootstrap"]["repetitions"] = 100
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "run_simulator", side_effect=fake_simulator):
            root = Path(temp)
            write_json(root / "execution_manifest.json", {"context": CONTEXT})
            write_frozen_table(root, "candidate_pool", pool_fixture(protocol))
            specifications = runner.phase_specs(protocol, "validation", root)
            self.assertEqual(len(specifications), 180)
            with runtime.execution_lock(root) as fd:
                for item in specifications:
                    runtime.run_block(root / item.relative_run_dir, item, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=False, lock_fd=fd)
            write_json(root / runner.PLAN_NAME, {"phases": {"validation": {
                "counts": {"total": 180, "completed": 180, "pending": 0, "other": 0}}}})
            args = analysis_cli.parse_args(["--experiment-root", temp, "--phase", "validation"])
            with patch.object(analysis_cli, "load_protocol", return_value=protocol):
                folder = analysis_cli.run_analysis(args)
            decision = read_json(folder / "decision.json")
            self.assertEqual(decision["status"], "multiseed_candidate_selection_complete")
            selected = pd.read_csv(folder / "tables/selected_candidates.csv")
            self.assertEqual(len(selected), 9)
            frozen_before = sha256_file(root / "selected_candidates.json")
            final_specs = runner.phase_specs(protocol, "final_test", root)
            self.assertEqual(len(final_specs), 120)
            self.assertEqual(Counter(item.condition_role for item in final_specs),
                             {"multiseed_candidate": 45, "historical_candidate": 45, "reference": 30})
            with runtime.execution_lock(root) as fd:
                for item in final_specs:
                    runtime.run_block(root / item.relative_run_dir, item, context=CONTEXT, experiment_id=EXPERIMENT_ID, resume=False, lock_fd=fd)
            write_json(root / runner.PLAN_NAME, {"phases": {"final_test": {
                "counts": {"total": 120, "completed": 120, "pending": 0, "other": 0}}}})
            args = analysis_cli.parse_args(["--experiment-root", temp, "--phase", "final_test"])
            with patch.object(analysis_cli, "load_protocol", return_value=protocol):
                final_folder = analysis_cli.run_analysis(args)
            final_decision = read_json(final_folder / "decision.json")
            self.assertFalse(final_decision["candidates_reselected"])
            for name, count in (("final_candidate_effects", 36), ("primary_candidate_effects", 6),
                                ("historical_candidate_1_comparison", 3)):
                self.assertEqual(len(pd.read_csv(final_folder / "tables" / f"{name}.csv")), count)
            self.assertEqual(frozen_before, sha256_file(root / "selected_candidates.json"))
            table = folder / "tables/selected_candidates.csv"
            table.write_text("damaged", encoding="utf-8")
            with self.assertRaisesRegex(ExperimentConfigurationError, "changed"):
                runner.phase_specs(protocol, "final_test", root)

    def test_analysis_refuses_partial_phase_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "execution_manifest.json", {"context": CONTEXT})
            write_json(root / runner.PLAN_NAME, {"phases": {"exploration": {
                "counts": {"total": 84, "completed": 1, "pending": 83, "other": 0}}}})
            args = analysis_cli.parse_args(["--experiment-root", temp, "--phase", "exploration"])
            with self.assertRaisesRegex(ExperimentConfigurationError, "entire phase"):
                analysis_cli.run_analysis(args)
            self.assertFalse((root / "exploration_analysis_v01").exists())


if __name__ == "__main__":
    unittest.main()
