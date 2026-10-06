"""Raw-data audits and inherited selection for the multiseed extension."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from analysis.candidate_reselection_analysis import (
    build_candidate_block_performance, build_candidate_selection,
)
from analysis.candidate_validation_analysis import (
    _paired_hierarchical_effect, build_candidate_effects,
    build_seed_summary, load_candidate_validation,
)
from experiment_runtime import ExperimentConfigurationError, sha256_file
from multiseed_protocol import STAGE, digest_json, optimization_specs, read_json, repo_path
from multiseed_runtime import verify_block
from optimize_multiseed_objective import trial_block_specs
from run_multiseed_reoptimization import load_frozen_rows, phase_specs, read_csv_rows

REFERENCES = ("none", "simple_max")


def audit_optimization(root: Path, protocol: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    context = read_json(root / "execution_manifest.json")["context"]
    trial_rows = []
    block_rows = []
    pool = []
    run_rows = []
    for spec in optimization_specs(protocol):
        directory = root / spec.relative_run_dir
        manifest = read_json(directory / "manifest.json")
        if manifest.get("status") != "completed" or manifest.get("counts") != {"complete": 50, "failed": 0, "pruned": 0}:
            raise ExperimentConfigurationError(f"incomplete optimization: {spec.key}")
        if manifest.get("context_sha256") != digest_json(context) or manifest.get("spec_sha256") != digest_json(asdict(spec)):
            raise ExperimentConfigurationError(f"optimization provenance mismatch: {spec.key}")
        for name in ("trials.csv", "summary.json"):
            if manifest.get("output_hashes", {}).get(name) != sha256_file(directory / name):
                raise ExperimentConfigurationError(f"optimization output changed: {name}")
        trials = pd.read_csv(directory / "trials.csv", float_precision="round_trip")
        if len(trials) != 50 or sorted(trials["trial"].tolist()) != list(range(50)) or not trials["state"].eq("COMPLETE").all():
            raise ExperimentConfigurationError(f"invalid evaluation budget: {spec.key}")
        for row in trials.to_dict("records"):
            if row["block_count"] != 5:
                raise ExperimentConfigurationError("an evaluation is missing seed blocks")
            for variable in ("certainty", "effectiveness"):
                proposed = row[f"proposed_{variable}"]
                applied = row[f"applied_{variable}"]
                if not math.isfinite(proposed) or not 0.5 <= proposed <= 1.0 or applied != round(proposed, 4):
                    raise ExperimentConfigurationError("invalid design parameters")
            # The raw block audit does not deserialize platform-specific Optuna checkpoints.
            trial = SimpleNamespace(number=int(row["trial"]), user_attrs={"applied_parameters": {
                "certainty": row["applied_certainty"], "effectiveness": row["applied_effectiveness"],
            }})
            values = []
            for item in trial_block_specs(spec, trial):
                block_manifest, metrics = verify_block(root / item.relative_run_dir, item, context)
                values.append(metrics.objective_value)
                block_rows.append({
                    "run_key": spec.key, "network": spec.network, "method": spec.method,
                    "optimizer_replicate": spec.optimizer_replicate, "trial": trial.number,
                    "simulator_seed": item.simulator_seed, "value": metrics.objective_value,
                    "iteration_count": len(metrics.per_iteration), "valid": True,
                    "pop_sha256": block_manifest["output_hashes"]["pop.arrow"],
                    "simulation_sec": block_manifest["timing_sec"]["simulation"],
                })
            mean = sum(values) / 5
            if not math.isclose(mean, row["value"], rel_tol=0, abs_tol=1e-12):
                raise ExperimentConfigurationError("five-block mean does not match saved objective")
            trial_rows.append({**row, "run_key": spec.key, "network": spec.network,
                               "method": spec.method, "optimizer_replicate": spec.optimizer_replicate,
                               "optimizer_seed": spec.optimizer_seed, "value": mean})
        best = min((row for row in trial_rows if row["run_key"] == spec.key), key=lambda row: (row["value"], row["trial"]))
        saved_best = manifest["best"]
        if saved_best["trial"] != best["trial"] or not math.isclose(saved_best["value"], best["value"], rel_tol=0, abs_tol=1e-12):
            raise ExperimentConfigurationError("best candidate does not match audited trials")
        pool.append({
            "network": spec.network, "condition_id": f"cand_{spec.method}_r{spec.optimizer_replicate:02d}",
            "certainty": best["applied_certainty"], "effectiveness": best["applied_effectiveness"],
            "source_method": spec.method, "source_optimizer_replicate": spec.optimizer_replicate,
            "source_optimizer_seed": spec.optimizer_seed, "candidate_source": spec.key,
            "source_final_best": best["value"], "source_best_trial": best["trial"],
        })
        run_rows.append({"run_key": spec.key, "network": spec.network, "method": spec.method,
                         "optimizer_replicate": spec.optimizer_replicate, "best_value": best["value"],
                         "total_sec": manifest["timing_sec"]["total"]})
    reference_specs = [spec for spec in phase_specs(protocol, "exploration", root) if not hasattr(spec, "method")]
    references = audit_fixed(root, reference_specs, context)
    trials_frame = pd.DataFrame(trial_rows)
    trials_frame["best_so_far"] = trials_frame.sort_values("trial").groupby("run_key")["value"].cummin()
    tables = {
        "trial_inventory": trials_frame, "block_inventory": pd.DataFrame(block_rows),
        "run_inventory": pd.DataFrame(run_rows), "candidate_pool": pd.DataFrame(pool),
        "exploration_reference_summary": build_seed_summary(references.iterations),
    }
    return tables, {"status": "multiseed_exploration_complete", "optimization_run_count": len(run_rows),
                    "evaluation_count": len(trial_rows), "optimization_block_count": len(block_rows),
                    "candidate_count": len(pool)}


def audit_fixed(root: Path, specs: list[Any], context: dict[str, Any]) -> Any:
    for spec in specs:
        verify_block(root / spec.relative_run_dir, spec, context)
    return load_candidate_validation(root, expected_specs=specs, expected_stage=STAGE)


def analyze_validation(root: Path, protocol: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    context = read_json(root / "execution_manifest.json")["context"]
    data = audit_fixed(root, phase_specs(protocol, "validation", root), context)
    inference = protocol["inference"]["bootstrap"]
    effects = {reference: build_candidate_effects(
        data.iterations, reference_id=reference, candidate_role="multiseed_candidate",
        repetitions=inference["repetitions"], seed=inference["validation_seed"] + offset * 1000,
    ) for offset, reference in enumerate(REFERENCES)}
    performance = build_candidate_block_performance(
        data.iterations, candidate_role="multiseed_candidate", reference_ids=REFERENCES,
    )
    ranking, clusters, selected, decision = build_candidate_selection(
        performance, effects, reference_ids=REFERENCES,
        maximum_cluster_distance=protocol["selection_rule"]["region_grouping"]["maximum_complete_linkage_distance"],
        target_candidates_per_network=3, minimum_positive_seed_blocks=2,
        reserved_revised_final_test_seeds=protocol["seed_policy"]["final_test"],
    )
    if len(selected) != 9 or any(len(group) != 3 for _, group in selected.groupby("network")):
        raise ExperimentConfigurationError("fewer than three distinct candidate regions; user confirmation is required, not automatic rule changes")
    pool = pd.DataFrame(load_frozen_rows(root, "candidate_pool", 54))
    selected = selected.merge(pool[["network", "condition_id", "source_optimizer_seed", "source_best_trial"]],
                              on=["network", "condition_id"], how="left", validate="one_to_one")
    decision["status"] = "multiseed_candidate_selection_complete"
    decision["amended_after_original_results"] = False
    decision["descriptive_references"] = []
    decision["selection_rule_inherited_from_amended_stage7"] = True
    tables = {
        "run_inventory": data.runs, "data_audit": data.audit, "iteration_metrics": data.iterations,
        "candidate_block_performance": performance, "candidate_ranking": ranking,
        "candidate_clusters": clusters, "selected_candidates": selected,
        **{f"effects_vs_{key}": frame for key, frame in effects.items()},
    }
    return tables, decision


def analyze_final_test(root: Path, protocol: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    context = read_json(root / "execution_manifest.json")["context"]
    data = audit_fixed(root, phase_specs(protocol, "final_test", root), context)
    inference = protocol["inference"]["bootstrap"]
    frames = []
    for family, role in (("new", "multiseed_candidate"), ("historical", "historical_candidate")):
        effects = []
        for offset, reference in enumerate(REFERENCES):
            frame = build_candidate_effects(data.iterations, reference_id=reference, candidate_role=role,
                                           repetitions=inference["repetitions"], seed=inference["final_test_seed"] + offset * 1000)
            frame["candidate_family"] = family
            effects.append(frame)
        frames.extend(effects)
    final_effects = pd.concat(frames, ignore_index=True)
    new_rows = load_frozen_rows(root, "selected_candidates", 9)
    source = protocol["final_test"]["historical_candidate_source"]
    historical_rows = read_csv_rows(repo_path(source["path"]))
    metadata = []
    for family, rows in (("new", new_rows), ("historical", historical_rows)):
        for row in rows:
            metadata.append({
                "network": row["network"], "condition_id": row["condition_id"] if family == "new" else f"historical_{row['condition_id']}",
                "selection_order": int(row["selection_order"]), "selection_role": row["selection_role"],
                "qualified": row["qualified"] == "True", "candidate_family": family,
            })
    final_effects = final_effects.merge(pd.DataFrame(metadata), on=["network", "condition_id", "candidate_family"], validate="many_to_one")
    primary = final_effects[final_effects["candidate_family"].eq("new") & final_effects["selection_order"].eq(1)].copy()
    comparisons = []
    for network in protocol["execution"]["networks"]:
        new = next(row for row in new_rows if row["network"] == network and int(row["selection_order"]) == 1)
        old = next(row for row in historical_rows if row["network"] == network and int(row["selection_order"]) == 1)
        wide = data.iterations[data.iterations["network"].eq(network)].pivot(
            index=["simulator_seed", "num_iter"], columns="condition_id", values="cumulative_selfish_fraction"
        ).reset_index()
        effect = _paired_hierarchical_effect(
            wide, reference_column=f"historical_{old['condition_id']}", candidate_column=new["condition_id"],
            repetitions=inference["repetitions"], seed=inference["final_test_seed"] + 2000,
        )
        comparisons.append({"network": network, "new_condition_id": new["condition_id"], "historical_condition_id": old["condition_id"],
                            "new_selection_role": new["selection_role"], "historical_selection_role": old["selection_role"],
                            "reporting_role": "supplementary_not_a_causal_test_of_replication_count", **effect})
    tables = {
        "run_inventory": data.runs, "data_audit": data.audit, "iteration_metrics": data.iterations,
        "condition_seed_summary": build_seed_summary(data.iterations),
        "final_candidate_effects": final_effects, "primary_candidate_effects": primary,
        "historical_candidate_1_comparison": pd.DataFrame(comparisons),
    }
    return tables, {"status": "multiseed_final_evaluation_complete", "new_candidate_count": 9,
                    "historical_candidate_count": 9, "block_count": len(data.runs),
                    "final_test_seeds": protocol["seed_policy"]["final_test"], "candidates_reselected": False}
