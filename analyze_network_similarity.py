#!/usr/bin/env python3
"""Audit and compare same-family network exploration results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.network_similarity_analysis import (
    GOOD_FRACTION,
    build_good_region_summary,
    build_good_region_transfer,
    build_pairwise_similarity,
    build_reference_comparisons,
    build_shared_random_points,
    load_similarity_subset,
    load_stage6_trial_inventory,
    plot_family_rank_landscape,
)
from experiment_runtime import (
    REPO_ROOT,
    git_state,
    now_iso,
    relative_to_repo,
    sha256_file,
    validate_safe_name,
    write_json,
)
from run_network_similarity_optimization import (
    DEFAULT_PROTOCOL_PATH,
    STAGE,
    build_run_specs,
    load_protocol,
)


DEFAULT_STAGE6_ANALYSIS_ROOT = (
    REPO_ROOT
    / "experiments"
    / "summer_2026"
    / "stage6_reoptimization"
    / "20260819_162842_stage6_reoptimization_v01"
    / "reoptimization_analysis_v01"
)
DEFAULT_ANALYSIS_ID = "network_similarity_analysis_v01"
FAMILIES = ("ba1000", "facebook", "wiki_vote")
NETWORK_LABELS = {
    "ba1000": "BA1000 seed 1",
    "ba1000_seed2": "BA1000 seed 2",
    "ba1000_seed3": "BA1000 seed 3",
    "ba1000_seed4": "BA1000 seed 4",
    "facebook": "SNAP ego-Facebook",
    "facebook_brandeis99": "Brandeis99",
    "facebook_bucknell39": "Bucknell39",
    "facebook_rice31": "Rice31",
    "wiki_vote": "SNAP Wiki-vote",
    "wiki_rfa_post2008": "Wiki-RfA post-2008",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--stage6-analysis-root",
        type=Path,
        default=DEFAULT_STAGE6_ANALYSIS_ROOT,
    )
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="Families to analyze; the default includes the complete planned scope.",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _output_entry(path: Path, rows: int | None = None) -> dict[str, Any]:
    if path.parent.name in {"tables", "figures"}:
        relative = f"{path.parent.name}/{path.name}"
    else:
        relative = path.name
    entry: dict[str, Any] = {"path": relative, "sha256": sha256_file(path)}
    if rows is not None:
        entry["rows"] = int(rows)
    return entry


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    outputs: dict[str, dict[str, Any]],
    key: str,
) -> None:
    frame.to_csv(path, index=False)
    outputs[key] = _output_entry(path, len(frame))


def _write_parquet(
    frame: pd.DataFrame,
    path: Path,
    outputs: dict[str, dict[str, Any]],
    key: str,
) -> None:
    frame.to_parquet(path, index=False)
    outputs[key] = _output_entry(path, len(frame))


def _validate_source_subset(
    experiment_root: Path,
    protocol_path: Path,
    selected_specs: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = experiment_root / "network_similarity_execution_plan.json"
    study_path = experiment_root / "study_manifest.json"
    if not plan_path.is_file() or not study_path.is_file():
        raise FileNotFoundError("execution plan or study manifest is missing")
    plan = _read_json(plan_path)
    study = _read_json(study_path)
    if plan.get("stage") != STAGE:
        raise ValueError("execution plan has an unexpected stage")
    if plan.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("local protocol does not match the executed protocol")

    plan_by_dir = {
        str(row.get("relative_run_dir")): row for row in plan.get("runs", [])
    }
    study_by_dir = {str(row.get("path")): row for row in study.get("runs", [])}
    selected_dirs = {spec.relative_run_dir for spec in selected_specs}
    missing_plan = sorted(selected_dirs - set(plan_by_dir))
    missing_study = sorted(selected_dirs - set(study_by_dir))
    if missing_plan or missing_study:
        raise ValueError(
            f"selected runs are absent from plan/study: "
            f"plan={missing_plan}, study={missing_study}"
        )
    incomplete_plan = sorted(
        path for path in selected_dirs if plan_by_dir[path].get("status") != "completed"
    )
    incomplete_study = sorted(
        path for path in selected_dirs if study_by_dir[path].get("status") != "completed"
    )
    if incomplete_plan or incomplete_study:
        raise ValueError(
            f"selected runs are incomplete: plan={incomplete_plan}, "
            f"study={incomplete_study}"
        )
    return plan, study


def _family_inventory(
    protocol: dict[str, Any],
    selected_families: list[str],
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    additions: dict[str, list[str]] = {family: [] for family in selected_families}
    for row in protocol["added_networks"]:
        family = str(row["family"])
        if family in additions:
            additions[family].append(str(row["id"]))
    empty = [family for family, networks in additions.items() if not networks]
    if empty:
        raise ValueError(f"selected families have no added networks: {empty}")
    family_networks = {
        family: [family, *networks] for family, networks in additions.items()
    }
    original_networks = list(selected_families)
    added_networks = [
        network for family in selected_families for network in additions[family]
    ]
    return family_networks, original_networks, added_networks


def run_analysis(args: argparse.Namespace) -> Path:
    experiment_root = args.experiment_root.resolve()
    protocol_path = args.protocol.resolve()
    stage6_analysis_root = args.stage6_analysis_root.resolve()
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    selected_families = list(dict.fromkeys(args.families))
    all_families_selected = set(selected_families) == set(FAMILIES)
    analysis_root = experiment_root / analysis_id
    if analysis_root.exists():
        raise FileExistsError(f"analysis directory already exists: {analysis_root}")
    tables_root = analysis_root / "tables"
    figures_root = analysis_root / "figures"
    tables_root.mkdir(parents=True)
    figures_root.mkdir()

    started = time.perf_counter()
    manifest_path = analysis_root / "analysis_manifest.json"
    module_path = REPO_ROOT / "analysis" / "network_similarity_analysis.py"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "stage": "network_similarity_analysis",
        "status": "running",
        "provisional": not all_families_selected,
        "started_at": now_iso(),
        "source_experiment_root": relative_to_repo(experiment_root),
        "selected_families": selected_families,
        "protocol": {
            "path": relative_to_repo(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "stage6_analysis_root": relative_to_repo(stage6_analysis_root),
        "analysis_code": {
            "cli": {
                "path": relative_to_repo(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "module": {
                "path": relative_to_repo(module_path),
                "sha256": sha256_file(module_path),
            },
        },
        "git": git_state(),
        "invocation": sys.argv,
        "outputs": {},
        "failure": None,
    }
    write_json(manifest_path, manifest)

    try:
        protocol = load_protocol(protocol_path)
        all_specs = build_run_specs(protocol)
        family_networks, original_networks, added_networks = _family_inventory(
            protocol,
            selected_families,
        )
        selected_specs = [spec for spec in all_specs if spec.network in added_networks]
        expected_selected_runs = len(added_networks) * 20
        if len(selected_specs) != expected_selected_runs:
            raise ValueError(
                f"selected run count={len(selected_specs)}, "
                f"expected={expected_selected_runs}"
            )
        plan, study = _validate_source_subset(
            experiment_root,
            protocol_path,
            selected_specs,
        )

        added = load_similarity_subset(
            experiment_root,
            expected_specs=selected_specs,
        )
        stage6 = load_stage6_trial_inventory(
            stage6_analysis_root,
            networks=original_networks,
        )
        added_trials = added.trials.copy()
        added_trials["source_experiment"] = "network_similarity_optimization"
        combined_trials = pd.concat([stage6, added_trials], ignore_index=True)

        shared = build_shared_random_points(
            combined_trials,
            family_networks=family_networks,
            good_fraction=GOOD_FRACTION,
        )
        pairwise = build_pairwise_similarity(shared)
        good_regions = build_good_region_summary(shared)
        transfers = build_good_region_transfer(shared)
        references = build_reference_comparisons(added_trials, added.fixed)

        outputs: dict[str, dict[str, Any]] = {}
        _write_csv(added.audit, tables_root / "data_audit.csv", outputs, "data_audit")
        _write_csv(
            added.runs,
            tables_root / "run_inventory.csv",
            outputs,
            "run_inventory",
        )
        _write_csv(
            added.fixed,
            tables_root / "fixed_reference_values.csv",
            outputs,
            "fixed_reference_values",
        )
        _write_parquet(
            added_trials,
            tables_root / "added_trial_inventory.parquet",
            outputs,
            "added_trial_inventory",
        )
        _write_parquet(
            combined_trials,
            tables_root / "combined_trial_inventory.parquet",
            outputs,
            "combined_trial_inventory",
        )
        _write_parquet(
            shared,
            tables_root / "shared_random_points.parquet",
            outputs,
            "shared_random_points",
        )
        _write_csv(
            pairwise,
            tables_root / "pairwise_random_surface_similarity.csv",
            outputs,
            "pairwise_random_surface_similarity",
        )
        _write_csv(
            good_regions,
            tables_root / "good_region_summary.csv",
            outputs,
            "good_region_summary",
        )
        _write_csv(
            transfers,
            tables_root / "good_region_transfer.csv",
            outputs,
            "good_region_transfer",
        )
        _write_csv(
            references,
            tables_root / "fixed_reference_comparisons.csv",
            outputs,
            "fixed_reference_comparisons",
        )

        for family, networks in family_networks.items():
            figure_path = figures_root / f"{family}_common_random_rank_landscape.png"
            plot_family_rank_landscape(
                shared,
                family=family,
                networks=networks,
                labels=NETWORK_LABELS,
                output_path=figure_path,
            )
            outputs[f"{family}_rank_landscape"] = _output_entry(figure_path)

        overall_counts = plan.get("counts", {})
        facebook_included = "facebook" in selected_families
        overall_execution_complete = overall_counts.get("completed") == overall_counts.get(
            "total"
        ) and overall_counts.get("pending") == 0 and overall_counts.get("other") == 0
        planned_scope_complete = all_families_selected and overall_execution_complete
        interpretation_limits = [
            "All added-network exploration uses one simulator seed and is not independent validation.",
            "The same numeric simulator seed on different graphs does not pair identical stochastic realizations.",
            "BO-GP and CMA-ES choose graph-dependent adaptive coordinates, so all-900-point density is descriptive only.",
            "Fixed-reference comparisons reuse the exploration simulator seed and cannot establish unseen-seed efficacy.",
        ]
        if not facebook_included:
            interpretation_limits.insert(
                0,
                "Facebook-family optimization is not included in this partial analysis.",
            )
        if "facebook" in selected_families:
            interpretation_limits.append(
                "Facebook-family graphs are separate observed networks, not structure-controlled replicas."
            )
        if "wiki_vote" in selected_families:
            interpretation_limits.append(
                "Wiki-RfA is a later related Wikipedia process, not a controlled structural replicate of Wiki-vote."
            )
        decision = {
            "status": (
                "network_similarity_analysis_complete"
                if planned_scope_complete
                else "partial_network_similarity_analysis_complete"
            ),
            "provisional": not planned_scope_complete,
            "planned_scope_complete": planned_scope_complete,
            "selected_families": selected_families,
            "facebook_included": facebook_included,
            "overall_execution_complete": overall_execution_complete,
            "claim_scope": "descriptive same-seed comparison within broad network families",
            "unseen_seed_efficacy_claim_allowed": False,
            "structural_causal_claim_allowed": False,
            "primary_comparison": {
                "method": "random_search_common_coordinates",
                "points_per_network": 300,
                "good_region_definition": "lowest 10% (30/300) J_cum points",
                "metrics": [
                    "Spearman objective-rank correlation",
                    "bottom-decile overlap and Jaccard index",
                    "bottom-decile certainty/effectiveness quartiles",
                    "cross-network transferred objective percentile",
                ],
            },
            "pairwise_results": _json_records(pairwise),
            "good_region_results": _json_records(good_regions),
            "interpretation_limits": interpretation_limits,
        }
        decision_path = analysis_root / "decision.json"
        write_json(decision_path, decision)
        outputs["decision"] = _output_entry(decision_path)

        summary = {
            "status": "completed",
            "provisional": not planned_scope_complete,
            "planned_scope_complete": planned_scope_complete,
            "experiment_id": plan.get("experiment_id"),
            "executed_git_commit": plan.get("git", {}).get("commit"),
            "overall_execution_counts": overall_counts,
            "study_manifest_recorded_runs": int(len(study.get("runs", []))),
            "selected_families": selected_families,
            "original_networks": original_networks,
            "added_networks": added_networks,
            "selected_run_count": int(len(selected_specs)),
            "valid_selected_run_count": int(added.audit["valid"].sum()),
            "optimization_run_count": int(len(added.runs)),
            "fixed_run_count": int(len(added.fixed)),
            "added_trial_count": int(len(added_trials)),
            "stage6_trial_count": int(len(stage6)),
            "combined_trial_count": int(len(combined_trials)),
            "shared_random_point_rows": int(len(shared)),
            "shared_random_points_per_network": 300,
            "pairwise_comparison_count": int(len(pairwise)),
            "good_fraction": GOOD_FRACTION,
            "decision_status": decision["status"],
        }
        summary_path = analysis_root / "analysis_summary.json"
        write_json(summary_path, summary)
        outputs["analysis_summary"] = _output_entry(summary_path)

        manifest.update(
            {
                "status": "completed",
                "provisional": not planned_scope_complete,
                "completed_at": now_iso(),
                "elapsed_sec": time.perf_counter() - started,
                "source_execution": {
                    "experiment_id": plan.get("experiment_id"),
                    "git_commit": plan.get("git", {}).get("commit"),
                    "overall_counts": overall_counts,
                    "selected_run_count": len(selected_specs),
                },
                "outputs": outputs,
            }
        )
        write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": now_iso(),
                "elapsed_sec": time.perf_counter() - started,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(manifest_path, manifest)
        raise
    return analysis_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analysis_root = run_analysis(args)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Completed network-similarity analysis: {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
