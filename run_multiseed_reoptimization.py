"""Dispatch the frozen exploration, validation, and final-test phases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiment_runtime import (
    NETWORKS, REPO_ROOT, RUST_BINARY, SUMMER_EXPERIMENT_ROOT,
    ExperimentConfigurationError, git_state, make_experiment_id, now_iso,
    resolve_output_root, sha256_file, validate_experiment_id, write_json,
)
from multiseed_protocol import (
    DEFAULT_PROTOCOL_PATH, STAGE, OptimizationSpec, block_spec,
    digest_json, load_protocol, optimization_specs, read_json, repo_path,
)
from multiseed_runtime import execution_lock, initialize_experiment, run_block
from optimize_multiseed_objective import run_optimization

PLAN_NAME = "multiseed_reoptimization_execution_plan.json"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_frozen_rows(root: Path, name: str, expected_count: int) -> list[dict[str, str]]:
    artifact = read_json(root / f"{name}.json")
    context = read_json(root / "execution_manifest.json")["context"]
    if artifact.get("status") != "frozen" or artifact.get("context_sha256") != digest_json(context):
        raise ExperimentConfigurationError(f"invalid frozen {name}")
    source = artifact["source_table"]
    path = (root / source["path"]).resolve()
    analysis_path = (root / artifact["source_analysis"]).resolve()
    if not path.is_relative_to(root.resolve()) or not analysis_path.is_relative_to(root.resolve()):
        raise ExperimentConfigurationError("frozen candidate source escaped experiment")
    analysis = read_json(analysis_path)
    if (analysis.get("status") != "completed"
            or analysis.get("context_sha256") != digest_json(context)
            or analysis["outputs"][name]["sha256"] != source["sha256"]):
        raise ExperimentConfigurationError("candidate source analysis is not completed or has changed")
    if sha256_file(path) != source["sha256"] or read_csv_rows(path) != artifact["rows"]:
        raise ExperimentConfigurationError(f"frozen {name} changed")
    rows = artifact["rows"]
    if len(rows) != expected_count or Counter(row["network"] for row in rows) != {network: expected_count // 3 for network in NETWORKS}:
        raise ExperimentConfigurationError(f"invalid row count in {name}")
    if len({(row["network"], row["condition_id"]) for row in rows}) != expected_count:
        raise ExperimentConfigurationError("candidate IDs are not unique within networks")
    if name == "selected_candidates":
        for network in NETWORKS:
            if sorted(int(row["selection_order"]) for row in rows if row["network"] == network) != [1, 2, 3]:
                raise ExperimentConfigurationError("shortlist must freeze selection orders 1, 2, 3")
    return rows


def _metadata(row: dict[str, str], role: str) -> dict[str, Any]:
    def optional_integer(field: str) -> int | None:
        if not row.get(field):
            return None
        value = float(row[field])
        if not math.isfinite(value) or not value.is_integer():
            raise ExperimentConfigurationError(f"invalid integer candidate metadata: {field}")
        return int(value)

    return {
        "condition_role": role, "candidate_source": row.get("candidate_source"),
        "source_method": row.get("source_method"),
        "source_optimizer_replicate": optional_integer("source_optimizer_replicate"),
        "source_optimizer_seed": optional_integer("source_optimizer_seed"),
        "source_final_best": float(row["source_final_best"]),
    }


def phase_specs(protocol: dict[str, Any], phase: str, root: Path) -> list[Any]:
    if phase not in {"exploration", "validation", "final_test"}:
        raise ExperimentConfigurationError(f"unknown phase: {phase}")
    specs: list[Any] = optimization_specs(protocol) if phase == "exploration" else []
    rows: list[dict[str, Any]] = []
    if phase == "validation":
        rows = load_frozen_rows(root, "candidate_pool", 54)
    elif phase == "final_test":
        rows = load_frozen_rows(root, "selected_candidates", 9)
        source = protocol["final_test"]["historical_candidate_source"]
        historical = read_csv_rows(repo_path(source["path"]))
        if len(historical) != 9:
            raise ExperimentConfigurationError("historical pool must have nine rows")
        for item in historical:
            row = dict(item)
            row["condition_id"] = f"historical_{item['condition_id']}"
            row["condition_role"] = "historical_candidate"
            rows.append(row)
    for row in rows:
        for seed in protocol["seed_policy"][phase]:
            specs.append(block_spec(
                phase=phase, network=row["network"], condition_id=row["condition_id"],
                simulator_seed=seed, certainty=float(row["certainty"]),
                effectiveness=float(row["effectiveness"]),
                metadata=_metadata(row, row.get("condition_role", "multiseed_candidate")),
            ))
    directory = "exploration_references" if phase == "exploration" else phase
    for network in protocol["execution"]["networks"]:
        for condition in protocol["comparators"]:
            for seed in protocol["seed_policy"][phase]:
                specs.append(block_spec(
                    phase=directory, network=network, condition_id=condition["id"],
                    simulator_seed=seed, certainty=condition["certainty"], effectiveness=condition["effectiveness"],
                ))
    if len({spec.key for spec in specs}) != len(specs):
        raise ExperimentConfigurationError("run keys are not unique")
    return specs


def inspect_status(root: Path, spec: Any, context: dict[str, Any]) -> str:
    directory = root / spec.relative_run_dir
    if not directory.exists():
        return "pending"
    try:
        manifest = read_json(directory / "manifest.json")
        if manifest.get("context_sha256") != digest_json(context) or manifest.get("spec_sha256") != digest_json(asdict(spec)):
            return "invalid_settings"
        status = manifest.get("status", "unknown")
        if status == "completed" and isinstance(spec, OptimizationSpec) and manifest.get("counts") != {"complete": 50, "failed": 0, "pruned": 0}:
            return "invalid_budget"
        return status
    except (OSError, ValueError, KeyError):
        return "invalid_manifest"


def _phase_inputs(root: Path, phase: str) -> dict[str, str]:
    names = {"exploration": [], "validation": ["candidate_pool"], "final_test": ["selected_candidates"]}[phase]
    return {name: sha256_file(root / f"{name}.json") for name in names}


def command_for_spec(spec: Any, args: argparse.Namespace, experiment_id: str) -> list[str]:
    command = [
        sys.executable, str(REPO_ROOT / "run_multiseed_reoptimization.py"),
        "--protocol", str(args.protocol.resolve()), "--experiment-id", experiment_id,
        "--phase", args.phase, "--run-key", spec.key,
        "--output-root", str(resolve_output_root(args.output_root)), "--resume",
    ]
    if isinstance(spec, OptimizationSpec) and args.max_evaluations != 50:
        command.extend(["--max-evaluations", str(args.max_evaluations)])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-id")
    parser.add_argument("--phase", choices=["exploration", "validation", "final_test"], default="exploration")
    parser.add_argument("--networks", nargs="+", choices=sorted(NETWORKS))
    parser.add_argument("--methods", nargs="+", choices=["bo_gp", "cma_es", "random_search"])
    parser.add_argument("--optimizer-replicates", nargs="+", type=int, choices=range(1, 7))
    parser.add_argument("--kinds", nargs="+", choices=["optimization", "fixed"])
    parser.add_argument("--run-key")
    parser.add_argument("--max-evaluations", type=int, default=50,
                        help="Stop after this cumulative evaluation count; --resume later continues to 50.")
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Path | None:
    protocol = load_protocol(args.protocol.resolve())
    experiment_id = validate_experiment_id(args.experiment_id) if args.experiment_id else make_experiment_id(STAGE)
    root = resolve_output_root(args.output_root) / STAGE / experiment_id
    all_specs = phase_specs(protocol, args.phase, root)
    if not 1 <= args.max_evaluations <= 50 or (args.phase != "exploration" and args.max_evaluations != 50):
        raise ExperimentConfigurationError("evaluation limit is only supported for exploration and must be in [1,50]")
    selected = [spec for spec in all_specs if (
        (not args.networks or spec.network in args.networks)
        and (not args.run_key or spec.key == args.run_key)
        and (not args.kinds or ("optimization" if isinstance(spec, OptimizationSpec) else "fixed") in args.kinds)
        and (not isinstance(spec, OptimizationSpec) or (
            (not args.methods or spec.method in args.methods)
            and (not args.optimizer_replicates or spec.optimizer_replicate in args.optimizer_replicates)
        ))
    )]
    if not selected:
        raise ExperimentConfigurationError("run selection is empty")
    if args.dry_run:
        print(json.dumps({
            "experiment_id": experiment_id, "stage": STAGE, "phase": args.phase,
            "full_run_count": len(all_specs), "selected_run_count": len(selected),
            "optimization_run_count": sum(isinstance(spec, OptimizationSpec) for spec in selected),
            "fixed_block_count": sum(not isinstance(spec, OptimizationSpec) for spec in selected),
            "full_phase_block_count": {"exploration": 13530, "validation": 180, "final_test": 120}[args.phase],
            "runs": [asdict(spec) for spec in selected],
            "commands": [command_for_spec(spec, args, experiment_id) for spec in selected],
        }, indent=2))
        return None
    if not RUST_BINARY.is_file():
        raise ExperimentConfigurationError("release simulator is missing")
    if git_state()["dirty"] and not args.allow_dirty:
        raise ExperimentConfigurationError("execution requires a clean Git worktree; commit first")
    if not protocol.get("input_inventory"):
        raise ExperimentConfigurationError("the input inventory must be frozen before execution")
    input_files = read_json(repo_path(protocol["input_inventory"]["path"]))["files"]
    with execution_lock(root) as descriptor:
        context = initialize_experiment(root, protocol_path=args.protocol.resolve(), input_files=input_files, resume=args.resume)
        source_hashes = _phase_inputs(root, args.phase)
        plan_path = root / PLAN_NAME
        plan = read_json(plan_path) if plan_path.exists() else {"schema_version": 1, "stage": STAGE, "experiment_id": experiment_id, "phases": {}}
        previous = plan["phases"].get(args.phase)
        if previous is not None and previous["phase_inputs"] != source_hashes:
            raise ExperimentConfigurationError("frozen phase inputs changed")

        def update_plan() -> None:
            rows = [{**asdict(spec), "kind": "optimization" if isinstance(spec, OptimizationSpec) else "fixed",
                     "status": inspect_status(root, spec, context)} for spec in all_specs]
            statuses = Counter(row["status"] for row in rows)
            counts = {"total": len(rows), "completed": statuses["completed"], "pending": statuses["pending"],
                      "other": len(rows) - statuses["completed"] - statuses["pending"]}
            plan["phases"][args.phase] = {"phase_inputs": source_hashes, "counts": counts, "runs": rows}
            plan.update({"updated_at": now_iso(), "context_sha256": digest_json(context), "current_phase": args.phase, "counts": counts})
            write_json(plan_path, plan)

        update_plan()
        try:
            for index, spec in enumerate(selected, start=1):
                status = inspect_status(root, spec, context)
                print(f"[{index}/{len(selected)}] {status}: {spec.key}", flush=True)
                if status.startswith("invalid"):
                    raise ExperimentConfigurationError(f"cannot resume invalid run: {spec.key}")
                if isinstance(spec, OptimizationSpec):
                    run_optimization(root, spec, context=context, resume=args.resume, lock_fd=descriptor,
                                     max_evaluations=args.max_evaluations)
                else:
                    run_block(root / spec.relative_run_dir, spec, context=context,
                              experiment_id=experiment_id, resume=args.resume, lock_fd=descriptor)
                update_plan()
        finally:
            update_plan()
    return root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = run_experiment(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if root is not None:
        print(f"Finished selected multiseed runs; inspect phase counts: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
