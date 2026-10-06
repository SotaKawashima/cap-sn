"""Audit each multiseed phase and freeze its downstream candidate artifact."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from analysis.multiseed_reoptimization_analysis import (
    analyze_final_test, analyze_validation, audit_optimization,
)
from experiment_runtime import (
    ExperimentConfigurationError, git_state, now_iso, sha256_file,
    validate_safe_name, write_json,
)
from multiseed_protocol import DEFAULT_PROTOCOL_PATH, STAGE, digest_json, load_protocol, read_json
from multiseed_runtime import execution_lock
from run_multiseed_reoptimization import PLAN_NAME, read_csv_rows


def freeze_candidate_rows(root: Path, name: str, analysis_root: Path, context: dict[str, Any]) -> None:
    table = analysis_root / "tables" / f"{name}.csv"
    rows = read_csv_rows(table)
    artifact = {
        "status": "frozen", "context_sha256": digest_json(context),
        "source_analysis": (analysis_root / "analysis_manifest.json").relative_to(root).as_posix(),
        "source_table": {"path": table.relative_to(root).as_posix(), "sha256": sha256_file(table)},
        "rows": rows,
    }
    target = root / f"{name}.json"
    if target.exists():
        existing = read_json(target)
        if existing["context_sha256"] != artifact["context_sha256"] or existing["rows"] != rows:
            raise ExperimentConfigurationError("a frozen candidate artifact differs; user confirmation is required")
        return
    write_json(target, artifact)


def run_analysis(args: argparse.Namespace) -> Path:
    protocol = load_protocol(args.protocol.resolve())
    root = args.experiment_root.resolve()
    if not (root / "execution_manifest.json").is_file():
        raise ExperimentConfigurationError("execution manifest is missing")
    context = read_json(root / "execution_manifest.json")["context"]
    if context["protocol"]["sha256"] != sha256_file(args.protocol):
        raise ExperimentConfigurationError("execution protocol hash does not match analysis")
    plan = read_json(root / PLAN_NAME)
    expected = {"exploration": 84, "validation": 180, "final_test": 120}[args.phase]
    phase = plan["phases"].get(args.phase)
    if phase is None or phase["counts"] != {"total": expected, "completed": expected, "pending": 0, "other": 0}:
        raise ExperimentConfigurationError("the entire phase must be completed before analysis or candidate freezing")
    analysis_id = validate_safe_name(args.analysis_id or f"{args.phase}_analysis_v01", "analysis_id")
    analysis_root = root / analysis_id
    with execution_lock(root):
        analysis_root.mkdir(exist_ok=False)
        tables_root = analysis_root / "tables"
        tables_root.mkdir()
        manifest = {
            "schema_version": 1, "stage": STAGE, "phase": args.phase, "status": "running",
            "analysis_id": analysis_id, "started_at": now_iso(), "git": git_state(),
            "context_sha256": digest_json(context), "outputs": {}, "failure": None,
            "analysis_code": {name: sha256_file(Path(__file__).parent / name) for name in (
                "analyze_multiseed_reoptimization.py", "analysis/multiseed_reoptimization_analysis.py",
                "analysis/candidate_reselection_analysis.py", "analysis/candidate_validation_analysis.py",
            )},
        }
        manifest_path = analysis_root / "analysis_manifest.json"
        write_json(manifest_path, manifest)
        started = time.perf_counter()
        try:
            function = {"exploration": audit_optimization, "validation": analyze_validation, "final_test": analyze_final_test}[args.phase]
            tables, decision = function(root, protocol)
            outputs = {}
            for name, frame in tables.items():
                extension = "parquet" if name in {"trial_inventory", "block_inventory", "iteration_metrics"} else "csv"
                path = tables_root / f"{name}.{extension}"
                if extension == "parquet":
                    frame.to_parquet(path, index=False)
                else:
                    frame.to_csv(path, index=False)
                outputs[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "rows": len(frame)}
            write_json(analysis_root / "decision.json", decision)
            summary = {
                "status": "completed", "phase": args.phase, "experiment_id": root.name,
                "execution_counts": phase["counts"], "table_rows": {name: len(frame) for name, frame in tables.items()},
                "decision_status": decision["status"],
            }
            write_json(analysis_root / "analysis_summary.json", summary)
            for name in ("decision", "analysis_summary"):
                path = analysis_root / f"{name}.json"
                outputs[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            manifest.update({"status": "completed", "outputs": outputs, "completed_at": now_iso(),
                             "elapsed_sec": time.perf_counter() - started})
            write_json(manifest_path, manifest)
            if args.phase in {"exploration", "validation"}:
                freeze_candidate_rows(root, "candidate_pool" if args.phase == "exploration" else "selected_candidates", analysis_root, context)
        except BaseException as exc:
            manifest.update({"status": "failed", "failure": {"type": type(exc).__name__, "message": str(exc)},
                             "completed_at": now_iso(), "elapsed_sec": time.perf_counter() - started})
            write_json(manifest_path, manifest)
            raise
    return analysis_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["exploration", "validation", "final_test"], required=True)
    parser.add_argument("--analysis-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        path = run_analysis(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Completed multiseed {args.phase} analysis: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
