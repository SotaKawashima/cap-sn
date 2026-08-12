"""CLI for Stage 2 formal reaggregation of saved optimization trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.optimization_metrics import (
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
)
from analysis.saved_trial_reanalysis import (
    EXPECTED_ITERATIONS,
    EXPECTED_TRIALS_PER_RUN,
    METHOD_SPECS,
    NETWORK_SOURCE_DIRS,
    reaggregate_saved_trials,
)
from experiment_runtime import (
    MANIFEST_SCHEMA_VERSION,
    REPO_ROOT,
    SUMMER_EXPERIMENT_ROOT,
    create_unique_run_directory,
    git_state,
    make_experiment_id,
    now_iso,
    relative_to_run,
    resolve_output_root,
    software_versions,
    validate_experiment_id,
    write_json,
)


STAGE = "stage2_existing_reanalysis"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reaggregate and audit the 3,600 saved raw optimization trials."
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--purpose", default="saved_trials_reanalysis")
    parser.add_argument(
        "--networks",
        nargs="+",
        choices=sorted(NETWORK_SOURCE_DIRS),
        default=list(NETWORK_SOURCE_DIRS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_SPECS),
        default=list(METHOD_SPECS),
    )
    parser.add_argument("--optseeds", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--expected-iterations", type=int, default=EXPECTED_ITERATIONS)
    parser.add_argument(
        "--expected-trials-per-run", type=int, default=EXPECTED_TRIALS_PER_RUN
    )
    parser.add_argument("--max-trials-per-run", type=int, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT)
    return parser.parse_args(argv)


def _write_table(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    name: str,
    csv: bool = True,
    parquet: bool = True,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if csv:
        csv_path = output_dir / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        outputs["csv"] = csv_path.name
    if parquet:
        parquet_path = output_dir / f"{name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        outputs["parquet"] = parquet_path.name
    return outputs


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def run_reaggregation(args: argparse.Namespace) -> Path:
    if args.expected_iterations <= 0:
        raise ValueError("expected_iterations must be positive")
    if args.expected_trials_per_run <= 0:
        raise ValueError("expected_trials_per_run must be positive")
    if args.max_trials_per_run is not None and args.max_trials_per_run <= 0:
        raise ValueError("max_trials_per_run must be positive")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")

    experiment_id = (
        validate_experiment_id(args.experiment_id)
        if args.experiment_id
        else make_experiment_id(args.purpose)
    )
    output_root = resolve_output_root(args.output_root)
    run_dir = output_root / STAGE / experiment_id
    create_unique_run_directory(run_dir)
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    tables_dir.mkdir()
    figures_dir.mkdir()

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "stage": STAGE,
        "run_type": "offline_saved_trial_reaggregation",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "git": git_state(),
        "software": software_versions(),
        "source": {
            "networks": list(args.networks),
            "methods": list(args.methods),
            "optseeds": list(args.optseeds),
            "expected_iterations": args.expected_iterations,
            "expected_trials_per_run": args.expected_trials_per_run,
            "max_trials_per_run": args.max_trials_per_run,
        },
        "objective": {
            "name": OBJECTIVE_NAME,
            "definition_version": OBJECTIVE_DEFINITION_VERSION,
            "former_metric_name": "mean_new_selfish_rate_per_step",
        },
        "analysis": {
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "parameter_response_primary_subset": "random_search_trials",
        },
        "counts": {},
        "outputs": {},
        "timing_sec": {},
        "failure": None,
    }
    write_json(run_dir / "manifest.json", manifest)

    try:
        result = reaggregate_saved_trials(
            repo_root=REPO_ROOT,
            networks=args.networks,
            methods=args.methods,
            optseeds=args.optseeds,
            expected_iterations=args.expected_iterations,
            expected_trials_per_run=args.expected_trials_per_run,
            max_trials_per_run=args.max_trials_per_run,
            n_bootstrap=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
            progress_every=args.progress_every,
        )

        outputs: dict[str, Any] = {}
        outputs["trial_summary"] = _write_table(
            result.trial_summary,
            output_dir=tables_dir,
            name="trial_summary",
        )
        outputs["iteration_metrics"] = _write_table(
            result.iteration_metrics,
            output_dir=tables_dir,
            name="iteration_metrics",
        )
        outputs["info_summary_long"] = _write_table(
            result.info_summary,
            output_dir=tables_dir,
            name="info_summary_long",
        )
        outputs["trial_audit"] = _write_table(
            result.trial_audit,
            output_dir=tables_dir,
            name="trial_audit",
            parquet=False,
        )
        outputs["run_inventory"] = _write_table(
            result.run_inventory,
            output_dir=tables_dir,
            name="run_inventory",
            parquet=False,
        )
        for name, frame in result.analysis_tables.items():
            outputs[name] = _write_table(
                frame,
                output_dir=tables_dir,
                name=name,
                parquet=name
                in {"trial_rank_comparison", "random_cross_network_responses"},
            )

        invalid_count = int((~result.trial_audit["valid"].astype(bool)).sum())
        incomplete_run_count = int(
            (~result.run_inventory["inventory_complete"].astype(bool)).sum()
        )
        manifest["counts"] = {
            "source_runs": len(result.run_inventory),
            "inventory_complete_runs": len(result.run_inventory)
            - incomplete_run_count,
            "trial_audit_rows": len(result.trial_audit),
            "valid_trials": int(result.trial_audit["valid"].sum()),
            "invalid_trials": invalid_count,
            "trial_summary_rows": len(result.trial_summary),
            "iteration_metric_rows": len(result.iteration_metrics),
            "info_summary_rows": len(result.info_summary),
        }
        manifest["outputs"] = {
            key: {
                format_name: relative_to_run(
                    tables_dir / filename,
                    run_dir,
                )
                for format_name, filename in formats.items()
            }
            for key, formats in outputs.items()
        }
        manifest["outputs"]["figures_dir"] = relative_to_run(figures_dir, run_dir)
        manifest["timing_sec"] = {"reaggregation": result.elapsed_sec}
        manifest["status"] = (
            "completed"
            if invalid_count == 0 and incomplete_run_count == 0
            else "failed_audit"
        )

        audit_summary = {
            "experiment_id": experiment_id,
            "status": manifest["status"],
            "counts": manifest["counts"],
            "old_metric_max_absolute_error": (
                None
                if result.trial_summary.empty
                else float(result.trial_summary["j_old_absolute_error"].max())
            ),
            "all_random_pairings_equal": bool(
                result.analysis_tables["random_pairing_audit"][
                    ["proposed_points_equal", "applied_points_equal"]
                ]
                .all()
                .all()
            ),
            "random_applied_duplicate_count": int(
                result.analysis_tables["random_duplicate_audit"][
                    "applied_duplicate_count"
                ].sum()
            ),
        }
        write_json(run_dir / "audit_summary.json", _json_compatible(audit_summary))
        manifest["outputs"]["audit_summary"] = "audit_summary.json"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        manifest["updated_at"] = now_iso()
        write_json(run_dir / "manifest.json", _json_compatible(manifest))

    if manifest["status"] != "completed":
        raise RuntimeError(
            "reaggregation finished with audit failures; inspect trial_audit.csv"
        )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = run_reaggregation(args)
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Completed saved-trial reaggregation: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
