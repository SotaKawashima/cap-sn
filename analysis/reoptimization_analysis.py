"""Formal audit and aggregation helpers for the Stage 6 reoptimization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from analysis.optimization_metrics import (
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
    compute_selfish_metrics_from_arrow,
)
from experiment_runtime import NETWORKS, read_intervention_opinion_csv


@dataclass(frozen=True)
class ReoptimizationData:
    """Validated Stage 6 data at trial, iteration, run, and audit levels."""

    trials: pd.DataFrame
    iterations: pd.DataFrame
    runs: pd.DataFrame
    audit: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_matches(observed: Any, expected: Any, tolerance: float) -> bool:
    if observed is None or expected is None:
        return observed is None and expected is None
    try:
        return math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _manifest_errors(
    manifest: dict[str, Any],
    spec: Any,
    *,
    tolerance: float,
) -> list[str]:
    optimization = manifest.get("optimization", {})
    objective = manifest.get("objective", {})
    intervention = manifest.get("intervention", {})
    observed = {
        "status": manifest.get("status"),
        "stage": manifest.get("stage"),
        "run_type": manifest.get("run_type"),
        "network": manifest.get("network", {}).get("id"),
        "simulator_seed": manifest.get("runtime", {}).get("simulator_seed"),
        "iterations": manifest.get("runtime", {}).get("iteration_count"),
        "method": optimization.get("method"),
        "optimizer_replicate": optimization.get("optimizer_replicate"),
        "optimizer_seed": optimization.get("optimizer_seed"),
        "trials": optimization.get("n_trials_requested"),
        "startup_trials": optimization.get("startup_trials"),
        "raw_level": optimization.get("raw_level"),
        "objective_name": objective.get("name"),
        "objective_version": objective.get("definition_version"),
        "objective_direction": objective.get("direction"),
        "application_precision": intervention.get(
            "application_precision_decimal_places"
        ),
    }
    expected = {
        "status": "completed",
        "stage": "stage6_reoptimization",
        "run_type": "single_objective_optimization",
        "network": spec.network,
        "simulator_seed": int(spec.simulator_seed),
        "iterations": int(spec.iterations),
        "method": spec.method,
        "optimizer_replicate": int(spec.optimizer_replicate),
        "optimizer_seed": int(spec.optimizer_seed),
        "trials": int(spec.trials),
        "startup_trials": (
            int(spec.startup_trials) if spec.method == "bo_gp" else None
        ),
        "raw_level": spec.raw_level,
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_DEFINITION_VERSION,
        "objective_direction": "minimize",
        "application_precision": 4,
    }
    errors = [
        f"{field}={observed[field]!r}, expected={value!r}"
        for field, value in expected.items()
        if observed[field] != value
    ]

    counts = manifest.get("counts", {})
    expected_counts = {
        "complete": int(spec.trials),
        "failed": 0,
        "pruned": 0,
    }
    if counts != expected_counts:
        errors.append(f"counts={counts!r}, expected={expected_counts!r}")

    bounds = intervention.get("parameter_bounds", {})
    for name in ("certainty", "effectiveness"):
        values = bounds.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or not _float_matches(values[0], 0.5, tolerance)
            or not _float_matches(values[1], 1.0, tolerance)
        ):
            errors.append(f"invalid {name} parameter bounds: {values!r}")
    return errors


def _metric_errors(
    saved: pd.DataFrame,
    recalculated: pd.DataFrame,
    *,
    expected_iterations: int,
    tolerance: float,
) -> list[str]:
    errors: list[str] = []
    if len(saved) != expected_iterations:
        return [
            f"metrics row count={len(saved)}, expected={expected_iterations}"
        ]
    if "num_iter" not in saved.columns:
        return ["saved metrics have no num_iter column"]
    observed_ids = set(pd.to_numeric(saved["num_iter"], errors="coerce").dropna())
    if observed_ids != set(range(expected_iterations)):
        return ["metrics iteration IDs do not match the expected range"]

    left = saved.sort_values("num_iter").reset_index(drop=True)
    right = recalculated.sort_values("num_iter").reset_index(drop=True)
    missing = [column for column in right.columns if column not in left.columns]
    if missing:
        errors.append(f"saved metrics are missing columns: {missing}")
        return errors
    for column in right.columns:
        if column == "num_iter":
            continue
        observed = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        expected = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if not np.allclose(
            observed,
            expected,
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        ):
            errors.append(f"saved metric does not reproduce: {column}")
    return errors


def _validate_trial_parameters(
    row: pd.Series,
    opinion_path: Path,
    *,
    tolerance: float,
) -> list[str]:
    errors: list[str] = []
    proposed: dict[str, float] = {}
    applied: dict[str, float] = {}
    for name in ("certainty", "effectiveness"):
        proposed_value = pd.to_numeric(
            pd.Series([row[f"proposed_{name}"]]), errors="coerce"
        ).iloc[0]
        applied_value = pd.to_numeric(
            pd.Series([row[f"applied_{name}"]]), errors="coerce"
        ).iloc[0]
        if pd.isna(proposed_value) or not 0.5 <= float(proposed_value) <= 1.0:
            errors.append(f"invalid proposed_{name}={proposed_value!r}")
            continue
        if pd.isna(applied_value) or not 0.5 <= float(applied_value) <= 1.0:
            errors.append(f"invalid applied_{name}={applied_value!r}")
            continue
        proposed[name] = float(proposed_value)
        applied[name] = float(applied_value)
        if not _float_matches(
            applied[name], round(proposed[name], 4), tolerance
        ):
            errors.append(
                f"applied_{name}={applied[name]!r} does not match rounded "
                f"proposal={proposed[name]!r}"
            )

    if errors:
        return errors
    _, opinion_applied = read_intervention_opinion_csv(opinion_path)
    for name in ("certainty", "effectiveness"):
        if not _float_matches(opinion_applied[name], applied[name], tolerance):
            errors.append(
                f"opinion {name}={opinion_applied[name]!r} does not match "
                f"trial applied value={applied[name]!r}"
            )
    return errors


def load_reoptimization(
    experiment_root: str | Path,
    *,
    expected_specs: Sequence[Any],
    numeric_tolerance: float = 1e-12,
) -> ReoptimizationData:
    """Load every expected run and reproduce all trial objectives from raw data."""

    root = Path(experiment_root).resolve()
    if numeric_tolerance < 0 or not math.isfinite(numeric_tolerance):
        raise ValueError("numeric_tolerance must be finite and non-negative")

    trial_frames: list[pd.DataFrame] = []
    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in expected_specs:
        run_dir = root / spec.relative_run_dir
        paths = {
            "manifest": run_dir / "manifest.json",
            "trials": run_dir / "trials.csv",
            "summary": run_dir / "summary.json",
            "study": run_dir / "study.db",
            "runtime": run_dir / "runtime.toml",
        }
        audit: dict[str, Any] = {
            "run_key": spec.key,
            "network": spec.network,
            "method": spec.method,
            "optimizer_replicate": int(spec.optimizer_replicate),
            "expected_trials": int(spec.trials),
            "files_present": False,
            "manifest_valid": False,
            "trial_table_valid": False,
            "raw_policy_valid": False,
            "objectives_reproduced": False,
            "valid": False,
            "error": None,
        }
        try:
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing run files: {missing}")
            audit["files_present"] = True

            manifest = _read_json(paths["manifest"])
            summary = _read_json(paths["summary"])
            manifest_errors = _manifest_errors(
                manifest,
                spec,
                tolerance=numeric_tolerance,
            )
            if manifest_errors:
                raise ValueError("; ".join(manifest_errors))
            audit["manifest_valid"] = True

            trials = pd.read_csv(paths["trials"])
            required_columns = {
                "trial",
                "state",
                "value",
                "proposed_certainty",
                "proposed_effectiveness",
                "applied_certainty",
                "applied_effectiveness",
                OBJECTIVE_NAME,
                "simulation_sec",
                "metric_calculation_sec",
                "trial_total_sec",
                "raw_dir",
            }
            missing_columns = sorted(required_columns - set(trials.columns))
            if missing_columns:
                raise ValueError(f"trials.csv is missing columns: {missing_columns}")
            if len(trials) != int(spec.trials):
                raise ValueError(
                    f"trial row count={len(trials)}, expected={spec.trials}"
                )
            observed_trials = set(
                pd.to_numeric(trials["trial"], errors="raise").astype(int)
            )
            if observed_trials != set(range(int(spec.trials))):
                raise ValueError("trial IDs do not match the expected range")
            if set(trials["state"].astype(str)) != {"COMPLETE"}:
                raise ValueError("trials.csv contains a non-COMPLETE trial")
            audit["trial_table_valid"] = True

            summary_expected = {
                "experiment_id": manifest.get("experiment_id"),
                "network": spec.network,
                "method": spec.method,
                "optimizer_replicate": int(spec.optimizer_replicate),
                "optimizer_seed": int(spec.optimizer_seed),
                "simulator_seed": int(spec.simulator_seed),
                "iteration_count": int(spec.iterations),
                "objective_name": OBJECTIVE_NAME,
                "objective_definition_version": OBJECTIVE_DEFINITION_VERSION,
                "n_trials_requested": int(spec.trials),
                "n_trials_recorded": int(spec.trials),
                "n_trials_complete": int(spec.trials),
                "n_trials_failed": 0,
                "n_trials_pruned": 0,
            }
            summary_errors = [
                f"summary {field}={summary.get(field)!r}, expected={value!r}"
                for field, value in summary_expected.items()
                if summary.get(field) != value
            ]
            if summary_errors:
                raise ValueError("; ".join(summary_errors))

            num_agents = int(NETWORKS[spec.network].num_agents)
            validated_trial_rows: list[dict[str, Any]] = []
            for _, row in trials.sort_values("trial").iterrows():
                trial_number = int(row["trial"])
                relative_raw = Path(str(row["raw_dir"]))
                trial_dir = (run_dir / relative_raw).resolve()
                if run_dir.resolve() not in trial_dir.parents:
                    raise ValueError(
                        f"trial {trial_number}: raw_dir escapes the run directory"
                    )
                trial_paths = {
                    "pop": trial_dir / "pop.arrow",
                    "metrics": trial_dir / "metrics.csv",
                    "metrics_summary": trial_dir / "metrics_summary.json",
                    "opinion": trial_dir / "inhibition_opinion.csv",
                    "strategy": trial_dir / "strategy.toml",
                }
                missing_trial = [
                    name for name, path in trial_paths.items() if not path.is_file()
                ]
                if missing_trial:
                    raise FileNotFoundError(
                        f"trial {trial_number}: missing files {missing_trial}"
                    )
                unexpected_raw = [
                    name
                    for name in ("info.arrow", "agent.arrow")
                    if (trial_dir / name).exists()
                ]
                if unexpected_raw:
                    raise ValueError(
                        f"trial {trial_number}: raw_level=pop retained "
                        f"unexpected files {unexpected_raw}"
                    )

                parameter_errors = _validate_trial_parameters(
                    row,
                    trial_paths["opinion"],
                    tolerance=numeric_tolerance,
                )
                if parameter_errors:
                    raise ValueError(
                        f"trial {trial_number}: " + "; ".join(parameter_errors)
                    )

                result = compute_selfish_metrics_from_arrow(
                    trial_paths["pop"],
                    num_agents=num_agents,
                    expected_iterations=int(spec.iterations),
                )
                saved_metrics = pd.read_csv(trial_paths["metrics"])
                metric_errors = _metric_errors(
                    saved_metrics,
                    result.per_iteration,
                    expected_iterations=int(spec.iterations),
                    tolerance=numeric_tolerance,
                )
                metrics_summary = _read_json(trial_paths["metrics_summary"])
                for field, value in (
                    ("value", row["value"]),
                    (OBJECTIVE_NAME, row[OBJECTIVE_NAME]),
                    (
                        "metrics_summary objective",
                        metrics_summary.get(OBJECTIVE_NAME),
                    ),
                ):
                    if not _float_matches(
                        value, result.objective_value, numeric_tolerance
                    ):
                        metric_errors.append(
                            f"{field} does not match raw objective"
                        )
                if metric_errors:
                    raise ValueError(
                        f"trial {trial_number}: " + "; ".join(metric_errors)
                    )

                trial_timing: dict[str, float] = {}
                for field in (
                    "simulation_sec",
                    "metric_calculation_sec",
                    "trial_total_sec",
                ):
                    value = float(row[field])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(
                            f"trial {trial_number}: invalid {field}={value!r}"
                        )
                    trial_timing[field] = value

                identifiers = {
                    "run_key": spec.key,
                    "network": spec.network,
                    "method": spec.method,
                    "optimizer_replicate": int(spec.optimizer_replicate),
                    "optimizer_seed": int(spec.optimizer_seed),
                    "simulator_seed": int(spec.simulator_seed),
                    "trial": trial_number,
                    "evaluation": trial_number + 1,
                    "proposed_certainty": float(row["proposed_certainty"]),
                    "proposed_effectiveness": float(row["proposed_effectiveness"]),
                    "applied_certainty": float(row["applied_certainty"]),
                    "applied_effectiveness": float(row["applied_effectiveness"]),
                }
                iteration_metrics = result.per_iteration.copy()
                for position, (field, value) in enumerate(identifiers.items()):
                    iteration_metrics.insert(position, field, value)
                iteration_frames.append(iteration_metrics)

                validated_trial_rows.append(
                    {
                        **identifiers,
                        "state": "COMPLETE",
                        "value": result.objective_value,
                        OBJECTIVE_NAME: result.objective_value,
                        "peak_new_selfish_ratio": result.summary[
                            "peak_new_selfish_ratio"
                        ],
                        "mean_new_selfish_rate_per_step": result.summary[
                            "mean_new_selfish_rate_per_step"
                        ],
                        "n_iterations": result.summary["n_iterations"],
                        "n_zero_selfish_iterations": result.summary[
                            "n_zero_selfish_iterations"
                        ],
                        **trial_timing,
                        "raw_dir": relative_raw.as_posix(),
                        "pop_bytes": trial_paths["pop"].stat().st_size,
                    }
                )

            audit["raw_policy_valid"] = True
            audit["objectives_reproduced"] = True
            validated_trials = pd.DataFrame(validated_trial_rows)
            trial_frames.append(validated_trials)

            best_position = validated_trials["value"].idxmin()
            best = validated_trials.loc[best_position]
            manifest_best = manifest.get("best") or {}
            summary_best = summary.get("best") or {}
            for label, recorded in (
                ("manifest", manifest_best),
                ("summary", summary_best),
            ):
                if int(recorded.get("trial", -1)) != int(best["trial"]):
                    raise ValueError(f"{label} best trial does not reproduce")
                if not _float_matches(
                    recorded.get("value"), best["value"], numeric_tolerance
                ):
                    raise ValueError(f"{label} best value does not reproduce")

                recorded_proposed = recorded.get("proposed_parameters") or {}
                recorded_applied = recorded.get("applied_parameters") or {}
                for field in ("certainty", "effectiveness"):
                    if not _float_matches(
                        recorded_proposed.get(field),
                        best[f"proposed_{field}"],
                        numeric_tolerance,
                    ):
                        raise ValueError(
                            f"{label} best proposed {field} does not reproduce"
                        )
                    if not _float_matches(
                        recorded_applied.get(field),
                        best[f"applied_{field}"],
                        numeric_tolerance,
                    ):
                        raise ValueError(
                            f"{label} best applied {field} does not reproduce"
                        )

            timing = manifest.get("timing_sec", {})
            summary_timing = summary.get("timing_sec", {})
            validated_timing: dict[str, float] = {}
            for field in (
                "optimization_total",
                "simulation_total",
                "metric_calculation_total",
                "optimizer_overhead",
            ):
                value = float(timing[field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid run timing {field}={value!r}")
                if not _float_matches(
                    summary_timing.get(field), value, numeric_tolerance
                ):
                    raise ValueError(
                        f"summary run timing does not match manifest: {field}"
                    )
                validated_timing[field] = value
            run_rows.append(
                {
                    "run_key": spec.key,
                    "network": spec.network,
                    "method": spec.method,
                    "optimizer_replicate": int(spec.optimizer_replicate),
                    "optimizer_seed": int(spec.optimizer_seed),
                    "simulator_seed": int(spec.simulator_seed),
                    "iterations_per_evaluation": int(spec.iterations),
                    "evaluations": int(spec.trials),
                    "startup_trials": (
                        int(spec.startup_trials) if spec.method == "bo_gp" else None
                    ),
                    "raw_level": spec.raw_level,
                    "optimization_total_sec": validated_timing[
                        "optimization_total"
                    ],
                    "simulation_total_sec": validated_timing["simulation_total"],
                    "metric_calculation_total_sec": validated_timing[
                        "metric_calculation_total"
                    ],
                    "optimizer_overhead_sec": validated_timing[
                        "optimizer_overhead"
                    ],
                    "run_size_bytes": _directory_size_bytes(run_dir),
                    "manifest_path": paths["manifest"].as_posix(),
                    "trials_path": paths["trials"].as_posix(),
                }
            )
            audit["valid"] = True
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"{spec.key}: {audit['error']}")
        audit_rows.append(audit)

    if failures:
        preview = "\n".join(failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n... {len(failures) - 20} more"
        raise ValueError(
            "Stage 6 reoptimization data failed formal validation:\n"
            + preview
            + suffix
        )
    if not trial_frames or not iteration_frames:
        raise ValueError("Stage 6 reoptimization contains no validated data")

    trials = pd.concat(trial_frames, ignore_index=True)
    iterations = pd.concat(iteration_frames, ignore_index=True)
    expected_trials = sum(int(spec.trials) for spec in expected_specs)
    expected_iterations = sum(
        int(spec.trials) * int(spec.iterations) for spec in expected_specs
    )
    if len(trials) != expected_trials:
        raise ValueError("validated Stage 6 trial count is inconsistent")
    if len(iterations) != expected_iterations:
        raise ValueError("validated Stage 6 iteration count is inconsistent")
    return ReoptimizationData(
        trials=trials.sort_values(["run_key", "trial"]).reset_index(drop=True),
        iterations=iterations.sort_values(
            ["run_key", "trial", "num_iter"]
        ).reset_index(drop=True),
        runs=pd.DataFrame(run_rows).sort_values("run_key").reset_index(drop=True),
        audit=pd.DataFrame(audit_rows).sort_values("run_key").reset_index(drop=True),
    )


def build_best_so_far(trials: pd.DataFrame) -> pd.DataFrame:
    """Return one row per trial with the incumbent objective after that trial."""

    data = trials.sort_values(["run_key", "trial"], kind="stable").copy()
    data["evaluation"] = data["trial"].astype(int) + 1
    data["best_so_far"] = data.groupby("run_key", sort=False)["value"].cummin()
    previous = data.groupby("run_key", sort=False)["best_so_far"].shift(1)
    data["incumbent_changed"] = previous.isna() | (
        data["best_so_far"] < previous
    )
    return data


def build_convergence_summary(best_so_far: pd.DataFrame) -> pd.DataFrame:
    """Summarize best-so-far curves across the six optimizer replicates."""

    return (
        best_so_far.groupby(["network", "method", "evaluation"], sort=True)[
            "best_so_far"
        ]
        .agg(
            n_runs="count",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )


def build_run_final_summary(
    trials: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """Describe the final incumbent and when it first appeared in every run."""

    rows: list[dict[str, Any]] = []
    for run_key, group in trials.groupby("run_key", sort=True):
        ordered = group.sort_values("trial", kind="stable")
        final_best = float(ordered["value"].min())
        best = ordered.loc[ordered["value"].idxmin()]
        rows.append(
            {
                "run_key": run_key,
                "network": best["network"],
                "method": best["method"],
                "optimizer_replicate": int(best["optimizer_replicate"]),
                "optimizer_seed": int(best["optimizer_seed"]),
                "simulator_seed": int(best["simulator_seed"]),
                "final_best": final_best,
                "best_trial": int(best["trial"]),
                "first_final_best_evaluation": int(best["trial"]) + 1,
                "proposed_certainty": float(best["proposed_certainty"]),
                "proposed_effectiveness": float(best["proposed_effectiveness"]),
                "applied_certainty": float(best["applied_certainty"]),
                "applied_effectiveness": float(best["applied_effectiveness"]),
                "peak_new_selfish_ratio": float(best["peak_new_selfish_ratio"]),
                "mean_new_selfish_rate_per_step": float(
                    best["mean_new_selfish_rate_per_step"]
                ),
                "raw_dir": str(best["raw_dir"]),
            }
        )
    final = pd.DataFrame(rows)
    timing_columns = [
        "run_key",
        "optimization_total_sec",
        "simulation_total_sec",
        "metric_calculation_total_sec",
        "optimizer_overhead_sec",
        "run_size_bytes",
    ]
    return final.merge(
        runs.loc[:, timing_columns],
        on="run_key",
        how="left",
        validate="one_to_one",
    )


def build_method_summary(run_final: pd.DataFrame) -> pd.DataFrame:
    """Summarize final search values without declaring an inferential winner."""

    return (
        run_final.groupby(["network", "method"], sort=True)
        .agg(
            n_runs=("run_key", "count"),
            final_best_median=("final_best", "median"),
            final_best_q1=("final_best", lambda values: values.quantile(0.25)),
            final_best_q3=("final_best", lambda values: values.quantile(0.75)),
            final_best_minimum=("final_best", "min"),
            final_best_maximum=("final_best", "max"),
            median_first_final_best_evaluation=(
                "first_final_best_evaluation",
                "median",
            ),
        )
        .reset_index()
    )


def build_timing_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep simulation, metric calculation, and optimizer overhead separate."""

    rows: list[dict[str, Any]] = []
    columns = (
        "optimization_total_sec",
        "simulation_total_sec",
        "metric_calculation_total_sec",
        "optimizer_overhead_sec",
        "run_size_bytes",
    )
    for (network, method), group in runs.groupby(
        ["network", "method"], sort=True
    ):
        row: dict[str, Any] = {
            "network": network,
            "method": method,
            "n_runs": int(len(group)),
        }
        for column in columns:
            values = pd.to_numeric(group[column], errors="raise")
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_q1"] = float(values.quantile(0.25))
            row[f"{column}_q3"] = float(values.quantile(0.75))
            row[f"{column}_sum"] = float(values.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_candidate_pool(run_final: pd.DataFrame) -> pd.DataFrame:
    """Preserve the best candidate and provenance from every optimization run."""

    columns = [
        "run_key",
        "network",
        "method",
        "optimizer_replicate",
        "optimizer_seed",
        "simulator_seed",
        "final_best",
        "best_trial",
        "first_final_best_evaluation",
        "proposed_certainty",
        "proposed_effectiveness",
        "applied_certainty",
        "applied_effectiveness",
        "peak_new_selfish_ratio",
        "mean_new_selfish_rate_per_step",
        "raw_dir",
    ]
    return run_final.loc[:, columns].sort_values(
        ["network", "method", "optimizer_replicate"]
    ).reset_index(drop=True)


def build_unique_candidate_pool(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate applied candidates while retaining all source run keys."""

    rows: list[dict[str, Any]] = []
    keys = ["network", "applied_certainty", "applied_effectiveness"]
    for values, group in candidate_pool.groupby(keys, sort=True, dropna=False):
        network, certainty, effectiveness = values
        rows.append(
            {
                "network": network,
                "applied_certainty": float(certainty),
                "applied_effectiveness": float(effectiveness),
                "source_count": int(len(group)),
                "search_value_minimum": float(group["final_best"].min()),
                "search_value_median": float(group["final_best"].median()),
                "source_methods": ";".join(sorted(set(group["method"]))),
                "source_run_keys": ";".join(sorted(group["run_key"])),
            }
        )
    return pd.DataFrame(rows)


def build_stage6_decision(
    run_final: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    unique_candidate_pool: pd.DataFrame,
) -> dict[str, Any]:
    """Record readiness for Stage 7 without selecting or validating a winner."""

    networks: list[dict[str, Any]] = []
    for network, group in run_final.groupby("network", sort=True):
        candidates = candidate_pool[candidate_pool["network"] == network]
        unique = unique_candidate_pool[
            unique_candidate_pool["network"] == network
        ]
        networks.append(
            {
                "network": network,
                "completed_runs": int(len(group)),
                "candidate_rows": int(len(candidates)),
                "unique_applied_candidates": int(len(unique)),
                "exploration_best": float(group["final_best"].min()),
            }
        )
    return {
        "status": "candidate_pool_ready",
        "stage6_complete": True,
        "candidate_selection_complete": False,
        "effect_claim_allowed": False,
        "next_stage": "independent_candidate_validation",
        "reason": (
            "Stage 6 used one exploration simulator-seed block. Preserve the "
            "run-level candidate pool and validate candidates on unused seeds "
            "before selecting a final condition or claiming suppression."
        ),
        "networks": networks,
    }
