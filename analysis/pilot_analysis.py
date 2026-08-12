"""Analysis helpers for Stage 3 precision and optimization-budget pilots."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OBJECTIVE_COLUMN = "cumulative_selfish_fraction"
OBJECTIVE_DEFINITION_VERSION = "cumulative_selfish_fraction_v1"


@dataclass(frozen=True)
class PrecisionPilotData:
    iterations: pd.DataFrame
    runs: pd.DataFrame


@dataclass(frozen=True)
class OptimizationPilotData:
    trials: pd.DataFrame
    runs: pd.DataFrame


def directory_size_bytes(path: str | Path) -> int:
    path = Path(path)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_mismatches(
    manifest: dict[str, Any],
    spec: Any,
    *,
    expected_run_type: str,
) -> list[str]:
    mismatches: list[str] = []
    expected = {
        "stage": "stage3_pilot",
        "run_type": expected_run_type,
        "network": spec.network,
        "simulator_seed": int(spec.simulator_seed),
        "iterations": int(spec.iterations),
        "objective_name": OBJECTIVE_COLUMN,
        "objective_version": OBJECTIVE_DEFINITION_VERSION,
    }
    observed = {
        "stage": manifest.get("stage"),
        "run_type": manifest.get("run_type"),
        "network": manifest.get("network", {}).get("id"),
        "simulator_seed": manifest.get("runtime", {}).get("simulator_seed"),
        "iterations": manifest.get("runtime", {}).get("iteration_count"),
        "objective_name": manifest.get("objective", {}).get("name"),
        "objective_version": manifest.get("objective", {}).get(
            "definition_version"
        ),
    }
    for field, expected_value in expected.items():
        if observed[field] != expected_value:
            mismatches.append(
                f"{field}={observed[field]!r} (expected {expected_value!r})"
            )
    return mismatches


def _fixed_condition_mismatches(
    manifest: dict[str, Any], spec: Any
) -> list[str]:
    intervention = manifest.get("intervention", {})
    mismatches: list[str] = []
    if intervention.get("condition_id") != spec.condition_id:
        mismatches.append(
            "condition_id="
            f"{intervention.get('condition_id')!r} (expected {spec.condition_id!r})"
        )
    if intervention.get("enabled") is not bool(spec.intervention_enabled):
        mismatches.append(
            f"intervention_enabled={intervention.get('enabled')!r} "
            f"(expected {bool(spec.intervention_enabled)!r})"
        )
    if spec.intervention_enabled:
        applied = intervention.get("applied_parameters") or {}
        for field in ("certainty", "effectiveness"):
            observed = applied.get(field)
            expected = round(float(getattr(spec, field)), 4)
            if observed is None or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                mismatches.append(
                    f"applied_{field}={observed!r} (expected {expected!r})"
                )
    return mismatches


def _optimization_mismatches(
    manifest: dict[str, Any], spec: Any
) -> list[str]:
    optimization = manifest.get("optimization", {})
    expected = {
        "method": spec.method,
        "optimizer_replicate": int(spec.optimizer_replicate),
        "optimizer_seed": int(spec.optimizer_seed),
        "n_trials_requested": int(spec.trials),
    }
    mismatches: list[str] = []
    for field, expected_value in expected.items():
        observed = optimization.get(field)
        if observed != expected_value:
            mismatches.append(
                f"{field}={observed!r} (expected {expected_value!r})"
            )
    expected_startup = int(spec.startup_trials) if spec.method == "bo_gp" else None
    if optimization.get("startup_trials") != expected_startup:
        mismatches.append(
            f"startup_trials={optimization.get('startup_trials')!r} "
            f"(expected {expected_startup!r})"
        )
    return mismatches


def load_precision_pilot(
    experiment_root: str | Path,
    *,
    expected_specs: Sequence[Any],
) -> PrecisionPilotData:
    experiment_root = Path(experiment_root).resolve()
    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for spec in expected_specs:
        run_dir = experiment_root / spec.relative_run_dir
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.csv"
        if not manifest_path.is_file() or not metrics_path.is_file():
            errors.append(f"missing completed output for {spec.key}")
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed":
            errors.append(f"run is not completed: {spec.key}")
            continue
        mismatches = _manifest_mismatches(
            manifest, spec, expected_run_type="fixed_condition"
        )
        mismatches.extend(_fixed_condition_mismatches(manifest, spec))
        if mismatches:
            errors.append(
                f"manifest does not match protocol for {spec.key}: "
                + "; ".join(mismatches)
            )
            continue
        metrics = pd.read_csv(metrics_path)
        if "num_iter" not in metrics or OBJECTIVE_COLUMN not in metrics:
            errors.append(f"required metrics columns are missing for {spec.key}")
            continue
        expected_iterations = int(spec.iterations)
        observed = set(pd.to_numeric(metrics["num_iter"]).astype(int))
        expected = set(range(expected_iterations))
        if observed != expected or len(metrics) != expected_iterations:
            errors.append(f"iteration IDs do not match for {spec.key}")
            continue
        objective_values = pd.to_numeric(
            metrics[OBJECTIVE_COLUMN], errors="coerce"
        )
        if objective_values.isna().any():
            errors.append(f"objective values are invalid for {spec.key}")
            continue
        manifest_objective = manifest.get("objective", {}).get("value")
        if manifest_objective is None or not math.isclose(
            float(manifest_objective),
            float(objective_values.mean()),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"manifest objective does not match metrics for {spec.key}")
            continue

        metrics.insert(0, "simulator_seed", int(spec.simulator_seed))
        metrics.insert(0, "condition_id", str(spec.condition_id))
        metrics.insert(0, "network", spec.network)
        metrics.insert(0, "run_key", spec.key)
        iteration_frames.append(metrics)

        output_sizes = {
            kind: (run_dir / f"{kind}.arrow").stat().st_size
            if (run_dir / f"{kind}.arrow").is_file()
            else 0
            for kind in ("pop", "info", "agent")
        }
        run_rows.append(
            {
                "run_key": spec.key,
                "network": spec.network,
                "condition_id": spec.condition_id,
                "simulator_seed": int(spec.simulator_seed),
                "iterations": expected_iterations,
                "certainty": spec.certainty,
                "effectiveness": spec.effectiveness,
                "objective": float(manifest["objective"]["value"]),
                "simulation_sec": float(manifest["timing_sec"]["simulation"]),
                "total_sec": float(manifest["timing_sec"]["total"]),
                "run_size_bytes": directory_size_bytes(run_dir),
                "pop_bytes": output_sizes["pop"],
                "info_bytes": output_sizes["info"],
                "agent_bytes": output_sizes["agent"],
                "manifest_path": manifest_path.as_posix(),
                "metrics_path": metrics_path.as_posix(),
            }
        )

    if errors:
        raise ValueError("precision pilot is incomplete:\n" + "\n".join(errors))
    return PrecisionPilotData(
        iterations=pd.concat(iteration_frames, ignore_index=True),
        runs=pd.DataFrame(run_rows),
    )


def load_optimization_pilot(
    experiment_root: str | Path,
    *,
    expected_specs: Sequence[Any],
) -> OptimizationPilotData:
    experiment_root = Path(experiment_root).resolve()
    trial_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for spec in expected_specs:
        run_dir = experiment_root / spec.relative_run_dir
        manifest_path = run_dir / "manifest.json"
        trials_path = run_dir / "trials.csv"
        if not manifest_path.is_file() or not trials_path.is_file():
            errors.append(f"missing completed output for {spec.key}")
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed":
            errors.append(f"run is not completed: {spec.key}")
            continue
        mismatches = _manifest_mismatches(
            manifest,
            spec,
            expected_run_type="single_objective_optimization",
        )
        mismatches.extend(_optimization_mismatches(manifest, spec))
        if mismatches:
            errors.append(
                f"manifest does not match protocol for {spec.key}: "
                + "; ".join(mismatches)
            )
            continue
        trials = pd.read_csv(trials_path)
        required_columns = {"trial", "state", "value"}
        if not required_columns.issubset(trials.columns):
            errors.append(f"required trial columns are missing for {spec.key}")
            continue
        complete = trials[trials["state"] == "COMPLETE"].copy()
        if len(complete) != int(spec.trials):
            errors.append(f"complete trial count does not match for {spec.key}")
            continue
        observed_trials = set(pd.to_numeric(complete["trial"]).astype(int))
        if observed_trials != set(range(int(spec.trials))):
            errors.append(f"trial IDs do not match for {spec.key}")
            continue
        values = pd.to_numeric(complete["value"], errors="coerce")
        if values.isna().any():
            errors.append(f"objective values are invalid for {spec.key}")
            continue
        complete.insert(0, "optimizer_seed", int(spec.optimizer_seed))
        complete.insert(0, "optimizer_replicate", int(spec.optimizer_replicate))
        complete.insert(0, "method", str(spec.method))
        complete.insert(0, "network", spec.network)
        complete.insert(0, "run_key", spec.key)
        trial_frames.append(complete)
        run_rows.append(
            {
                "run_key": spec.key,
                "network": spec.network,
                "method": spec.method,
                "optimizer_replicate": int(spec.optimizer_replicate),
                "optimizer_seed": int(spec.optimizer_seed),
                "simulator_seed": int(spec.simulator_seed),
                "iterations": int(spec.iterations),
                "trials": int(spec.trials),
                "optimization_total_sec": float(
                    manifest["timing_sec"]["optimization_total"]
                ),
                "simulation_total_sec": float(
                    manifest["timing_sec"]["simulation_total"]
                ),
                "optimizer_overhead_sec": float(
                    manifest["timing_sec"]["optimizer_overhead"]
                ),
                "run_size_bytes": directory_size_bytes(run_dir),
                "manifest_path": manifest_path.as_posix(),
                "trials_path": trials_path.as_posix(),
            }
        )

    if errors:
        raise ValueError("optimization pilot is incomplete:\n" + "\n".join(errors))
    return OptimizationPilotData(
        trials=pd.concat(trial_frames, ignore_index=True),
        runs=pd.DataFrame(run_rows),
    )


def _hierarchical_bootstrap_means(
    values_by_block: Sequence[np.ndarray],
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = [np.asarray(values, dtype=float) for values in values_by_block]
    if len(blocks) < 2 or any(len(values) < 2 for values in blocks):
        raise ValueError(
            "hierarchical bootstrap requires at least two nontrivial blocks"
        )
    n_blocks = len(blocks)
    results = np.zeros(repetitions, dtype=float)
    for position in range(n_blocks):
        selected_blocks = rng.integers(0, n_blocks, size=repetitions)
        contribution = np.empty(repetitions, dtype=float)
        for block_index, block in enumerate(blocks):
            rows = np.flatnonzero(selected_blocks == block_index)
            if len(rows) == 0:
                continue
            indices = rng.integers(0, len(block), size=(len(rows), len(block)))
            contribution[rows] = block[indices].mean(axis=1)
        results += contribution / n_blocks
    return results


def hierarchical_mean_interval(
    frame: pd.DataFrame,
    *,
    block_column: str,
    value_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    data = frame[[block_column, value_column]].dropna()
    blocks = [
        group[value_column].to_numpy(dtype=float)
        for _, group in data.groupby(block_column, sort=True)
    ]
    estimate = float(np.mean([values.mean() for values in blocks]))
    bootstrapped = _hierarchical_bootstrap_means(
        blocks,
        repetitions=repetitions,
        rng=np.random.default_rng(seed),
    )
    low, high = np.quantile(bootstrapped, [0.025, 0.975])
    return {
        "n_blocks": len(blocks),
        "n_per_block": min(len(values) for values in blocks),
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_half_width": float((high - low) / 2),
    }


def build_prefix_estimates(
    iterations: pd.DataFrame, *, prefixes: Sequence[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["network", "condition_id", "simulator_seed"]
    for prefix in prefixes:
        subset = iterations[iterations["num_iter"] < int(prefix)]
        for key, group in subset.groupby(keys, sort=True):
            values = group[OBJECTIVE_COLUMN]
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "prefix_iterations": int(prefix),
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "sem": float(values.sem()),
                }
            )
    return pd.DataFrame(rows)


def build_condition_precision(
    iterations: pd.DataFrame,
    *,
    prefixes: Sequence[int],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    for prefix in prefixes:
        subset = iterations[iterations["num_iter"] < int(prefix)]
        for (network, condition), group in subset.groupby(
            ["network", "condition_id"], sort=True
        ):
            result = hierarchical_mean_interval(
                group,
                block_column="simulator_seed",
                value_column=OBJECTIVE_COLUMN,
                repetitions=repetitions,
                seed=seed + offset,
            )
            block_means = group.groupby("simulator_seed")[OBJECTIVE_COLUMN].mean()
            rows.append(
                {
                    "network": network,
                    "condition_id": condition,
                    "prefix_iterations": int(prefix),
                    "between_block_sd": float(block_means.std()),
                    **result,
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def build_paired_differences(
    iterations: pd.DataFrame,
    *,
    prefixes: Sequence[int],
    comparisons: Sequence[dict[str, str]],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    wide = iterations.pivot(
        index=["network", "simulator_seed", "num_iter"],
        columns="condition_id",
        values=OBJECTIVE_COLUMN,
    ).reset_index()
    rows: list[dict[str, Any]] = []
    offset = 0
    for prefix in prefixes:
        prefix_data = wide[wide["num_iter"] < int(prefix)]
        for comparison in comparisons:
            reference = comparison["reference"]
            candidate = comparison["candidate"]
            if reference not in prefix_data or candidate not in prefix_data:
                raise ValueError(f"comparison condition is missing: {comparison['id']}")
            data = prefix_data[
                ["network", "simulator_seed", reference, candidate]
            ].copy()
            data["difference"] = data[reference] - data[candidate]
            for network, group in data.groupby("network", sort=True):
                result = hierarchical_mean_interval(
                    group,
                    block_column="simulator_seed",
                    value_column="difference",
                    repetitions=repetitions,
                    seed=seed + 10000 + offset,
                )
                rows.append(
                    {
                        "network": network,
                        "comparison_id": comparison["id"],
                        "reference": reference,
                        "candidate": candidate,
                        "prefix_iterations": int(prefix),
                        "difference_definition": "reference_minus_candidate",
                        **result,
                    }
                )
                offset += 1
    return pd.DataFrame(rows)


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return math.nan
    return float(spearmanr(frame["x"], frame["y"]).statistic)


def build_rank_stability(
    iterations: pd.DataFrame, *, prefixes: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_prefix = max(prefixes)
    rows: list[dict[str, Any]] = []
    for network, network_data in iterations.groupby("network", sort=True):
        seeds = sorted(network_data["simulator_seed"].unique())
        for seed in seeds:
            reference = (
                network_data[
                    (network_data["simulator_seed"] != seed)
                    & (network_data["num_iter"] < max_prefix)
                ]
                .groupby("condition_id")[OBJECTIVE_COLUMN]
                .mean()
            )
            reference_best = str(reference.idxmin())
            for prefix in prefixes:
                estimate = (
                    network_data[
                        (network_data["simulator_seed"] == seed)
                        & (network_data["num_iter"] < int(prefix))
                    ]
                    .groupby("condition_id")[OBJECTIVE_COLUMN]
                    .mean()
                )
                common = reference.index.intersection(estimate.index)
                estimated_best = str(estimate.idxmin())
                rows.append(
                    {
                        "network": network,
                        "simulator_seed": int(seed),
                        "prefix_iterations": int(prefix),
                        "rank_spearman": _safe_spearman(
                            estimate.loc[common], reference.loc[common]
                        ),
                        "estimated_best_condition": estimated_best,
                        "reference_best_condition": reference_best,
                        "best_condition_matches": estimated_best == reference_best,
                        "reference_definition": (
                            "other_four_blocks_at_max_prefix"
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["network", "prefix_iterations"], sort=True)
        .agg(
            median_rank_spearman=("rank_spearman", "median"),
            min_rank_spearman=("rank_spearman", "min"),
            best_condition_agreement=("best_condition_matches", "mean"),
            n_blocks=("simulator_seed", "nunique"),
        )
        .reset_index()
    )
    return detail, summary


def recommend_iteration_count(
    paired_differences: pd.DataFrame,
    rank_summary: pd.DataFrame,
    *,
    prefixes: Sequence[int],
    delta_min: float | None,
    ci_fraction: float,
    minimum_median_rank_spearman: float,
    minimum_best_condition_agreement: float,
) -> dict[str, Any]:
    if delta_min is None:
        return {
            "status": "delta_min_required",
            "recommended_iterations": None,
            "reason": "Set delta_min with the supervisor before Stage 4.",
        }
    if not math.isfinite(delta_min) or delta_min <= 0:
        raise ValueError("delta_min must be positive")
    threshold = delta_min * ci_fraction
    diagnostics: list[dict[str, Any]] = []
    for prefix in prefixes:
        differences = paired_differences[
            paired_differences["prefix_iterations"] == int(prefix)
        ]
        ranks = rank_summary[rank_summary["prefix_iterations"] == int(prefix)]
        precision_pass = bool((differences["ci_half_width"] <= threshold).all())
        rank_pass = bool(
            (ranks["median_rank_spearman"] >= minimum_median_rank_spearman).all()
            and (
                ranks["best_condition_agreement"]
                >= minimum_best_condition_agreement
            ).all()
        )
        diagnostics.append(
            {
                "prefix_iterations": int(prefix),
                "max_ci_half_width": float(differences["ci_half_width"].max()),
                "ci_threshold": threshold,
                "minimum_median_rank_spearman": float(
                    ranks["median_rank_spearman"].min()
                ),
                "minimum_best_condition_agreement": float(
                    ranks["best_condition_agreement"].min()
                ),
                "precision_pass": precision_pass,
                "rank_pass": rank_pass,
                "passes": precision_pass and rank_pass,
            }
        )
    passing = [row for row in diagnostics if row["passes"]]
    if passing:
        recommendation = passing[0]["prefix_iterations"]
        status = "recommended"
        reason = "Smallest tested prefix satisfying precision and rank rules."
    else:
        recommendation = None
        status = "insufficient_precision"
        reason = "No tested prefix satisfies all pre-specified rules."
    return {
        "status": status,
        "recommended_iterations": recommendation,
        "delta_min": delta_min,
        "diagnostics": diagnostics,
        "reason": reason,
    }


def build_resource_summary(runs: pd.DataFrame) -> pd.DataFrame:
    data = runs.copy()
    for column in (
        "simulation_sec",
        "total_sec",
        "run_size_bytes",
        "pop_bytes",
        "info_bytes",
        "agent_bytes",
    ):
        data[f"{column}_per_iteration"] = data[column] / data["iterations"]
    return (
        data.groupby("network", sort=True)
        .agg(
            n_runs=("run_key", "count"),
            median_simulation_sec_per_iteration=(
                "simulation_sec_per_iteration",
                "median",
            ),
            p90_simulation_sec_per_iteration=(
                "simulation_sec_per_iteration",
                lambda values: values.quantile(0.9),
            ),
            median_total_bytes_per_iteration=(
                "run_size_bytes_per_iteration",
                "median",
            ),
            median_pop_bytes_per_iteration=("pop_bytes_per_iteration", "median"),
            median_info_bytes_per_iteration=("info_bytes_per_iteration", "median"),
            median_agent_bytes_per_iteration=(
                "agent_bytes_per_iteration",
                "median",
            ),
        )
        .reset_index()
    )


def project_future_resources(
    resource_summary: pd.DataFrame,
    *,
    iterations: int,
    optimization_evaluations: int | None,
    fixed_conditions: int = 14,
    fixed_seed_blocks: int = 5,
    optimization_methods: int = 3,
    optimization_replicates: int = 6,
) -> pd.DataFrame:
    """Project Stage 4 and Stage 6 simulator time and retained raw size."""

    iterations = int(iterations)
    rows: list[dict[str, Any]] = []
    for _, network in resource_summary.iterrows():
        fixed_runs = fixed_conditions * fixed_seed_blocks
        rows.append(
            {
                "phase": "stage4_fixed_confirmation",
                "network": network["network"],
                "simulation_evaluations": fixed_runs,
                "iterations_per_evaluation": iterations,
                "projected_simulation_sec": float(
                    network["median_simulation_sec_per_iteration"]
                    * fixed_runs
                    * iterations
                ),
                "projected_retained_raw_bytes": float(
                    (
                        network["median_pop_bytes_per_iteration"]
                        + network["median_info_bytes_per_iteration"]
                        + network["median_agent_bytes_per_iteration"]
                    )
                    * fixed_runs
                    * iterations
                ),
                "raw_policy": "all",
            }
        )
        if optimization_evaluations is not None:
            evaluations = (
                optimization_methods
                * optimization_replicates
                * int(optimization_evaluations)
            )
            rows.append(
                {
                    "phase": "stage6_optimization",
                    "network": network["network"],
                    "simulation_evaluations": evaluations,
                    "iterations_per_evaluation": iterations,
                    "projected_simulation_sec": float(
                        network["median_simulation_sec_per_iteration"]
                        * evaluations
                        * iterations
                    ),
                    "projected_retained_raw_bytes": float(
                        network["median_pop_bytes_per_iteration"]
                        * evaluations
                        * iterations
                    ),
                    "raw_policy": "pop",
                }
            )
    return pd.DataFrame(rows)


def build_best_so_far(
    trials: pd.DataFrame, *, checkpoints: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = trials.sort_values(["run_key", "trial"], kind="stable").copy()
    data["evaluation"] = data["trial"].astype(int) + 1
    data["best_so_far"] = data.groupby("run_key")["value"].cummin()

    checkpoint_rows: list[dict[str, Any]] = []
    for (run_key, method, replicate), group in data.groupby(
        ["run_key", "method", "optimizer_replicate"], sort=True
    ):
        for checkpoint in checkpoints:
            eligible = group[group["evaluation"] <= int(checkpoint)]
            if len(eligible) < int(checkpoint):
                raise ValueError(
                    f"run {run_key} does not reach checkpoint {checkpoint}"
                )
            checkpoint_rows.append(
                {
                    "run_key": run_key,
                    "method": method,
                    "optimizer_replicate": int(replicate),
                    "checkpoint": int(checkpoint),
                    "best_so_far": float(eligible["value"].min()),
                }
            )
    checkpoint_table = pd.DataFrame(checkpoint_rows)
    wide = checkpoint_table.pivot(
        index=["run_key", "method", "optimizer_replicate"],
        columns="checkpoint",
        values="best_so_far",
    ).reset_index()
    improvement_rows: list[dict[str, Any]] = []
    sorted_checkpoints = sorted(int(value) for value in checkpoints)
    for first, second in zip(sorted_checkpoints[:-1], sorted_checkpoints[1:]):
        for _, row in wide.iterrows():
            improvement_rows.append(
                {
                    "run_key": row["run_key"],
                    "method": row["method"],
                    "optimizer_replicate": int(row["optimizer_replicate"]),
                    "from_checkpoint": first,
                    "to_checkpoint": second,
                    "improvement": float(row[first] - row[second]),
                }
            )
    improvements = pd.DataFrame(improvement_rows)
    summary = (
        improvements.groupby(
            ["method", "from_checkpoint", "to_checkpoint"], sort=True
        )["improvement"]
        .agg(
            n="count",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            maximum="max",
        )
        .reset_index()
    )
    return data, checkpoint_table, summary


def assess_optimization_budget(
    checkpoint_table: pd.DataFrame,
    *,
    checkpoints: Sequence[int],
    delta_min: float | None,
    late_improvement_fraction: float,
) -> dict[str, Any]:
    if delta_min is None:
        return {
            "status": "delta_min_required",
            "recommended_evaluations": None,
        }
    threshold = delta_min * late_improvement_fraction
    checkpoints = sorted(int(value) for value in checkpoints)
    final_checkpoint = checkpoints[-1]
    wide = checkpoint_table.pivot(
        index=["run_key", "method", "optimizer_replicate"],
        columns="checkpoint",
        values="best_so_far",
    ).reset_index()
    diagnostics: list[dict[str, Any]] = []
    for checkpoint in checkpoints[:-1]:
        by_method = wide.assign(
            improvement_to_final=wide[checkpoint] - wide[final_checkpoint]
        ).groupby("method")["improvement_to_final"].median()
        passes = bool((by_method <= threshold).all())
        diagnostics.append(
            {
                "checkpoint": checkpoint,
                "threshold": threshold,
                "max_method_median_improvement_to_final": float(by_method.max()),
                "passes": passes,
            }
        )
    passing = [row for row in diagnostics if row["passes"]]
    if passing:
        return {
            "status": "recommended",
            "recommended_evaluations": passing[0]["checkpoint"],
            "tested_maximum": final_checkpoint,
            "diagnostics": diagnostics,
        }
    return {
        "status": "extend_beyond_tested_maximum",
        "recommended_evaluations": None,
        "tested_maximum": final_checkpoint,
        "diagnostics": diagnostics,
    }
