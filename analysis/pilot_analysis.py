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

from experiment_runtime import sha256_file


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
        expected_opinion = getattr(spec, "opinion_csv", None)
        if expected_opinion is not None:
            expected_hash = getattr(spec, "opinion_sha256", None)
            if expected_hash is None:
                expected_hash = sha256_file(expected_opinion)
            if intervention.get("opinion_mode") != "existing_csv":
                mismatches.append(
                    "opinion_mode="
                    f"{intervention.get('opinion_mode')!r} "
                    "(expected 'existing_csv')"
                )
            for field in ("opinion_source", "opinion_csv"):
                entry = intervention.get(field) or {}
                if entry.get("sha256") != expected_hash:
                    mismatches.append(
                        f"{field}_sha256={entry.get('sha256')!r} "
                        f"(expected {expected_hash!r})"
                    )
    return mismatches


def replace_precision_condition(
    original: PrecisionPilotData,
    correction: PrecisionPilotData,
    *,
    condition_id: str,
) -> PrecisionPilotData:
    """Replace one complete condition while preserving all other pilot rows."""

    correction_conditions = set(correction.runs["condition_id"].astype(str))
    if correction_conditions != {condition_id}:
        raise ValueError(
            "correction data must contain only the requested condition: "
            f"{sorted(correction_conditions)}"
        )
    original_keys = set(
        original.runs.loc[
            original.runs["condition_id"].astype(str) == condition_id,
            "run_key",
        ].astype(str)
    )
    correction_keys = set(correction.runs["run_key"].astype(str))
    if correction_keys != original_keys:
        raise ValueError(
            "correction run keys do not exactly match the original condition"
        )

    def replace(
        original_frame: pd.DataFrame, corrected_frame: pd.DataFrame
    ) -> pd.DataFrame:
        retained = original_frame[
            ~original_frame["run_key"].astype(str).isin(original_keys)
        ]
        corrected = corrected_frame.loc[:, original_frame.columns]
        return pd.concat([retained, corrected], ignore_index=True)

    merged_runs = replace(original.runs, correction.runs)
    merged_iterations = replace(original.iterations, correction.iterations)
    if merged_runs["run_key"].duplicated().any():
        raise ValueError("merged precision run inventory contains duplicate keys")
    iteration_key = ["run_key", "num_iter"]
    if merged_iterations.duplicated(iteration_key).any():
        raise ValueError("merged precision iteration table contains duplicate rows")
    if set(merged_runs["run_key"].astype(str)) != set(
        merged_iterations["run_key"].astype(str)
    ):
        raise ValueError("merged precision run and iteration keys differ")
    return PrecisionPilotData(
        iterations=merged_iterations.sort_values(iteration_key).reset_index(drop=True),
        runs=merged_runs.sort_values("run_key").reset_index(drop=True),
    )


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


def load_precision_analysis_snapshot(
    source_analysis_root: str | Path,
    *,
    expected_specs: Sequence[Any],
) -> PrecisionPilotData:
    """Load and validate the complete iteration table from a prior analysis."""
    source_analysis_root = Path(source_analysis_root).resolve()
    manifest_path = source_analysis_root / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"source analysis manifest does not exist: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("source precision analysis is not completed")
    if (
        manifest.get("stage") != "stage3_pilot"
        or manifest.get("phase") != "precision"
    ):
        raise ValueError("source analysis is not a Stage 3 precision analysis")

    outputs = manifest.get("outputs", {})

    def output_path(key: str) -> Path:
        relative = outputs.get(key)
        if not isinstance(relative, str):
            raise ValueError(f"source analysis output is missing: {key}")
        path = (source_analysis_root / relative).resolve()
        if source_analysis_root not in path.parents:
            raise ValueError(f"source analysis output escapes its directory: {key}")
        if not path.is_file():
            raise ValueError(f"source analysis output does not exist: {path}")
        return path

    iterations = pd.read_parquet(output_path("iteration_metrics"))
    runs = pd.read_csv(output_path("run_inventory"))
    required_iteration_columns = {
        "run_key",
        "network",
        "condition_id",
        "simulator_seed",
        "num_iter",
        OBJECTIVE_COLUMN,
    }
    if not required_iteration_columns.issubset(iterations.columns):
        missing = sorted(required_iteration_columns - set(iterations.columns))
        raise ValueError(f"source iteration table is missing columns: {missing}")
    required_run_columns = {
        "run_key",
        "network",
        "condition_id",
        "simulator_seed",
        "iterations",
        "certainty",
        "effectiveness",
        "objective",
    }
    if not required_run_columns.issubset(runs.columns):
        missing = sorted(required_run_columns - set(runs.columns))
        raise ValueError(f"source run inventory is missing columns: {missing}")

    expected_by_key = {spec.key: spec for spec in expected_specs}
    if len(expected_by_key) != len(expected_specs):
        raise ValueError("expected precision run keys are not unique")
    observed_run_keys = set(runs["run_key"].astype(str))
    observed_iteration_keys = set(iterations["run_key"].astype(str))
    expected_run_keys = set(expected_by_key)
    if observed_run_keys != expected_run_keys:
        raise ValueError("source run inventory does not match the protocol run keys")
    if observed_iteration_keys != expected_run_keys:
        raise ValueError("source iteration table does not match the protocol run keys")
    if runs["run_key"].duplicated().any():
        raise ValueError("source run inventory contains duplicate run keys")

    errors: list[str] = []
    runs_by_key = runs.set_index("run_key", drop=False)
    for run_key, spec in expected_by_key.items():
        run = runs_by_key.loc[run_key]
        group = iterations[iterations["run_key"] == run_key]
        expected_iterations = int(spec.iterations)
        observed_ids = set(pd.to_numeric(group["num_iter"]).astype(int))
        if len(group) != expected_iterations or observed_ids != set(
            range(expected_iterations)
        ):
            errors.append(f"iteration IDs do not match for {run_key}")
            continue
        expected_fields = {
            "network": spec.network,
            "condition_id": str(spec.condition_id),
            "simulator_seed": int(spec.simulator_seed),
            "iterations": expected_iterations,
        }
        for field, expected in expected_fields.items():
            observed = run[field]
            if field in {"simulator_seed", "iterations"}:
                observed = int(observed)
            else:
                observed = str(observed)
            if observed != expected:
                errors.append(
                    f"{field} does not match for {run_key}: "
                    f"{observed!r} != {expected!r}"
                )
        for field in ("network", "condition_id", "simulator_seed"):
            values = group[field].drop_duplicates().tolist()
            expected = expected_fields[field]
            if field == "simulator_seed":
                values = [int(value) for value in values]
            else:
                values = [str(value) for value in values]
            if values != [expected]:
                errors.append(f"iteration {field} does not match for {run_key}")
        objective_values = pd.to_numeric(group[OBJECTIVE_COLUMN], errors="coerce")
        if objective_values.isna().any() or not math.isclose(
            float(objective_values.mean()),
            float(run["objective"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"objective values do not match for {run_key}")
        for field in ("certainty", "effectiveness"):
            expected = getattr(spec, field)
            observed = run[field]
            if expected is None:
                if not pd.isna(observed):
                    errors.append(f"{field} should be empty for {run_key}")
            elif pd.isna(observed) or not math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"{field} does not match for {run_key}")

    if errors:
        raise ValueError(
            "source precision analysis does not match the protocol:\n"
            + "\n".join(errors)
        )
    return PrecisionPilotData(
        iterations=iterations.reset_index(drop=True),
        runs=runs.reset_index(drop=True),
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


def build_selection_regret(
    iterations: pd.DataFrame,
    *,
    prefixes: Sequence[int],
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure held-out loss from selecting a condition within one seed block."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("selection regret tolerance must be nonnegative")
    max_prefix = max(int(prefix) for prefix in prefixes)
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
            if reference.empty:
                raise ValueError(
                    f"selection regret has no reference blocks for {network}"
                )
            reference_best_condition = str(reference.idxmin())
            reference_best_value = float(reference.min())
            for prefix in prefixes:
                estimate = (
                    network_data[
                        (network_data["simulator_seed"] == seed)
                        & (network_data["num_iter"] < int(prefix))
                    ]
                    .groupby("condition_id")[OBJECTIVE_COLUMN]
                    .mean()
                )
                if set(estimate.index) != set(reference.index):
                    raise ValueError(
                        "selection regret condition sets do not match for "
                        f"{network}, seed {seed}, prefix {prefix}"
                    )
                selected_condition = str(estimate.idxmin())
                selected_reference_value = float(reference[selected_condition])
                regret = max(0.0, selected_reference_value - reference_best_value)
                rows.append(
                    {
                        "network": network,
                        "simulator_seed": int(seed),
                        "prefix_iterations": int(prefix),
                        "selected_condition": selected_condition,
                        "reference_best_condition": reference_best_condition,
                        "selected_reference_value": selected_reference_value,
                        "reference_best_value": reference_best_value,
                        "selection_regret": regret,
                        "tolerance": float(tolerance),
                        "within_tolerance": regret <= tolerance,
                        "reference_definition": "other_blocks_at_max_prefix",
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["network", "prefix_iterations"], sort=True)
        .agg(
            n_blocks=("simulator_seed", "nunique"),
            median_regret=("selection_regret", "median"),
            max_regret=("selection_regret", "max"),
            block_pass_rate=("within_tolerance", "mean"),
        )
        .reset_index()
    )
    summary["tolerance"] = float(tolerance)
    return detail, summary


def recommend_iteration_count_by_selection_regret(
    selection_regret_summary: pd.DataFrame,
    *,
    prefixes: Sequence[int],
    tolerance: float,
    minimum_block_pass_rate_per_network: float,
) -> dict[str, Any]:
    """Choose the smallest prefix that avoids material held-out selection loss."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("selection regret tolerance must be nonnegative")
    if not 0 < minimum_block_pass_rate_per_network <= 1:
        raise ValueError("minimum block pass rate must be in (0, 1]")
    diagnostics: list[dict[str, Any]] = []
    for prefix in prefixes:
        rows = selection_regret_summary[
            selection_regret_summary["prefix_iterations"] == int(prefix)
        ]
        if rows.empty:
            raise ValueError(f"selection regret summary is missing prefix {prefix}")
        network_diagnostics = []
        for _, row in rows.sort_values("network").iterrows():
            network_passes = bool(
                float(row["block_pass_rate"])
                >= minimum_block_pass_rate_per_network
            )
            network_diagnostics.append(
                {
                    "network": str(row["network"]),
                    "max_regret": float(row["max_regret"]),
                    "block_pass_rate": float(row["block_pass_rate"]),
                    "passes": network_passes,
                }
            )
        diagnostics.append(
            {
                "prefix_iterations": int(prefix),
                "tolerance": float(tolerance),
                "minimum_block_pass_rate_per_network": float(
                    minimum_block_pass_rate_per_network
                ),
                "max_regret_across_networks": max(
                    row["max_regret"] for row in network_diagnostics
                ),
                "passes": all(row["passes"] for row in network_diagnostics),
                "networks": network_diagnostics,
            }
        )
    passing = [row for row in diagnostics if row["passes"]]
    if passing:
        return {
            "status": "recommended",
            "recommended_iterations": passing[0]["prefix_iterations"],
            "method": "leave_one_simulator_seed_block_out_selection_regret",
            "selection_regret_tolerance": float(tolerance),
            "diagnostics": diagnostics,
            "reason": (
                "Smallest tested prefix satisfying the selection-regret rule "
                "in every network."
            ),
        }
    return {
        "status": "insufficient_selection_precision",
        "recommended_iterations": None,
        "method": "leave_one_simulator_seed_block_out_selection_regret",
        "selection_regret_tolerance": float(tolerance),
        "diagnostics": diagnostics,
        "reason": "No tested prefix satisfies the selection-regret rule.",
    }


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
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    integer_settings = {
        "fixed_conditions": fixed_conditions,
        "fixed_seed_blocks": fixed_seed_blocks,
        "optimization_methods": optimization_methods,
        "optimization_replicates": optimization_replicates,
    }
    for name, value in integer_settings.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        optimization_evaluations is not None
        and int(optimization_evaluations) <= 0
    ):
        raise ValueError("optimization_evaluations must be positive")

    rows: list[dict[str, Any]] = []
    for _, network in resource_summary.iterrows():
        fixed_runs = int(fixed_conditions) * int(fixed_seed_blocks)
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
                "projected_simulation_sec_p90": float(
                    network["p90_simulation_sec_per_iteration"]
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
                int(optimization_methods)
                * int(optimization_replicates)
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
                    "projected_simulation_sec_p90": float(
                        network["p90_simulation_sec_per_iteration"]
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
    for (run_key, network, method, replicate), group in data.groupby(
        ["run_key", "network", "method", "optimizer_replicate"], sort=True
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
                    "network": network,
                    "method": method,
                    "optimizer_replicate": int(replicate),
                    "checkpoint": int(checkpoint),
                    "best_so_far": float(eligible["value"].min()),
                }
            )
    checkpoint_table = pd.DataFrame(checkpoint_rows)
    wide = checkpoint_table.pivot(
        index=["run_key", "network", "method", "optimizer_replicate"],
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
                    "network": row["network"],
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
            ["network", "method", "from_checkpoint", "to_checkpoint"], sort=True
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
        index=["run_key", "network", "method", "optimizer_replicate"],
        columns="checkpoint",
        values="best_so_far",
    ).reset_index()
    diagnostics: list[dict[str, Any]] = []
    for checkpoint in checkpoints[:-1]:
        by_network_method = wide.assign(
            improvement_to_final=wide[checkpoint] - wide[final_checkpoint]
        ).groupby(["network", "method"])["improvement_to_final"].median()
        passes = bool((by_network_method <= threshold).all())
        diagnostics.append(
            {
                "checkpoint": checkpoint,
                "threshold": threshold,
                "max_network_method_median_improvement_to_final": float(
                    by_network_method.max()
                ),
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


def assess_optimization_budget_by_tolerance(
    checkpoint_table: pd.DataFrame,
    *,
    checkpoints: Sequence[int],
    improvement_tolerance: float,
    minimum_passing_replicates_per_network_method: int,
) -> dict[str, Any]:
    """Select an optimization checkpoint using an absolute lost-gain tolerance."""
    if not math.isfinite(improvement_tolerance) or improvement_tolerance < 0:
        raise ValueError("optimization improvement tolerance must be nonnegative")
    if minimum_passing_replicates_per_network_method <= 0:
        raise ValueError("minimum passing replicates must be positive")
    checkpoints = sorted(int(value) for value in checkpoints)
    final_checkpoint = checkpoints[-1]
    wide = checkpoint_table.pivot(
        index=["run_key", "network", "method", "optimizer_replicate"],
        columns="checkpoint",
        values="best_so_far",
    ).reset_index()
    network_method_sizes = wide.groupby(["network", "method"])[
        "optimizer_replicate"
    ].nunique()
    if (
        network_method_sizes < minimum_passing_replicates_per_network_method
    ).any():
        raise ValueError(
            "minimum passing replicates exceeds the available runs for a "
            "network-method combination"
        )

    diagnostics: list[dict[str, Any]] = []
    for checkpoint in checkpoints[:-1]:
        improvements = wide.assign(
            improvement_to_final=wide[checkpoint] - wide[final_checkpoint]
        )
        network_method_diagnostics: list[dict[str, Any]] = []
        for (network, method), group in improvements.groupby(
            ["network", "method"], sort=True
        ):
            values = group["improvement_to_final"].astype(float)
            passing_replicates = int((values <= improvement_tolerance).sum())
            median_improvement = float(values.median())
            network_method_passes = bool(
                passing_replicates
                >= minimum_passing_replicates_per_network_method
            )
            network_method_diagnostics.append(
                {
                    "network": str(network),
                    "method": str(method),
                    "n_replicates": int(len(values)),
                    "passing_replicates": passing_replicates,
                    "median_improvement_to_final": median_improvement,
                    "max_improvement_to_final": float(values.max()),
                    "passes": network_method_passes,
                }
            )
        diagnostics.append(
            {
                "checkpoint": int(checkpoint),
                "final_checkpoint": final_checkpoint,
                "improvement_tolerance": float(improvement_tolerance),
                "minimum_passing_replicates_per_network_method": int(
                    minimum_passing_replicates_per_network_method
                ),
                "passes": all(
                    row["passes"] for row in network_method_diagnostics
                ),
                "network_methods": network_method_diagnostics,
            }
        )
    passing = [row for row in diagnostics if row["passes"]]
    if passing:
        return {
            "status": "recommended",
            "recommended_evaluations": passing[0]["checkpoint"],
            "tested_maximum": final_checkpoint,
            "method": "best_so_far_improvement_to_final_checkpoint",
            "improvement_tolerance": float(improvement_tolerance),
            "diagnostics": diagnostics,
            "reason": (
                "Smallest nonfinal checkpoint whose lost improvement is within "
                "tolerance for every network-method combination."
            ),
        }
    return {
        "status": "extend_beyond_tested_maximum",
        "recommended_evaluations": None,
        "tested_maximum": final_checkpoint,
        "method": "best_so_far_improvement_to_final_checkpoint",
        "improvement_tolerance": float(improvement_tolerance),
        "diagnostics": diagnostics,
        "reason": (
            "The last nonfinal checkpoint did not satisfy the stopping rule; "
            "the pilot must be extended beyond the tested maximum."
        ),
    }
