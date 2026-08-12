"""Formal reaggregation of the 3,600 saved optimization trials.

The source runs were optimized with the former per-step metric.  This module
keeps those runs immutable, recalculates both the former metric and the new
cumulative objective from raw Arrow files, and emits audit and analysis-ready
tables through one reproducible code path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.optimization_metrics import (
    LEGACY_METRIC_NAME,
    MetricValidationError,
    compute_selfish_metrics,
    validate_pop_agent_consistency,
)
from experiment_runtime import NETWORKS, REPO_ROOT, relative_to_repo


SOURCE_RUN_NAME = "optimize_runs_auc_raw_20260626"
EXPECTED_ITERATIONS = 100
EXPECTED_TRIALS_PER_RUN = 100
OLD_METRIC_TOLERANCE = 1e-12

NETWORK_SOURCE_DIRS = {
    "ba1000": "optimization_ba1000",
    "facebook": "optimization_facebook",
    "wiki_vote": "optimization_wiki_vote",
}

METHOD_SPECS = {
    "gpr": {
        "display": "GPR",
        "implementation": "botorch_gp",
    },
    "cmaes": {
        "display": "CMA-ES",
        "implementation": "optuna_cma_es",
    },
    "random": {
        "display": "Random",
        "implementation": "optuna_random",
    },
    "ga": {
        "display": "GA (legacy label)",
        "implementation": "optuna_nsga_ii_legacy_ga_label",
    },
}

INFO_NAMES = {
    0: "misinformation",
    1: "corrective",
    2: "observational",
    3: "behavior_guiding",
}
INFO_COUNT_COLUMNS = (
    "num_posted",
    "num_received",
    "num_shared",
    "num_viewed",
    "num_fst_viewed",
)
REQUIRED_INFO_COLUMNS = ("num_iter", "t", "info_label", *INFO_COUNT_COLUMNS)


@dataclass(frozen=True)
class SourceRun:
    network: str
    method: str
    optseed: int
    run_dir: Path
    timing_path: Path
    summary_path: Path

    @property
    def run_id(self) -> str:
        return f"{self.network}:{self.method}:optseed{self.optseed}"


@dataclass(frozen=True)
class ReaggregationResult:
    trial_summary: pd.DataFrame
    iteration_metrics: pd.DataFrame
    info_summary: pd.DataFrame
    trial_audit: pd.DataFrame
    run_inventory: pd.DataFrame
    analysis_tables: dict[str, pd.DataFrame]
    elapsed_sec: float


def discover_source_runs(
    *,
    repo_root: Path = REPO_ROOT,
    networks: Sequence[str] = tuple(NETWORK_SOURCE_DIRS),
    methods: Sequence[str] = tuple(METHOD_SPECS),
    optseeds: Sequence[int] = (4, 5, 6),
) -> list[SourceRun]:
    """Return the protocol-defined source runs in stable order."""

    runs: list[SourceRun] = []
    for network in networks:
        if network not in NETWORK_SOURCE_DIRS:
            raise ValueError(f"unsupported source network: {network}")
        source_root = (
            Path(repo_root)
            / "experiments"
            / NETWORK_SOURCE_DIRS[network]
            / SOURCE_RUN_NAME
        )
        for method in methods:
            if method not in METHOD_SPECS:
                raise ValueError(f"unsupported source method: {method}")
            for optseed in optseeds:
                run_dir = source_root / f"{method}_optseed{int(optseed)}"
                runs.append(
                    SourceRun(
                        network=network,
                        method=method,
                        optseed=int(optseed),
                        run_dir=run_dir,
                        timing_path=run_dir / "logs" / f"timing_{method}.csv",
                        summary_path=run_dir / "logs" / f"summary_{method}.json",
                    )
                )
    return runs


def reconstruct_applied_parameter(value: float) -> float:
    """Reproduce ``optimize_test.py``'s four-decimal application rule."""

    value = float(value)
    if not math.isfinite(value) or not 0.5 <= value <= 1.0:
        raise ValueError(f"design parameter outside [0.5, 1.0]: {value}")
    return round(value, 4)


def _validate_info_data(
    info_df: pd.DataFrame, *, expected_iterations: int
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_INFO_COLUMNS if column not in info_df]
    if missing:
        raise MetricValidationError(
            f"info data is missing required columns: {', '.join(missing)}"
        )
    if info_df.empty:
        raise MetricValidationError("info data is empty")

    data = info_df.loc[:, REQUIRED_INFO_COLUMNS].copy()
    for column in REQUIRED_INFO_COLUMNS:
        if data[column].isna().any():
            raise MetricValidationError(f"info column '{column}' has missing values")
        numeric = pd.to_numeric(data[column], errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise MetricValidationError(
                f"info column '{column}' has non-finite values"
            )
        if not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise MetricValidationError(
                f"info column '{column}' must contain non-negative integers"
            )
        data[column] = numeric.astype("int64")

    if data.duplicated(["num_iter", "t", "info_label"]).any():
        raise MetricValidationError(
            "info data contains duplicate (num_iter, t, info_label) rows"
        )

    observed_iterations = set(int(value) for value in data["num_iter"].unique())
    expected = set(range(expected_iterations))
    if observed_iterations != expected:
        raise MetricValidationError(
            "info iteration IDs do not match the expected range; "
            f"missing={sorted(expected - observed_iterations)}, "
            f"unexpected={sorted(observed_iterations - expected)}"
        )

    labels = set(int(value) for value in data["info_label"].unique())
    if labels != set(INFO_NAMES):
        raise MetricValidationError(
            f"info labels must be {sorted(INFO_NAMES)}; observed={sorted(labels)}"
        )
    return data.sort_values(["num_iter", "t", "info_label"], kind="stable")


def summarize_info_data(
    info_df: pd.DataFrame,
    *,
    num_agents: int,
    expected_iterations: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aggregate information diffusion by information type for one trial."""

    data = _validate_info_data(
        info_df, expected_iterations=expected_iterations
    )
    rows: list[dict[str, Any]] = []
    wide: dict[str, float] = {}
    denominator = num_agents * expected_iterations

    for label, info_name in INFO_NAMES.items():
        subset = data[data["info_label"] == label]
        row: dict[str, Any] = {
            "info_label": label,
            "info_name": info_name,
            "n_rows": int(len(subset)),
            "n_iterations_present": int(subset["num_iter"].nunique()),
            "min_t": int(subset["t"].min()),
            "max_t": int(subset["t"].max()),
        }
        for column in INFO_COUNT_COLUMNS:
            total = int(subset[column].sum())
            short_name = column.removeprefix("num_")
            row[f"{short_name}_sum"] = total
            row[f"{short_name}_per_iteration"] = total / expected_iterations
            row[f"{short_name}_per_agent_iteration"] = total / denominator
            wide[f"{info_name}_{short_name}_sum"] = float(total)
            wide[f"{info_name}_{short_name}_per_agent_iteration"] = (
                total / denominator
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    indexed = summary.set_index("info_name")
    for metric in ("shared", "viewed", "fst_viewed"):
        numerator = float(indexed.loc["corrective", f"{metric}_sum"])
        denominator_value = float(
            indexed.loc["misinformation", f"{metric}_sum"]
        )
        wide[f"corrective_to_misinformation_{metric}_ratio"] = (
            numerator / denominator_value
            if denominator_value > 0
            else math.nan
        )
    return summary, wide


def _resolve_trial_raw_path(
    run: SourceRun, row: pd.Series, kind: str, trial: int
) -> Path:
    field = f"trial_{kind}_arrow"
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return (run.run_dir / value).resolve()
    return (
        run.run_dir
        / "result"
        / "trials"
        / f"trial_{trial:04d}_{kind}.arrow"
    ).resolve()


def _base_audit_row(
    run: SourceRun, trial: int, trial_id: str
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "run_id": run.run_id,
        "network": run.network,
        "method": run.method,
        "optseed": run.optseed,
        "trial": trial,
        "valid": False,
        "source_state_complete": False,
        "raw_files_exist": False,
        "pop_readable": False,
        "info_readable": False,
        "agent_readable": False,
        "iteration_count_valid": False,
        "old_metric_reproduced": False,
        "pop_agent_consistent": False,
        "info_data_valid": False,
        "error_type": None,
        "error_message": None,
    }


def _process_trial(
    run: SourceRun,
    source_row: pd.Series,
    *,
    expected_iterations: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trial = int(source_row["trial"])
    trial_id = (
        f"{run.network}:{run.method}:optseed{run.optseed}:trial{trial:04d}"
    )
    audit = _base_audit_row(run, trial, trial_id)
    state = str(source_row.get("state", ""))
    audit["source_state_complete"] = "COMPLETE" in state
    if not audit["source_state_complete"]:
        raise MetricValidationError(f"source trial state is not COMPLETE: {state}")

    raw_paths = {
        kind: _resolve_trial_raw_path(run, source_row, kind, trial)
        for kind in ("pop", "info", "agent")
    }
    audit["raw_files_exist"] = all(path.is_file() for path in raw_paths.values())
    if not audit["raw_files_exist"]:
        missing = [kind for kind, path in raw_paths.items() if not path.is_file()]
        raise FileNotFoundError(f"missing raw Arrow files: {missing}")

    pop_df = pd.read_feather(raw_paths["pop"])
    audit["pop_readable"] = True
    info_df = pd.read_feather(raw_paths["info"])
    audit["info_readable"] = True
    agent_df = pd.read_feather(raw_paths["agent"])
    audit["agent_readable"] = True

    num_agents = NETWORKS[run.network].num_agents
    metrics = compute_selfish_metrics(
        pop_df,
        num_agents=num_agents,
        expected_iterations=expected_iterations,
    )
    audit["iteration_count_valid"] = (
        metrics.summary["n_iterations"] == expected_iterations
    )

    validate_pop_agent_consistency(
        pop_df,
        agent_df,
        num_agents=num_agents,
        expected_iterations=expected_iterations,
    )
    audit["pop_agent_consistent"] = True

    info_summary, info_wide = summarize_info_data(
        info_df,
        num_agents=num_agents,
        expected_iterations=expected_iterations,
    )
    audit["info_data_valid"] = True

    old_logged = float(source_row["normalized_selfish_auc"])
    old_recalculated = float(metrics.summary[LEGACY_METRIC_NAME])
    old_error = abs(old_logged - old_recalculated)
    audit["old_metric_reproduced"] = old_error <= OLD_METRIC_TOLERANCE
    if not audit["old_metric_reproduced"]:
        raise MetricValidationError(
            "logged old metric does not match raw recalculation: "
            f"logged={old_logged}, recalculated={old_recalculated}, "
            f"error={old_error}"
        )

    proposed_certainty = float(source_row["certainty"])
    proposed_effectiveness = float(source_row["effectiveness"])
    trial_row: dict[str, Any] = {
        "trial_id": trial_id,
        "run_id": run.run_id,
        "network": run.network,
        "network_num_agents": num_agents,
        "method": run.method,
        "method_display": METHOD_SPECS[run.method]["display"],
        "method_implementation": METHOD_SPECS[run.method]["implementation"],
        "optseed": run.optseed,
        "optimizer_seed": int(source_row["sampler_seed"]),
        "trial": trial,
        "source_state": state,
        "proposed_certainty": proposed_certainty,
        "proposed_effectiveness": proposed_effectiveness,
        "applied_certainty": reconstruct_applied_parameter(proposed_certainty),
        "applied_effectiveness": reconstruct_applied_parameter(
            proposed_effectiveness
        ),
        "applied_value_source": "reconstructed_from_optimize_test_round_4dp",
        "j_old_logged": old_logged,
        "j_old_recalculated": old_recalculated,
        "j_old_absolute_error": old_error,
        "j_cum": float(metrics.summary["cumulative_selfish_fraction"]),
        "j_peak": float(metrics.summary["peak_new_selfish_ratio"]),
        "mean_first_selfish_step": metrics.summary["mean_first_selfish_step"],
        "mean_last_selfish_step": metrics.summary["mean_last_selfish_step"],
        "mean_t50_selfish_step": metrics.summary["mean_t50_selfish_step"],
        "mean_t90_selfish_step": metrics.summary["mean_t90_selfish_step"],
        "mean_selfish_timing_centroid": metrics.summary[
            "mean_selfish_timing_centroid"
        ],
        "mean_selfish_span_steps": metrics.summary["mean_selfish_span_steps"],
        "mean_active_selfish_steps": metrics.summary[
            "mean_active_selfish_steps"
        ],
        "mean_recorded_steps": metrics.summary["mean_recorded_steps"],
        "min_recorded_steps": metrics.summary["min_recorded_steps"],
        "max_recorded_steps": metrics.summary["max_recorded_steps"],
        "n_zero_selfish_iterations": metrics.summary[
            "n_zero_selfish_iterations"
        ],
        "source_trial_elapsed_sec": source_row.get("trial_elapsed_sec"),
        "source_simulation_elapsed_sec": source_row.get(
            "simulation_elapsed_sec"
        ),
        "source_score_elapsed_sec": source_row.get("score_elapsed_sec"),
        "pop_arrow": relative_to_repo(raw_paths["pop"]),
        "info_arrow": relative_to_repo(raw_paths["info"]),
        "agent_arrow": relative_to_repo(raw_paths["agent"]),
        **info_wide,
    }

    iteration = metrics.per_iteration.copy()
    iteration.insert(0, "trial", trial)
    iteration.insert(0, "optseed", run.optseed)
    iteration.insert(0, "method", run.method)
    iteration.insert(0, "network", run.network)
    iteration.insert(0, "trial_id", trial_id)

    info_summary.insert(0, "trial", trial)
    info_summary.insert(0, "optseed", run.optseed)
    info_summary.insert(0, "method", run.method)
    info_summary.insert(0, "network", run.network)
    info_summary.insert(0, "trial_id", trial_id)

    audit["valid"] = True
    return trial_row, iteration, info_summary, audit


def _run_inventory(run: SourceRun, timing: pd.DataFrame | None) -> dict[str, Any]:
    trial_dir = run.run_dir / "result" / "trials"
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "network": run.network,
        "method": run.method,
        "method_implementation": METHOD_SPECS[run.method]["implementation"],
        "optseed": run.optseed,
        "run_dir": relative_to_repo(run.run_dir),
        "timing_csv": relative_to_repo(run.timing_path),
        "summary_json": relative_to_repo(run.summary_path),
        "run_dir_exists": run.run_dir.is_dir(),
        "timing_exists": run.timing_path.is_file(),
        "summary_exists": run.summary_path.is_file(),
        "timing_rows": 0 if timing is None else int(len(timing)),
        "complete_timing_rows": (
            0
            if timing is None
            else int(timing["state"].astype(str).str.contains("COMPLETE").sum())
        ),
    }
    for kind in ("pop", "info", "agent"):
        row[f"{kind}_arrow_count"] = len(
            list(trial_dir.glob(f"trial_*_{kind}.arrow"))
        )
    row["inventory_complete"] = (
        row["timing_rows"] == EXPECTED_TRIALS_PER_RUN
        and row["complete_timing_rows"] == EXPECTED_TRIALS_PER_RUN
        and all(
            row[f"{kind}_arrow_count"] == EXPECTED_TRIALS_PER_RUN
            for kind in ("pop", "info", "agent")
        )
    )
    return row


def spearman_with_bootstrap(
    x: Iterable[float],
    y: Iterable[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    """Calculate Spearman rho and a percentile bootstrap interval."""

    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    x_values = frame["x"].to_numpy(dtype=float)
    y_values = frame["y"].to_numpy(dtype=float)
    if len(frame) < 3 or np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return {"n": len(frame), "rho": math.nan, "ci_low": math.nan, "ci_high": math.nan}

    rho = float(spearmanr(x_values, y_values).statistic)
    rng = np.random.default_rng(seed)
    bootstrapped: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(frame), size=len(frame))
        sampled_x = x_values[indices]
        sampled_y = y_values[indices]
        if np.unique(sampled_x).size < 2 or np.unique(sampled_y).size < 2:
            continue
        value = float(spearmanr(sampled_x, sampled_y).statistic)
        if math.isfinite(value):
            bootstrapped.append(value)
    if not bootstrapped:
        low = high = math.nan
    else:
        low, high = np.quantile(bootstrapped, [0.025, 0.975])
    return {
        "n": len(frame),
        "rho": rho,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if (
        len(frame) < 3
        or frame["x"].nunique() < 2
        or frame["y"].nunique() < 2
    ):
        return math.nan
    return float(spearmanr(frame["x"], frame["y"]).statistic)


def _metric_comparison(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = trials.copy()
    ranked["j_old_percentile"] = ranked.groupby("network")[
        "j_old_recalculated"
    ].rank(method="average", pct=True)
    ranked["j_cum_percentile"] = ranked.groupby("network")["j_cum"].rank(
        method="average", pct=True
    )
    ranked["old_low_10pct"] = False
    ranked["new_low_10pct"] = False
    for _, group in ranked.groupby("network", sort=True):
        n_low = max(1, int(math.ceil(len(group) * 0.10)))
        old_indices = group.sort_values(
            ["j_old_recalculated", "trial_id"], kind="stable"
        ).index[:n_low]
        new_indices = group.sort_values(
            ["j_cum", "trial_id"], kind="stable"
        ).index[:n_low]
        ranked.loc[old_indices, "old_low_10pct"] = True
        ranked.loc[new_indices, "new_low_10pct"] = True
    ranked["rank_percentile_change"] = (
        ranked["j_cum_percentile"] - ranked["j_old_percentile"]
    )

    rows: list[dict[str, Any]] = []
    for network, group in ranked.groupby("network", sort=True):
        old_set = set(group.loc[group["old_low_10pct"], "trial_id"])
        new_set = set(group.loc[group["new_low_10pct"], "trial_id"])
        intersection = old_set & new_set
        union = old_set | new_set
        rows.append(
            {
                "network": network,
                "n_trials": len(group),
                "old_new_spearman": _safe_spearman(
                    group["j_old_recalculated"], group["j_cum"]
                ),
                "old_low_10pct_count": len(old_set),
                "new_low_10pct_count": len(new_set),
                "low_10pct_overlap_count": len(intersection),
                "old_group_retained_fraction": (
                    len(intersection) / len(old_set) if old_set else math.nan
                ),
                "low_10pct_jaccard": (
                    len(intersection) / len(union) if union else math.nan
                ),
                "old_metric_span_spearman": _safe_spearman(
                    group["j_old_recalculated"],
                    group["mean_selfish_span_steps"],
                ),
            }
        )
    return ranked, pd.DataFrame(rows)


def _parameter_correlations(
    random_trials: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for network, group in random_trials.groupby("network", sort=True):
        for offset, parameter in enumerate(
            ("applied_certainty", "applied_effectiveness")
        ):
            result = spearman_with_bootstrap(
                group[parameter],
                group["j_cum"],
                n_bootstrap=n_bootstrap,
                seed=seed + offset + 100 * len(rows),
            )
            rows.append({"network": network, "parameter": parameter, **result})
    return pd.DataFrame(rows)


def _parameter_bins(random_trials: pd.DataFrame) -> pd.DataFrame:
    edges = np.linspace(0.5, 1.0, 11)
    rows: list[pd.DataFrame] = []
    for network, network_group in random_trials.groupby("network", sort=True):
        for parameter in ("applied_certainty", "applied_effectiveness"):
            group = network_group.copy()
            group["parameter_bin"] = pd.cut(
                group[parameter], edges, include_lowest=True, right=True
            )
            summary = (
                group.groupby("parameter_bin", observed=False)["j_cum"]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            summary["sem"] = summary["std"] / np.sqrt(summary["count"])
            summary["bin_left"] = (
                summary["parameter_bin"]
                .map(lambda value: float(value.left))
                .astype(float)
            )
            summary["bin_right"] = (
                summary["parameter_bin"]
                .map(lambda value: float(value.right))
                .astype(float)
            )
            summary["bin_midpoint"] = (
                summary["bin_left"] + summary["bin_right"]
            ) / 2
            summary.insert(0, "parameter", parameter)
            summary.insert(0, "network", network)
            rows.append(summary.drop(columns=["parameter_bin"]))
    return pd.concat(rows, ignore_index=True)


def _response_surface(random_trials: pd.DataFrame) -> pd.DataFrame:
    edges = np.linspace(0.5, 1.0, 6)
    data = random_trials.copy()
    data["certainty_bin"] = pd.cut(
        data["applied_certainty"], edges, include_lowest=True
    )
    data["effectiveness_bin"] = pd.cut(
        data["applied_effectiveness"], edges, include_lowest=True
    )
    summary = (
        data.groupby(
            ["network", "certainty_bin", "effectiveness_bin"],
            observed=False,
        )["j_cum"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    for column in ("certainty_bin", "effectiveness_bin"):
        prefix = column.removesuffix("_bin")
        summary[f"{prefix}_midpoint"] = (
            summary[column]
            .map(lambda value: (float(value.left) + float(value.right)) / 2)
            .astype(float)
        )
    return summary.drop(columns=["certainty_bin", "effectiveness_bin"])


def _conditional_certainty(random_trials: pd.DataFrame) -> pd.DataFrame:
    data = random_trials.copy()
    certainty_edges = np.linspace(0.5, 1.0, 6)
    effectiveness_edges = [0.5, 2 / 3, 5 / 6, 1.0]
    labels = ["low", "middle", "high"]
    data["certainty_bin"] = pd.cut(
        data["applied_certainty"], certainty_edges, include_lowest=True
    )
    data["effectiveness_level"] = pd.cut(
        data["applied_effectiveness"],
        effectiveness_edges,
        labels=labels,
        include_lowest=True,
    )
    summary = (
        data.groupby(
            ["network", "effectiveness_level", "certainty_bin"],
            observed=False,
        )["j_cum"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["certainty_midpoint"] = (
        summary["certainty_bin"]
        .map(lambda value: (float(value.left) + float(value.right)) / 2)
        .astype(float)
    )
    return summary.drop(columns=["certainty_bin"])


def _auxiliary_correlations(random_trials: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "j_peak",
        "mean_first_selfish_step",
        "mean_last_selfish_step",
        "mean_t50_selfish_step",
        "mean_t90_selfish_step",
        "mean_selfish_timing_centroid",
        "mean_selfish_span_steps",
        "mean_active_selfish_steps",
        "mean_recorded_steps",
    )
    rows: list[dict[str, Any]] = []
    for network, group in random_trials.groupby("network", sort=True):
        for metric in metrics:
            valid = group[["j_cum", metric]].dropna()
            rows.append(
                {
                    "network": network,
                    "metric": metric,
                    "n": len(valid),
                    "spearman_with_j_cum": _safe_spearman(
                        valid["j_cum"], valid[metric]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _parameter_metric_correlations(random_trials: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "j_old_recalculated",
        "j_cum",
        "j_peak",
        "mean_recorded_steps",
        "mean_selfish_span_steps",
        "mean_selfish_timing_centroid",
        "corrective_to_misinformation_fst_viewed_ratio",
        "behavior_guiding_fst_viewed_per_agent_iteration",
        "corrective_fst_viewed_per_agent_iteration",
    )
    rows: list[dict[str, Any]] = []
    for network, group in random_trials.groupby("network", sort=True):
        for parameter in ("applied_certainty", "applied_effectiveness"):
            for metric in metrics:
                rows.append(
                    {
                        "network": network,
                        "parameter": parameter,
                        "metric": metric,
                        "n": int(group[[parameter, metric]].dropna().shape[0]),
                        "spearman": _safe_spearman(group[parameter], group[metric]),
                    }
                )
    return pd.DataFrame(rows)


def _method_metric_comparison(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (network, method), group in trials.groupby(
        ["network", "method"], sort=True
    ):
        rows.append(
            {
                "network": network,
                "method": method,
                "n": len(group),
                "old_new_spearman": _safe_spearman(
                    group["j_old_recalculated"], group["j_cum"]
                ),
                "old_metric_span_spearman": _safe_spearman(
                    group["j_old_recalculated"],
                    group["mean_selfish_span_steps"],
                ),
                "j_old_mean": float(group["j_old_recalculated"].mean()),
                "j_cum_mean": float(group["j_cum"].mean()),
                "j_cum_std": float(group["j_cum"].std()),
                "mean_recorded_steps": float(group["mean_recorded_steps"].mean()),
                "mean_selfish_span_steps": float(
                    group["mean_selfish_span_steps"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _cross_network_summary(cross_network: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("ba1000", "facebook"),
        ("ba1000", "wiki_vote"),
        ("facebook", "wiki_vote"),
    )
    rows: list[dict[str, Any]] = []
    for first, second in pairs:
        if first not in cross_network or second not in cross_network:
            continue
        difference = cross_network[second] - cross_network[first]
        rows.append(
            {
                "network_1": first,
                "network_2": second,
                "n_paired_points": len(cross_network),
                "response_spearman": _safe_spearman(
                    cross_network[first], cross_network[second]
                ),
                "mean_network_2_minus_1": float(difference.mean()),
                "std_network_2_minus_1": float(difference.std()),
                "min_network_2_minus_1": float(difference.min()),
                "max_network_2_minus_1": float(difference.max()),
            }
        )
    return pd.DataFrame(rows)


def _random_pairing_tables(
    random_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duplicate_rows: list[dict[str, Any]] = []
    for (network, optseed), group in random_trials.groupby(
        ["network", "optseed"], sort=True
    ):
        duplicate_rows.append(
            {
                "network": network,
                "optseed": optseed,
                "n_trials": len(group),
                "proposed_duplicate_count": int(
                    group.duplicated(
                        ["proposed_certainty", "proposed_effectiveness"]
                    ).sum()
                ),
                "applied_duplicate_count": int(
                    group.duplicated(
                        ["applied_certainty", "applied_effectiveness"]
                    ).sum()
                ),
            }
        )

    pairing_rows: list[dict[str, Any]] = []
    for (optseed, trial), group in random_trials.groupby(
        ["optseed", "trial"], sort=True
    ):
        proposed = group[["proposed_certainty", "proposed_effectiveness"]]
        applied = group[["applied_certainty", "applied_effectiveness"]]
        pairing_rows.append(
            {
                "optseed": optseed,
                "trial": trial,
                "network_count": group["network"].nunique(),
                "proposed_points_equal": bool(
                    proposed.nunique(dropna=False).max() == 1
                ),
                "applied_points_equal": bool(
                    applied.nunique(dropna=False).max() == 1
                ),
                "certainty": float(applied.iloc[0]["applied_certainty"]),
                "effectiveness": float(
                    applied.iloc[0]["applied_effectiveness"]
                ),
            }
        )

    response = random_trials.pivot(
        index=[
            "optseed",
            "trial",
            "applied_certainty",
            "applied_effectiveness",
        ],
        columns="network",
        values="j_cum",
    ).reset_index()
    response.columns.name = None
    if {"ba1000", "facebook", "wiki_vote"}.issubset(response.columns):
        response["facebook_minus_ba1000"] = (
            response["facebook"] - response["ba1000"]
        )
        response["wiki_vote_minus_ba1000"] = (
            response["wiki_vote"] - response["ba1000"]
        )
        response["wiki_vote_minus_facebook"] = (
            response["wiki_vote"] - response["facebook"]
        )
    return (
        pd.DataFrame(duplicate_rows),
        pd.DataFrame(pairing_rows),
        response,
    )


def build_analysis_tables(
    trials: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    ranked, comparison = _metric_comparison(trials)
    random_trials = trials[trials["method"] == "random"].copy()
    duplicates, pairing, cross_network = _random_pairing_tables(random_trials)
    return {
        "trial_rank_comparison": ranked,
        "metric_comparison_summary": comparison,
        "method_metric_comparison": _method_metric_comparison(trials),
        "parameter_correlations": _parameter_correlations(
            random_trials,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        ),
        "parameter_metric_correlations": _parameter_metric_correlations(
            random_trials
        ),
        "parameter_bins": _parameter_bins(random_trials),
        "response_surface": _response_surface(random_trials),
        "conditional_certainty": _conditional_certainty(random_trials),
        "auxiliary_correlations": _auxiliary_correlations(random_trials),
        "random_duplicate_audit": duplicates,
        "random_pairing_audit": pairing,
        "random_cross_network_responses": cross_network,
        "random_cross_network_summary": _cross_network_summary(cross_network),
    }


def reaggregate_saved_trials(
    *,
    repo_root: Path = REPO_ROOT,
    networks: Sequence[str] = tuple(NETWORK_SOURCE_DIRS),
    methods: Sequence[str] = tuple(METHOD_SPECS),
    optseeds: Sequence[int] = (4, 5, 6),
    expected_iterations: int = EXPECTED_ITERATIONS,
    expected_trials_per_run: int = EXPECTED_TRIALS_PER_RUN,
    max_trials_per_run: int | None = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 20260812,
    progress_every: int = 100,
) -> ReaggregationResult:
    """Audit and reaggregate all requested source trials."""

    start = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []
    iteration_frames: list[pd.DataFrame] = []
    info_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    processed = 0

    runs = discover_source_runs(
        repo_root=repo_root,
        networks=networks,
        methods=methods,
        optseeds=optseeds,
    )
    for run in runs:
        timing: pd.DataFrame | None = None
        try:
            timing = pd.read_csv(run.timing_path)
        except (OSError, pd.errors.ParserError):
            pass
        inventory_rows.append(_run_inventory(run, timing))
        if timing is None:
            continue

        if len(timing) != expected_trials_per_run and max_trials_per_run is None:
            continue
        timing = timing.sort_values("trial", kind="stable")
        if max_trials_per_run is not None:
            timing = timing.head(max_trials_per_run)

        for _, source_row in timing.iterrows():
            trial = int(source_row["trial"])
            trial_id = (
                f"{run.network}:{run.method}:optseed{run.optseed}:"
                f"trial{trial:04d}"
            )
            try:
                trial_row, iteration, info, audit = _process_trial(
                    run,
                    source_row,
                    expected_iterations=expected_iterations,
                )
                trial_rows.append(trial_row)
                iteration_frames.append(iteration)
                info_frames.append(info)
                audit_rows.append(audit)
            except Exception as exc:
                audit = _base_audit_row(run, trial, trial_id)
                audit["source_state_complete"] = "COMPLETE" in str(
                    source_row.get("state", "")
                )
                audit["error_type"] = type(exc).__name__
                audit["error_message"] = str(exc)
                audit_rows.append(audit)
            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"Processed {processed} trials", flush=True)

    trial_summary = pd.DataFrame(trial_rows)
    iteration_metrics = (
        pd.concat(iteration_frames, ignore_index=True)
        if iteration_frames
        else pd.DataFrame()
    )
    info_summary = (
        pd.concat(info_frames, ignore_index=True)
        if info_frames
        else pd.DataFrame()
    )
    trial_audit = pd.DataFrame(audit_rows)
    run_inventory = pd.DataFrame(inventory_rows)
    analysis_tables = (
        build_analysis_tables(
            trial_summary,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        if not trial_summary.empty
        else {}
    )
    return ReaggregationResult(
        trial_summary=trial_summary,
        iteration_metrics=iteration_metrics,
        info_summary=info_summary,
        trial_audit=trial_audit,
        run_inventory=run_inventory,
        analysis_tables=analysis_tables,
        elapsed_sec=time.perf_counter() - start,
    )
