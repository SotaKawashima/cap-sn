"""Metrics shared by online optimization and offline reanalysis.

The simulator's ``num_selfish`` column is a per-step flow: it records agents
that newly completed selfish behavior at that step.  Each agent can complete
the behavior at most once in a simulation iteration.  The primary objective
therefore sums the flow within each iteration before normalizing by the number
of agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd


OBJECTIVE_NAME = "cumulative_selfish_fraction"
OBJECTIVE_DEFINITION_VERSION = "cumulative_selfish_fraction_v1"
LEGACY_METRIC_NAME = "mean_new_selfish_rate_per_step"

REQUIRED_POP_COLUMNS = ("num_iter", "t", "num_selfish")
REQUIRED_AGENT_COLUMNS = ("num_iter", "agent_idx")


class MetricValidationError(ValueError):
    """Raised when raw simulation data cannot support a valid metric."""


@dataclass(frozen=True)
class SelfishMetricResult:
    """Per-iteration metrics and their aggregate summary."""

    per_iteration: pd.DataFrame
    summary: dict[str, Any]

    @property
    def objective_value(self) -> float:
        return float(self.summary[OBJECTIVE_NAME])


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MetricValidationError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise MetricValidationError(f"{name} must be a positive integer")
    return value


def _validate_integer_column(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if series.isna().any():
        raise MetricValidationError(f"column '{column}' contains missing values")
    if not pd.api.types.is_numeric_dtype(series):
        raise MetricValidationError(f"column '{column}' must be numeric")

    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise MetricValidationError(f"column '{column}' contains non-finite values")
    if not np.equal(values, np.floor(values)).all():
        raise MetricValidationError(f"column '{column}' must contain integers")
    if (values < 0).any():
        raise MetricValidationError(f"column '{column}' must be non-negative")

    return numeric.astype("int64")


def validate_pop_data(
    pop_df: pd.DataFrame,
    *,
    num_agents: int,
    expected_iterations: int | None = None,
) -> pd.DataFrame:
    """Validate and canonically order a ``pop.arrow`` DataFrame.

    The returned frame is a copy sorted by ``num_iter`` and ``t`` with the
    required columns converted to signed 64-bit integers.  Signed integers are
    used so aggregation cannot silently wrap unsigned source values.
    """

    num_agents = _validate_positive_integer(num_agents, "num_agents")
    if expected_iterations is not None:
        expected_iterations = _validate_positive_integer(
            expected_iterations, "expected_iterations"
        )

    if not isinstance(pop_df, pd.DataFrame):
        raise MetricValidationError("pop data must be a pandas DataFrame")
    if pop_df.empty:
        raise MetricValidationError("pop data is empty")

    missing_columns = [
        column for column in REQUIRED_POP_COLUMNS if column not in pop_df.columns
    ]
    if missing_columns:
        raise MetricValidationError(
            f"pop data is missing required columns: {', '.join(missing_columns)}"
        )

    ordered = pop_df.loc[:, REQUIRED_POP_COLUMNS].copy()
    for column in REQUIRED_POP_COLUMNS:
        ordered[column] = _validate_integer_column(ordered, column)

    if ordered.duplicated(["num_iter", "t"]).any():
        duplicates = ordered.loc[
            ordered.duplicated(["num_iter", "t"], keep=False), ["num_iter", "t"]
        ].drop_duplicates()
        preview = duplicates.head(5).to_dict(orient="records")
        raise MetricValidationError(
            f"pop data contains duplicate (num_iter, t) rows: {preview}"
        )

    observed_iterations = set(int(value) for value in ordered["num_iter"].unique())
    if expected_iterations is not None:
        expected = set(range(expected_iterations))
        missing = sorted(expected - observed_iterations)
        unexpected = sorted(observed_iterations - expected)
        if missing or unexpected:
            raise MetricValidationError(
                "iteration IDs do not match expected range "
                f"0..{expected_iterations - 1}; missing={missing}, "
                f"unexpected={unexpected}"
            )

    ordered = ordered.sort_values(["num_iter", "t"], kind="stable").reset_index(
        drop=True
    )

    cumulative = ordered.groupby("num_iter", sort=True)["num_selfish"].sum()
    invalid = cumulative[cumulative > num_agents]
    if not invalid.empty:
        details = {int(index): int(value) for index, value in invalid.items()}
        raise MetricValidationError(
            "cumulative selfish count exceeds num_agents: " f"{details} > {num_agents}"
        )

    return ordered


def _first_threshold_step(
    steps: np.ndarray, cumulative_counts: np.ndarray, threshold: float
) -> int:
    index = int(np.searchsorted(cumulative_counts, threshold, side="left"))
    return int(steps[index])


def _iteration_metrics(
    num_iter: int, group: pd.DataFrame, num_agents: int
) -> dict[str, Any]:
    steps = group["t"].to_numpy(dtype=np.int64)
    counts = group["num_selfish"].to_numpy(dtype=np.int64)
    cumulative_count = int(counts.sum())
    positive = counts > 0

    row: dict[str, Any] = {
        "num_iter": int(num_iter),
        "n_steps": int(len(group)),
        "cumulative_selfish_count": cumulative_count,
        OBJECTIVE_NAME: cumulative_count / num_agents,
        "peak_new_selfish_ratio": float(counts.max()) / num_agents,
        LEGACY_METRIC_NAME: float(counts.mean()) / num_agents,
        "first_selfish_step": pd.NA,
        "last_selfish_step": pd.NA,
        "t50_selfish_step": pd.NA,
        "t90_selfish_step": pd.NA,
        "selfish_timing_centroid": np.nan,
        "selfish_span_steps": pd.NA,
        "active_selfish_steps": int(positive.sum()),
    }

    if cumulative_count == 0:
        return row

    positive_steps = steps[positive]
    first_step = int(positive_steps[0])
    last_step = int(positive_steps[-1])
    cumulative_counts = np.cumsum(counts, dtype=np.int64)

    row.update(
        {
            "first_selfish_step": first_step,
            "last_selfish_step": last_step,
            "t50_selfish_step": _first_threshold_step(
                steps, cumulative_counts, cumulative_count * 0.50
            ),
            "t90_selfish_step": _first_threshold_step(
                steps, cumulative_counts, cumulative_count * 0.90
            ),
            "selfish_timing_centroid": float(np.dot(steps, counts))
            / cumulative_count,
            "selfish_span_steps": last_step - first_step + 1,
        }
    )
    return row


def _mean_or_none(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def compute_selfish_metrics(
    pop_df: pd.DataFrame,
    *,
    num_agents: int,
    expected_iterations: int | None = None,
) -> SelfishMetricResult:
    """Compute the primary objective and explanatory timing metrics.

    ``expected_iterations`` should be supplied for experiment data.  It makes
    a missing zero-event iteration an explicit error instead of silently
    dropping that iteration from the average.
    """

    num_agents = _validate_positive_integer(num_agents, "num_agents")
    ordered = validate_pop_data(
        pop_df,
        num_agents=num_agents,
        expected_iterations=expected_iterations,
    )

    rows = [
        _iteration_metrics(num_iter, group, num_agents)
        for num_iter, group in ordered.groupby("num_iter", sort=True)
    ]
    per_iteration = pd.DataFrame(rows)

    nullable_integer_columns = [
        "first_selfish_step",
        "last_selfish_step",
        "t50_selfish_step",
        "t90_selfish_step",
        "selfish_span_steps",
    ]
    for column in nullable_integer_columns:
        per_iteration[column] = per_iteration[column].astype("Int64")

    n_iterations = int(len(per_iteration))
    n_with_selfish = int((per_iteration["cumulative_selfish_count"] > 0).sum())
    summary: dict[str, Any] = {
        "objective_name": OBJECTIVE_NAME,
        "objective_definition_version": OBJECTIVE_DEFINITION_VERSION,
        "num_agents": num_agents,
        "n_iterations": n_iterations,
        "n_iterations_with_selfish": n_with_selfish,
        "n_zero_selfish_iterations": n_iterations - n_with_selfish,
        OBJECTIVE_NAME: float(per_iteration[OBJECTIVE_NAME].mean()),
        "peak_new_selfish_ratio": float(
            per_iteration["peak_new_selfish_ratio"].mean()
        ),
        LEGACY_METRIC_NAME: float(per_iteration[LEGACY_METRIC_NAME].mean()),
        "mean_first_selfish_step": _mean_or_none(
            per_iteration["first_selfish_step"]
        ),
        "mean_last_selfish_step": _mean_or_none(
            per_iteration["last_selfish_step"]
        ),
        "mean_t50_selfish_step": _mean_or_none(
            per_iteration["t50_selfish_step"]
        ),
        "mean_t90_selfish_step": _mean_or_none(
            per_iteration["t90_selfish_step"]
        ),
        "mean_selfish_timing_centroid": _mean_or_none(
            per_iteration["selfish_timing_centroid"]
        ),
        "mean_selfish_span_steps": _mean_or_none(
            per_iteration["selfish_span_steps"]
        ),
        "mean_active_selfish_steps": float(
            per_iteration["active_selfish_steps"].mean()
        ),
        "mean_recorded_steps": float(per_iteration["n_steps"].mean()),
        "min_recorded_steps": int(per_iteration["n_steps"].min()),
        "max_recorded_steps": int(per_iteration["n_steps"].max()),
    }

    return SelfishMetricResult(per_iteration=per_iteration, summary=summary)


def validate_pop_agent_consistency(
    pop_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    *,
    num_agents: int,
    expected_iterations: int,
) -> pd.DataFrame:
    """Compare population counts with unique selfish actors in agent data.

    The returned table contains one row per expected iteration.  A mismatch is
    treated as invalid raw data because it means the population flow no longer
    has the actor-level interpretation required by the primary objective.
    """

    num_agents = _validate_positive_integer(num_agents, "num_agents")
    expected_iterations = _validate_positive_integer(
        expected_iterations, "expected_iterations"
    )
    ordered_pop = validate_pop_data(
        pop_df,
        num_agents=num_agents,
        expected_iterations=expected_iterations,
    )

    if not isinstance(agent_df, pd.DataFrame):
        raise MetricValidationError("agent data must be a pandas DataFrame")
    missing_columns = [
        column for column in REQUIRED_AGENT_COLUMNS if column not in agent_df.columns
    ]
    if missing_columns:
        raise MetricValidationError(
            f"agent data is missing required columns: {', '.join(missing_columns)}"
        )

    agents = agent_df.loc[:, REQUIRED_AGENT_COLUMNS].copy()
    for column in REQUIRED_AGENT_COLUMNS:
        agents[column] = _validate_integer_column(agents, column)

    if "selfish" in agent_df.columns:
        selfish_flags = agent_df["selfish"]
        if selfish_flags.isna().any() or not selfish_flags.astype(bool).all():
            raise MetricValidationError(
                "agent data contains rows that are not selfish-completion events"
            )

    if not agents.empty:
        unexpected_iterations = sorted(
            set(int(value) for value in agents["num_iter"].unique())
            - set(range(expected_iterations))
        )
        if unexpected_iterations:
            raise MetricValidationError(
                "agent data contains unexpected iteration IDs: "
                f"{unexpected_iterations}"
            )
        invalid_agent_ids = agents.loc[
            agents["agent_idx"] >= num_agents, "agent_idx"
        ].unique()
        if len(invalid_agent_ids):
            raise MetricValidationError(
                "agent data contains agent_idx outside the network: "
                f"{sorted(int(value) for value in invalid_agent_ids)[:10]}"
            )
        duplicate_agents = agents.duplicated(["num_iter", "agent_idx"], keep=False)
        if duplicate_agents.any():
            preview = (
                agents.loc[duplicate_agents, ["num_iter", "agent_idx"]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise MetricValidationError(
                "agent data records a selfish actor more than once per iteration: "
                f"{preview}"
            )

    iteration_index = pd.Index(range(expected_iterations), name="num_iter")
    pop_counts = (
        ordered_pop.groupby("num_iter")["num_selfish"]
        .sum()
        .reindex(iteration_index, fill_value=0)
        .astype("int64")
    )
    if agents.empty:
        agent_counts = pd.Series(0, index=iteration_index, dtype="int64")
    else:
        agent_counts = (
            agents.groupby("num_iter")["agent_idx"]
            .nunique()
            .reindex(iteration_index, fill_value=0)
            .astype("int64")
        )

    comparison = pd.DataFrame(
        {
            "pop_cumulative_selfish_count": pop_counts,
            "agent_unique_selfish_count": agent_counts,
        }
    ).reset_index()
    comparison["difference"] = (
        comparison["pop_cumulative_selfish_count"]
        - comparison["agent_unique_selfish_count"]
    )
    mismatches = comparison[comparison["difference"] != 0]
    if not mismatches.empty:
        raise MetricValidationError(
            "pop and agent selfish counts disagree: "
            f"{mismatches.head(10).to_dict(orient='records')}"
        )
    return comparison


def read_pop_arrow(path: str | PathLike[str]) -> pd.DataFrame:
    """Read a simulator ``pop.arrow`` file without calculating metrics."""

    try:
        return pd.read_feather(path)
    except Exception as exc:  # pandas/pyarrow expose several exception types
        raise MetricValidationError(f"failed to read pop Arrow file '{path}': {exc}") from exc


def compute_selfish_metrics_from_arrow(
    path: str | PathLike[str],
    *,
    num_agents: int,
    expected_iterations: int | None = None,
) -> SelfishMetricResult:
    """Read ``pop.arrow`` and compute metrics through the shared code path."""

    return compute_selfish_metrics(
        read_pop_arrow(path),
        num_agents=num_agents,
        expected_iterations=expected_iterations,
    )
