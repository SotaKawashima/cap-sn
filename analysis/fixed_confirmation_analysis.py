"""Formal analysis helpers for the Stage 4 fixed confirmation experiment."""

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
    compute_selfish_metrics,
    validate_pop_agent_consistency,
)
from analysis.pilot_analysis import hierarchical_mean_interval
from analysis.saved_trial_reanalysis import INFO_COUNT_COLUMNS, summarize_info_data
from experiment_runtime import NETWORKS, sha256_file


@dataclass(frozen=True)
class FixedConfirmationData:
    """Validated Stage 4 data at iteration, run, and diffusion levels."""

    iterations: pd.DataFrame
    runs: pd.DataFrame
    audit: pd.DataFrame
    info_runs: pd.DataFrame
    info_long: pd.DataFrame
    pop_events: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_matches(observed: Any, expected: Any, tolerance: float = 1e-12) -> bool:
    if observed is None or expected is None:
        return observed is None and expected is None
    return math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
    )


def _manifest_errors(manifest: dict[str, Any], spec: Any) -> list[str]:
    observed = {
        "stage": manifest.get("stage"),
        "run_type": manifest.get("run_type"),
        "network": manifest.get("network", {}).get("id"),
        "condition_id": manifest.get("intervention", {}).get("condition_id"),
        "intervention_enabled": manifest.get("intervention", {}).get("enabled"),
        "simulator_seed": manifest.get("runtime", {}).get("simulator_seed"),
        "iterations": manifest.get("runtime", {}).get("iteration_count"),
        "objective_name": manifest.get("objective", {}).get("name"),
        "objective_version": manifest.get("objective", {}).get(
            "definition_version"
        ),
    }
    expected = {
        "stage": "stage4_fixed_confirmation",
        "run_type": "fixed_condition",
        "network": spec.network,
        "condition_id": spec.condition_id,
        "intervention_enabled": bool(spec.intervention_enabled),
        "simulator_seed": int(spec.simulator_seed),
        "iterations": int(spec.iterations),
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_DEFINITION_VERSION,
    }
    errors = [
        f"{field}={observed[field]!r}, expected={value!r}"
        for field, value in expected.items()
        if observed[field] != value
    ]
    applied = manifest.get("intervention", {}).get("applied_parameters")
    if spec.intervention_enabled:
        if not isinstance(applied, dict):
            errors.append("applied_parameters is missing")
        else:
            for field in ("certainty", "effectiveness"):
                if not _float_matches(applied.get(field), getattr(spec, field)):
                    errors.append(
                        f"applied_{field}={applied.get(field)!r}, "
                        f"expected={getattr(spec, field)!r}"
                    )
        expected_opinion = getattr(spec, "opinion_csv", None)
        if expected_opinion is not None:
            expected_hash = getattr(spec, "opinion_sha256", None)
            if expected_hash is None:
                expected_hash = sha256_file(expected_opinion)
            intervention = manifest.get("intervention", {})
            if intervention.get("opinion_mode") != "existing_csv":
                errors.append(
                    "opinion_mode="
                    f"{intervention.get('opinion_mode')!r}, "
                    "expected='existing_csv'"
                )
            for field in ("opinion_source", "opinion_csv"):
                entry = intervention.get(field) or {}
                if entry.get("sha256") != expected_hash:
                    errors.append(
                        f"{field}_sha256={entry.get('sha256')!r}, "
                        f"expected={expected_hash!r}"
                    )
    elif applied is not None:
        errors.append("no-intervention run has applied parameters")
    return errors


def _metric_errors(
    saved: pd.DataFrame,
    recalculated: pd.DataFrame,
    *,
    expected_iterations: int,
) -> list[str]:
    errors: list[str] = []
    if len(saved) != expected_iterations:
        errors.append(
            f"metrics row count={len(saved)}, expected={expected_iterations}"
        )
        return errors
    observed_ids = set(pd.to_numeric(saved["num_iter"]).astype(int))
    if observed_ids != set(range(expected_iterations)):
        errors.append("metrics iteration IDs do not match the expected range")
        return errors
    left = saved.sort_values("num_iter").reset_index(drop=True)
    right = recalculated.sort_values("num_iter").reset_index(drop=True)
    shared_columns = [
        column
        for column in right.columns
        if column in left.columns and column != "num_iter"
    ]
    for column in shared_columns:
        observed = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        expected = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if not np.allclose(
            observed, expected, rtol=0.0, atol=1e-12, equal_nan=True
        ):
            errors.append(f"saved metric does not reproduce: {column}")
    return errors


def _summarize_stage4_info(
    info: pd.DataFrame,
    *,
    intervention_enabled: bool,
    num_agents: int,
    expected_iterations: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Represent the intentionally absent behavior-guiding stream as zero."""

    observed_labels = set(pd.to_numeric(info["info_label"]).astype(int).unique())
    if intervention_enabled or observed_labels == {0, 1, 2, 3}:
        return summarize_info_data(
            info,
            num_agents=num_agents,
            expected_iterations=expected_iterations,
        )
    if observed_labels != {0, 1, 2}:
        raise ValueError(
            "no-intervention info labels must be [0, 1, 2]; "
            f"observed={sorted(observed_labels)}"
        )
    zero_rows = pd.DataFrame(
        {
            "num_iter": np.arange(expected_iterations, dtype=np.int64),
            "t": np.zeros(expected_iterations, dtype=np.int64),
            "info_label": np.full(expected_iterations, 3, dtype=np.int64),
            **{
                column: np.zeros(expected_iterations, dtype=np.int64)
                for column in INFO_COUNT_COLUMNS
            },
        }
    )
    completed = pd.concat([info, zero_rows], ignore_index=True)
    return summarize_info_data(
        completed,
        num_agents=num_agents,
        expected_iterations=expected_iterations,
    )


def load_fixed_confirmation(
    experiment_root: str | Path,
    *,
    expected_specs: Sequence[Any],
) -> FixedConfirmationData:
    """Read all Stage 4 runs and reproduce metrics from raw Arrow files."""

    root = Path(experiment_root).resolve()
    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    info_run_rows: list[dict[str, Any]] = []
    info_long_frames: list[pd.DataFrame] = []
    pop_event_frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for spec in expected_specs:
        run_dir = root / spec.relative_run_dir
        paths = {
            name: run_dir / filename
            for name, filename in {
                "manifest": "manifest.json",
                "metrics": "metrics.csv",
                "pop": "pop.arrow",
                "info": "info.arrow",
                "agent": "agent.arrow",
            }.items()
        }
        audit: dict[str, Any] = {
            "run_key": spec.key,
            "network": spec.network,
            "condition_id": spec.condition_id,
            "simulator_seed": int(spec.simulator_seed),
            "files_present": False,
            "manifest_valid": False,
            "metrics_reproduced": False,
            "pop_agent_consistent": False,
            "info_valid": False,
            "valid": False,
            "error": None,
        }
        try:
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing files: {missing}")
            audit["files_present"] = True

            manifest = _read_json(paths["manifest"])
            if manifest.get("status") != "completed":
                raise ValueError(f"run status is {manifest.get('status')!r}")
            manifest_errors = _manifest_errors(manifest, spec)
            if manifest_errors:
                raise ValueError("; ".join(manifest_errors))
            audit["manifest_valid"] = True

            num_agents = int(NETWORKS[spec.network].num_agents)
            pop = pd.read_feather(paths["pop"])
            info = pd.read_feather(paths["info"])
            agent = pd.read_feather(paths["agent"])
            result = compute_selfish_metrics(
                pop,
                num_agents=num_agents,
                expected_iterations=int(spec.iterations),
            )
            saved_metrics = pd.read_csv(paths["metrics"])
            metric_errors = _metric_errors(
                saved_metrics,
                result.per_iteration,
                expected_iterations=int(spec.iterations),
            )
            manifest_objective = manifest.get("objective", {}).get("value")
            if not _float_matches(manifest_objective, result.objective_value):
                metric_errors.append(
                    "manifest objective does not match raw recalculation"
                )
            if metric_errors:
                raise ValueError("; ".join(metric_errors))
            audit["metrics_reproduced"] = True

            validate_pop_agent_consistency(
                pop,
                agent,
                num_agents=num_agents,
                expected_iterations=int(spec.iterations),
            )
            audit["pop_agent_consistent"] = True

            info_summary, info_wide = _summarize_stage4_info(
                info,
                intervention_enabled=bool(spec.intervention_enabled),
                num_agents=num_agents,
                expected_iterations=int(spec.iterations),
            )
            audit["info_valid"] = True

            identifiers = {
                "run_key": spec.key,
                "network": spec.network,
                "condition_id": spec.condition_id,
                "condition_role": spec.condition_role,
                "intervention_enabled": bool(spec.intervention_enabled),
                "certainty": spec.certainty,
                "effectiveness": spec.effectiveness,
                "simulator_seed": int(spec.simulator_seed),
                "num_agents": num_agents,
                "iterations": int(spec.iterations),
            }
            metrics = result.per_iteration.copy()
            for position, (field, value) in enumerate(identifiers.items()):
                metrics.insert(position, field, value)
            iteration_frames.append(metrics)

            info_summary = info_summary.copy()
            for position, (field, value) in enumerate(identifiers.items()):
                info_summary.insert(position, field, value)
            info_long_frames.append(info_summary)
            info_run_rows.append({**identifiers, **info_wide})

            events = pop.loc[:, ["num_iter", "t", "num_selfish"]].copy()
            for position, (field, value) in enumerate(identifiers.items()):
                events.insert(position, field, value)
            pop_event_frames.append(events)

            run_rows.append(
                {
                    **identifiers,
                    "iterations": int(spec.iterations),
                    "objective": result.objective_value,
                    "simulation_sec": float(manifest["timing_sec"]["simulation"]),
                    "total_sec": float(manifest["timing_sec"]["total"]),
                    "git_commit": str(manifest.get("git", {}).get("commit", "")),
                    "manifest_path": paths["manifest"].as_posix(),
                    "metrics_path": paths["metrics"].as_posix(),
                }
            )
            audit["valid"] = True
        except Exception as exc:  # Keep an audit row for every protocol run.
            audit["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"{spec.key}: {audit['error']}")
        audit_rows.append(audit)

    audit_frame = pd.DataFrame(audit_rows)
    if failures:
        raise ValueError(
            "Stage 4 data audit failed:\n" + "\n".join(failures[:25])
        )
    return FixedConfirmationData(
        iterations=pd.concat(iteration_frames, ignore_index=True),
        runs=pd.DataFrame(run_rows),
        audit=audit_frame,
        info_runs=pd.DataFrame(info_run_rows),
        info_long=pd.concat(info_long_frames, ignore_index=True),
        pop_events=pd.concat(pop_event_frames, ignore_index=True),
    )


def replace_fixed_confirmation_condition(
    original: FixedConfirmationData,
    correction: FixedConfirmationData,
    *,
    condition_id: str,
) -> FixedConfirmationData:
    """Replace one complete Stage 4 condition in every analysis table."""

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

    def replace(original_frame: pd.DataFrame, corrected_frame: pd.DataFrame) -> pd.DataFrame:
        retained = original_frame[
            ~original_frame["run_key"].astype(str).isin(original_keys)
        ]
        corrected = corrected_frame.loc[:, original_frame.columns]
        merged = pd.concat([retained, corrected], ignore_index=True)
        return merged

    merged = FixedConfirmationData(
        iterations=replace(original.iterations, correction.iterations),
        runs=replace(original.runs, correction.runs),
        audit=replace(original.audit, correction.audit),
        info_runs=replace(original.info_runs, correction.info_runs),
        info_long=replace(original.info_long, correction.info_long),
        pop_events=replace(original.pop_events, correction.pop_events),
    )
    if merged.runs["run_key"].duplicated().any():
        raise ValueError("merged Stage 4 run inventory contains duplicate keys")
    if merged.audit["run_key"].duplicated().any():
        raise ValueError("merged Stage 4 audit contains duplicate keys")
    iteration_key = ["run_key", "num_iter"]
    if merged.iterations.duplicated(iteration_key).any():
        raise ValueError("merged Stage 4 iteration table contains duplicate rows")
    if set(merged.runs["run_key"].astype(str)) != set(
        merged.iterations["run_key"].astype(str)
    ):
        raise ValueError("merged Stage 4 run and iteration keys differ")
    return FixedConfirmationData(
        iterations=merged.iterations.sort_values(iteration_key).reset_index(drop=True),
        runs=merged.runs.sort_values("run_key").reset_index(drop=True),
        audit=merged.audit.sort_values("run_key").reset_index(drop=True),
        info_runs=merged.info_runs.sort_values("run_key").reset_index(drop=True),
        info_long=merged.info_long.sort_values(
            ["run_key", "num_iter", "info_label"]
        ).reset_index(drop=True),
        pop_events=merged.pop_events.sort_values(
            ["run_key", "num_iter", "t"]
        ).reset_index(drop=True),
    )


def build_seed_summary(iterations: pd.DataFrame) -> pd.DataFrame:
    grouped = iterations.groupby(
        [
            "network",
            "condition_id",
            "condition_role",
            "intervention_enabled",
            "certainty",
            "effectiveness",
            "simulator_seed",
            "num_agents",
        ],
        dropna=False,
        sort=True,
    )[OBJECTIVE_NAME]
    return grouped.agg(n="size", mean="mean", std="std", sem="sem").reset_index()


def build_condition_summary(
    iterations: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "network",
        "condition_id",
        "condition_role",
        "intervention_enabled",
        "certainty",
        "effectiveness",
        "num_agents",
    ]
    for offset, (key, group) in enumerate(
        iterations.groupby(keys, dropna=False, sort=True)
    ):
        estimate = hierarchical_mean_interval(
            group,
            block_column="simulator_seed",
            value_column=OBJECTIVE_NAME,
            repetitions=repetitions,
            seed=seed + offset,
        )
        block_means = group.groupby("simulator_seed")[OBJECTIVE_NAME].mean()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_iterations": int(len(group)),
                "between_seed_sd": float(block_means.std()),
                **estimate,
            }
        )
    return pd.DataFrame(rows)


def _paired_bootstrap_effect(
    frame: pd.DataFrame,
    *,
    reference_column: str,
    candidate_column: str,
    block_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    data = frame[[block_column, reference_column, candidate_column]].dropna()
    blocks = [
        group[[reference_column, candidate_column]].to_numpy(dtype=float)
        for _, group in data.groupby(block_column, sort=True)
    ]
    if len(blocks) < 2 or any(len(block) < 2 for block in blocks):
        raise ValueError("paired bootstrap requires at least two nontrivial blocks")

    reference_estimate = float(np.mean([block[:, 0].mean() for block in blocks]))
    candidate_estimate = float(np.mean([block[:, 1].mean() for block in blocks]))
    delta_estimate = reference_estimate - candidate_estimate
    eta_estimate = delta_estimate / reference_estimate

    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    reference_samples = np.zeros(repetitions, dtype=float)
    candidate_samples = np.zeros(repetitions, dtype=float)
    for position in range(n_blocks):
        selected_blocks = rng.integers(0, n_blocks, size=repetitions)
        reference_contribution = np.empty(repetitions, dtype=float)
        candidate_contribution = np.empty(repetitions, dtype=float)
        for block_index, block in enumerate(blocks):
            rows = np.flatnonzero(selected_blocks == block_index)
            if len(rows) == 0:
                continue
            indices = rng.integers(0, len(block), size=(len(rows), len(block)))
            reference_contribution[rows] = block[indices, 0].mean(axis=1)
            candidate_contribution[rows] = block[indices, 1].mean(axis=1)
        reference_samples += reference_contribution / n_blocks
        candidate_samples += candidate_contribution / n_blocks
    delta_samples = reference_samples - candidate_samples
    eta_samples = delta_samples / reference_samples
    delta_low, delta_high = np.quantile(delta_samples, [0.025, 0.975])
    eta_low, eta_high = np.quantile(eta_samples, [0.025, 0.975])

    if delta_low > 0:
        interpretation = "reduction"
    elif delta_estimate > 0:
        interpretation = "reduction_tendency_with_uncertainty"
    elif delta_high < 0:
        interpretation = "increase"
    else:
        interpretation = "no_clear_reduction"
    return {
        "n_blocks": n_blocks,
        "n_per_block": min(len(block) for block in blocks),
        "reference_estimate": reference_estimate,
        "candidate_estimate": candidate_estimate,
        "absolute_suppression": delta_estimate,
        "absolute_ci_low": float(delta_low),
        "absolute_ci_high": float(delta_high),
        "relative_suppression": eta_estimate,
        "relative_ci_low": float(eta_low),
        "relative_ci_high": float(eta_high),
        "interpretation": interpretation,
    }


def _wide_iterations(iterations: pd.DataFrame) -> pd.DataFrame:
    key = ["network", "simulator_seed", "num_iter"]
    if iterations.duplicated(key + ["condition_id"]).any():
        raise ValueError("iteration table contains duplicate paired observations")
    return iterations.pivot(
        index=key,
        columns="condition_id",
        values=OBJECTIVE_NAME,
    ).reset_index()


def build_comparison_table(
    iterations: pd.DataFrame,
    *,
    comparisons: Sequence[dict[str, Any]],
    family: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    wide = _wide_iterations(iterations)
    rows: list[dict[str, Any]] = []
    offset = 0
    for comparison in comparisons:
        reference = str(comparison["reference"])
        candidate = str(comparison["candidate"])
        if reference not in wide or candidate not in wide:
            raise ValueError(f"comparison condition is missing: {comparison['id']}")
        for network, group in wide.groupby("network", sort=True):
            result = _paired_bootstrap_effect(
                group,
                reference_column=reference,
                candidate_column=candidate,
                block_column="simulator_seed",
                repetitions=repetitions,
                seed=seed + offset,
            )
            num_agents = int(NETWORKS[str(network)].num_agents)
            rows.append(
                {
                    "family": family,
                    "network": network,
                    "comparison_id": str(comparison["id"]),
                    "reference": reference,
                    "candidate": candidate,
                    "estimand": comparison.get("estimand"),
                    "num_agents": num_agents,
                    **result,
                    "equivalent_agents": result["absolute_suppression"]
                    * num_agents,
                    "equivalent_agents_ci_low": result["absolute_ci_low"]
                    * num_agents,
                    "equivalent_agents_ci_high": result["absolute_ci_high"]
                    * num_agents,
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def build_all_vs_none(
    iterations: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        {
            "id": f"{condition['id']}_vs_none",
            "reference": "none",
            "candidate": condition["id"],
            "estimand": "descriptive_intervention_effect",
        }
        for condition in protocol["design"]["conditions"]
        if bool(condition["enabled"])
    ]
    return build_comparison_table(
        iterations,
        comparisons=comparisons,
        family="all_enabled_vs_none_exploratory",
        repetitions=repetitions,
        seed=seed,
    )


def build_factorial_tables(
    iterations: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factorial = protocol["factorial_contrasts"]
    comparisons: list[dict[str, Any]] = []
    for factor_name in ("effectiveness_low_to_high", "certainty_low_to_high"):
        for comparison in factorial[factor_name]:
            comparisons.append({**comparison, "estimand": factor_name})
    contrasts = build_comparison_table(
        iterations,
        comparisons=comparisons,
        family="pre_specified_factorial_contrast",
        repetitions=repetitions,
        seed=seed,
    )

    interaction = factorial["corner_interaction"]
    low = interaction["effectiveness_contrast_at_low_certainty"]
    high = interaction["effectiveness_contrast_at_high_certainty"]
    wide = _wide_iterations(iterations)
    required = {
        low["reference"],
        low["candidate"],
        high["reference"],
        high["candidate"],
    }
    if not required.issubset(wide.columns):
        raise ValueError("corner interaction conditions are missing")
    wide = wide.copy()
    wide["interaction"] = (
        wide[high["reference"]]
        - wide[high["candidate"]]
        - wide[low["reference"]]
        + wide[low["candidate"]]
    )
    rows: list[dict[str, Any]] = []
    for offset, (network, group) in enumerate(wide.groupby("network", sort=True)):
        result = hierarchical_mean_interval(
            group,
            block_column="simulator_seed",
            value_column="interaction",
            repetitions=repetitions,
            seed=seed + 5000 + offset,
        )
        if result["ci_low"] > 0 or result["ci_high"] < 0:
            interpretation = "interaction_detected"
        else:
            interpretation = "interaction_uncertain"
        rows.append(
            {
                "family": "pre_specified_corner_interaction",
                "network": network,
                "comparison_id": interaction["id"],
                "definition": "effectiveness_at_c100_minus_effectiveness_at_c050",
                "num_agents": int(NETWORKS[str(network)].num_agents),
                **result,
                "equivalent_agents": result["estimate"]
                * int(NETWORKS[str(network)].num_agents),
                "interpretation": interpretation,
            }
        )
    return contrasts, pd.DataFrame(rows)


def build_information_summary(info_long: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "network",
        "condition_id",
        "condition_role",
        "intervention_enabled",
        "certainty",
        "effectiveness",
        "num_agents",
        "info_label",
        "info_name",
    ]
    sum_columns = [column for column in info_long if column.endswith("_sum")]
    data = info_long.copy()
    data["normalization_denominator"] = (
        data["num_agents"] * data["iterations"]
    )
    grouped = data.groupby(keys, dropna=False, sort=True)
    result = grouped[sum_columns].sum().reset_index()
    denominator = grouped["normalization_denominator"].sum().to_numpy(float)
    for column in sum_columns:
        result[column.replace("_sum", "_per_agent_iteration")] = (
            result[column] / denominator
        )
    return result


def build_time_series_summary(pop_events: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive cumulative curves with final values carried forward."""

    events = pop_events.copy()
    events["num_selfish"] = pd.to_numeric(events["num_selfish"]).astype(float)
    max_t_by_network = events.groupby("network")["t"].max().astype(int).to_dict()
    series_frames: list[pd.DataFrame] = []
    keys = [
        "network",
        "condition_id",
        "simulator_seed",
        "num_iter",
        "num_agents",
    ]
    for key, group in events.groupby(keys, sort=True):
        identifiers = dict(zip(keys, key, strict=True))
        max_t = int(max_t_by_network[str(identifiers["network"])])
        values = (
            group.groupby("t", sort=True)["num_selfish"].sum().cumsum()
            / int(identifiers["num_agents"])
        )
        values = values.reindex(range(max_t + 1)).ffill().fillna(0.0)
        frame = pd.DataFrame(
            {
                **identifiers,
                "t": values.index.astype(int),
                "cumulative_selfish_fraction": values.to_numpy(float),
            }
        )
        series_frames.append(frame)
    panel = pd.concat(series_frames, ignore_index=True)
    return (
        panel.groupby(["network", "condition_id", "t"], sort=True)[
            "cumulative_selfish_fraction"
        ]
        .agg(n="size", mean="mean", std="std", sem="sem")
        .reset_index()
    )


def build_stage4_decision(
    condition_summary: pd.DataFrame,
    primary_comparisons: pd.DataFrame,
    all_vs_none: pd.DataFrame,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for network in sorted(condition_summary["network"].unique()):
        conditions = condition_summary[condition_summary["network"] == network]
        enabled = conditions[conditions["intervention_enabled"].astype(bool)]
        none = conditions[conditions["condition_id"] == "none"].iloc[0]
        best = enabled.sort_values("estimate").iloc[0]
        effects = all_vs_none[all_vs_none["network"] == network]
        primary_none = primary_comparisons[
            (primary_comparisons["network"] == network)
            & (primary_comparisons["reference"] == "none")
        ]
        confirmed_reductions = int((effects["absolute_ci_low"] > 0).sum())
        confirmed_increases = int((effects["absolute_ci_high"] < 0).sum())
        if (primary_none["absolute_ci_low"] > 0).any():
            evidence = "pre_specified_intervention_reduction"
            action = "proceed_to_network_specific_reoptimization"
        elif (primary_none["absolute_ci_high"] < 0).all():
            evidence = "pre_specified_interventions_increase_objective"
            action = "pause_suppression_reoptimization_and_reconsider_intervention"
        else:
            evidence = "pre_specified_intervention_effect_inconclusive"
            action = "conditional_reoptimization_only"
        decisions.append(
            {
                "network": network,
                "none_estimate": float(none["estimate"]),
                "exploratory_best_condition": str(best["condition_id"]),
                "exploratory_best_estimate": float(best["estimate"]),
                "exploratory_best_absolute_suppression": float(
                    none["estimate"] - best["estimate"]
                ),
                "enabled_condition_range": float(
                    enabled["estimate"].max() - enabled["estimate"].min()
                ),
                "enabled_conditions_with_ci_reduction_vs_none": confirmed_reductions,
                "enabled_conditions_with_ci_increase_vs_none": confirmed_increases,
                "evidence": evidence,
                "stage6_action": action,
            }
        )
    return {
        "status": "completed",
        "global_conclusion": "network_specific_decision_required",
        "minimum_important_effect_threshold": None,
        "post_hoc_best_condition_status": "exploratory_only",
        "network_decisions": decisions,
    }
