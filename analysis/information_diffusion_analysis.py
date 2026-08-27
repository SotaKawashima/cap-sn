"""Stage 10 information-diffusion reanalysis and confirmation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from analysis.optimization_metrics import (
    OBJECTIVE_NAME,
    compute_selfish_metrics,
)
from experiment_runtime import NETWORKS, REPO_ROOT, sha256_file
from run_stage10_information_diffusion import (
    EXECUTION_PLAN_NAME,
    STAGE,
    InformationDiffusionRunSpec,
    build_specs,
)


INFO_NAMES = {
    0: "misinformation",
    1: "corrective",
    2: "observational",
    3: "behavior_guiding",
}
INFO_COUNT_COLUMNS = {
    "shared": "num_shared",
    "viewed": "num_viewed",
    "fst_viewed": "num_fst_viewed",
}
REQUIRED_INFO_COLUMNS = (
    "num_iter",
    "t",
    "info_label",
    *INFO_COUNT_COLUMNS.values(),
)
OUTCOME_COLUMNS = (OBJECTIVE_NAME, "peak_new_selfish_ratio")


def information_indicator_columns() -> list[str]:
    columns = [
        f"{info_name}_{metric}_per_agent_iteration"
        for info_name in INFO_NAMES.values()
        for metric in INFO_COUNT_COLUMNS
    ]
    columns.extend(
        f"corrective_to_misinformation_{metric}_ratio"
        for metric in INFO_COUNT_COLUMNS
    )
    return columns


INFORMATION_INDICATORS = information_indicator_columns()


@dataclass(frozen=True)
class ExistingInformationData:
    stage2_trials: pd.DataFrame
    stage2_info: pd.DataFrame
    stage4_iterations: pd.DataFrame
    stage4_info_runs: pd.DataFrame
    stage4_run_inventory: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class RawConditionData:
    iterations: pd.DataFrame
    runs: pd.DataFrame
    info_events: pd.DataFrame
    pop_events: pd.DataFrame
    audit: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _verified_artifact(root: Path, spec: dict[str, Any], role: str) -> Path:
    path = (root / str(spec["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} is missing: {path}")
    expected = str(spec["sha256"]).lower()
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{role} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return path


def load_existing_information_data(
    protocol: dict[str, Any],
) -> ExistingInformationData:
    """Load and hash-check the completed Stage 2 and corrected Stage 4 tables."""

    existing = protocol["existing_data"]
    audit_rows: list[dict[str, Any]] = []

    stage2_spec = existing["stage2_saved_trials"]
    stage2_root = _resolve_repo_path(stage2_spec["analysis_root"])
    stage2_manifest_path = _verified_artifact(
        stage2_root, stage2_spec["manifest"], "Stage 2 manifest"
    )
    stage2_trial_path = _verified_artifact(
        stage2_root, stage2_spec["trial_summary"], "Stage 2 trial summary"
    )
    stage2_info_path = _verified_artifact(
        stage2_root, stage2_spec["info_summary"], "Stage 2 info summary"
    )
    stage2_manifest = _read_json(stage2_manifest_path)
    if stage2_manifest.get("status") != "completed":
        raise ValueError("Stage 2 source analysis is not completed")
    stage2_trials = pd.read_parquet(stage2_trial_path)
    stage2_info = pd.read_parquet(stage2_info_path)
    if len(stage2_trials) != int(stage2_spec["trial_summary"]["expected_rows"]):
        raise ValueError("Stage 2 trial-summary row count is invalid")
    if len(stage2_info) != int(stage2_spec["info_summary"]["expected_rows"]):
        raise ValueError("Stage 2 info-summary row count is invalid")
    if stage2_trials["trial_id"].duplicated().any():
        raise ValueError("Stage 2 trial IDs are duplicated")
    info_counts = stage2_info.groupby("trial_id")["info_label"].nunique()
    if len(info_counts) != len(stage2_trials) or not (info_counts == 4).all():
        raise ValueError("Stage 2 does not contain four info labels per trial")
    audit_rows.append(
        {
            "source": "stage2_saved_trials",
            "status": "valid",
            "run_count": int(stage2_manifest["counts"]["source_runs"]),
            "iteration_count": int(len(stage2_trials)),
            "detail": "3600 saved optimization trials; fixed simulator seed",
        }
    )

    stage4_spec = existing["stage4_fixed_confirmation"]
    stage4_root = _resolve_repo_path(stage4_spec["analysis_root"])
    stage4_manifest_path = _verified_artifact(
        stage4_root,
        stage4_spec["analysis_manifest"],
        "corrected Stage 4 analysis manifest",
    )
    stage4_iteration_path = _verified_artifact(
        stage4_root,
        stage4_spec["iteration_metrics"],
        "corrected Stage 4 iteration metrics",
    )
    stage4_info_path = _verified_artifact(
        stage4_root,
        stage4_spec["information_run_summary"],
        "corrected Stage 4 information summary",
    )
    stage4_inventory_path = _verified_artifact(
        stage4_root,
        stage4_spec["run_inventory"],
        "corrected Stage 4 run inventory",
    )
    stage4_manifest = _read_json(stage4_manifest_path)
    if stage4_manifest.get("status") != "completed":
        raise ValueError("corrected Stage 4 source analysis is not completed")
    override = stage4_manifest.get("condition_override", {})
    if override.get("condition_id") != "prior_high" or int(
        override.get("run_count", -1)
    ) != 15:
        raise ValueError("Stage 4 source does not contain the corrected prior_high")
    stage4_iterations = pd.read_parquet(stage4_iteration_path)
    stage4_info_runs = pd.read_csv(stage4_info_path)
    stage4_inventory = pd.read_csv(stage4_inventory_path)
    if len(stage4_iterations) != int(stage4_spec["expected_formal_rows"]):
        raise ValueError("Stage 4 iteration row count is invalid")
    if len(stage4_info_runs) != 210 or len(stage4_inventory) != 210:
        raise ValueError("Stage 4 run-level tables must contain 210 rows")
    if stage4_inventory["run_key"].duplicated().any():
        raise ValueError("Stage 4 run keys are duplicated")
    corrected = stage4_inventory[stage4_inventory["condition_id"] == "prior_high"]
    expected_override_name = Path(stage4_spec["prior_high_override_root"]).name
    if len(corrected) != 15 or not corrected["manifest_path"].astype(str).str.contains(
        expected_override_name, regex=False
    ).all():
        raise ValueError("Stage 4 prior_high inventory is not the corrected source")
    audit_rows.append(
        {
            "source": "stage4_fixed_confirmation_corrected",
            "status": "valid",
            "run_count": int(len(stage4_inventory)),
            "iteration_count": int(len(stage4_iterations)),
            "detail": "corrected prior_high 15 runs replace original runs",
        }
    )

    required_stage2 = {
        "trial_id",
        "network",
        "method",
        "applied_certainty",
        "applied_effectiveness",
        "j_cum",
        "j_peak",
        *INFORMATION_INDICATORS,
    }
    missing_stage2 = required_stage2 - set(stage2_trials.columns)
    if missing_stage2:
        raise ValueError(f"Stage 2 summary is missing columns: {sorted(missing_stage2)}")
    required_stage4 = {
        "run_key",
        "network",
        "condition_id",
        "certainty",
        "effectiveness",
        *INFORMATION_INDICATORS,
    }
    missing_stage4 = required_stage4 - set(stage4_info_runs.columns)
    if missing_stage4:
        raise ValueError(f"Stage 4 info summary is missing: {sorted(missing_stage4)}")

    return ExistingInformationData(
        stage2_trials=stage2_trials,
        stage2_info=stage2_info,
        stage4_iterations=stage4_iterations,
        stage4_info_runs=stage4_info_runs,
        stage4_run_inventory=stage4_inventory,
        audit=pd.DataFrame(audit_rows),
    )


def _validate_integer_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame[column].isna().any():
        raise ValueError(f"info column {column!r} contains missing values")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"info column {column!r} contains non-finite values")
    if not np.equal(values, np.floor(values)).all() or (values < 0).any():
        raise ValueError(f"info column {column!r} must be non-negative integers")
    return numeric.astype("int64")


def validate_info_arrow(
    info: pd.DataFrame,
    *,
    expected_iterations: int,
    intervention_enabled: bool,
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_INFO_COLUMNS if column not in info]
    if missing:
        raise ValueError(f"info Arrow is missing columns: {missing}")
    if info.empty:
        raise ValueError("info Arrow is empty")
    data = info.loc[:, REQUIRED_INFO_COLUMNS].copy()
    for column in REQUIRED_INFO_COLUMNS:
        data[column] = _validate_integer_column(data, column)
    if data.duplicated(["num_iter", "t", "info_label"]).any():
        raise ValueError("info Arrow contains duplicate iteration/time/label rows")
    observed_iterations = set(data["num_iter"].unique())
    if observed_iterations != set(range(expected_iterations)):
        raise ValueError("info Arrow iteration IDs are incomplete")
    labels = set(data["info_label"].unique())
    expected_labels = set(INFO_NAMES) if intervention_enabled else {0, 1, 2}
    if labels != expected_labels:
        raise ValueError(
            f"info Arrow labels are {sorted(labels)}, expected {sorted(expected_labels)}"
        )
    return data.sort_values(["num_iter", "t", "info_label"]).reset_index(drop=True)


def summarize_info_by_iteration(
    info: pd.DataFrame,
    *,
    num_agents: int,
    expected_iterations: int,
    intervention_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one wide row per iteration and validated event rows."""

    data = validate_info_arrow(
        info,
        expected_iterations=expected_iterations,
        intervention_enabled=intervention_enabled,
    )
    grouped = (
        data.groupby(["num_iter", "info_label"], sort=True)[
            list(INFO_COUNT_COLUMNS.values())
        ]
        .sum()
        .reindex(
            pd.MultiIndex.from_product(
                [range(expected_iterations), sorted(INFO_NAMES)],
                names=["num_iter", "info_label"],
            ),
            fill_value=0,
        )
        .reset_index()
    )
    result = pd.DataFrame({"num_iter": np.arange(expected_iterations, dtype=int)})
    for label, info_name in INFO_NAMES.items():
        subset = grouped[grouped["info_label"] == label].sort_values("num_iter")
        for metric, source_column in INFO_COUNT_COLUMNS.items():
            counts = subset[source_column].to_numpy(dtype=float)
            result[f"{info_name}_{metric}_count"] = counts
            result[f"{info_name}_{metric}_per_agent_iteration"] = (
                counts / int(num_agents)
            )
    for metric in INFO_COUNT_COLUMNS:
        numerator = result[f"corrective_{metric}_count"].to_numpy(float)
        denominator = result[f"misinformation_{metric}_count"].to_numpy(float)
        result[f"corrective_to_misinformation_{metric}_ratio"] = np.divide(
            numerator,
            denominator,
            out=np.full(len(result), np.nan, dtype=float),
            where=denominator > 0,
        )
        result[f"misinformation_{metric}_denominator_zero"] = denominator == 0
    return result, data


def _metrics_match(saved: pd.DataFrame, recalculated: pd.DataFrame) -> bool:
    if len(saved) != len(recalculated) or "num_iter" not in saved:
        return False
    left = saved.sort_values("num_iter").reset_index(drop=True)
    right = recalculated.sort_values("num_iter").reset_index(drop=True)
    if any(column not in left for column in right.columns):
        return False
    if not np.array_equal(
        pd.to_numeric(left["num_iter"]).to_numpy(int),
        pd.to_numeric(right["num_iter"]).to_numpy(int),
    ):
        return False
    for column in right.columns:
        if column == "num_iter" or column not in left:
            continue
        observed = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        expected = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if not np.allclose(observed, expected, atol=1e-12, rtol=0, equal_nan=True):
            return False
    return True


def _manifest_matches(
    manifest: dict[str, Any],
    *,
    stage: str,
    network: str,
    condition_id: str,
    simulator_seed: int,
    iterations: int,
    certainty: float | None,
    effectiveness: float | None,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("status") != "completed":
        errors.append(f"status={manifest.get('status')!r}")
    if manifest.get("stage") != stage:
        errors.append(f"stage={manifest.get('stage')!r}")
    if manifest.get("network", {}).get("id") != network:
        errors.append("network mismatch")
    runtime = manifest.get("runtime", {})
    if int(runtime.get("simulator_seed", -1)) != int(simulator_seed):
        errors.append("simulator seed mismatch")
    if int(runtime.get("iteration_count", -1)) != int(iterations):
        errors.append("iteration count mismatch")
    intervention = manifest.get("intervention", {})
    if intervention.get("condition_id") != condition_id:
        errors.append("condition ID mismatch")
    applied = intervention.get("applied_parameters")
    if certainty is None or effectiveness is None:
        if intervention.get("enabled") is not False or applied is not None:
            errors.append("no-intervention metadata mismatch")
    else:
        if intervention.get("enabled") is not True or not isinstance(applied, dict):
            errors.append("intervention metadata is missing")
        else:
            if not math.isclose(
                float(applied.get("certainty", math.nan)),
                certainty,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                errors.append("certainty mismatch")
            if not math.isclose(
                float(applied.get("effectiveness", math.nan)),
                effectiveness,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                errors.append("effectiveness mismatch")
    return errors


def _load_raw_run(
    *,
    run_dir: Path,
    stage: str,
    source: str,
    network: str,
    condition_id: str,
    condition_group: str,
    condition_role: str,
    simulator_seed: int,
    iterations: int,
    certainty: float | None,
    effectiveness: float | None,
    intervention_enabled: bool,
    forbid_agent: bool,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    paths = {
        "manifest": run_dir / "manifest.json",
        "metrics": run_dir / "metrics.csv",
        "pop": run_dir / "pop.arrow",
        "info": run_dir / "info.arrow",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing files in {run_dir}: {missing}")
    if forbid_agent and (run_dir / "agent.arrow").exists():
        raise ValueError(f"Stage 10 run unexpectedly retained agent.arrow: {run_dir}")
    manifest = _read_json(paths["manifest"])
    errors = _manifest_matches(
        manifest,
        stage=stage,
        network=network,
        condition_id=condition_id,
        simulator_seed=simulator_seed,
        iterations=iterations,
        certainty=certainty,
        effectiveness=effectiveness,
    )
    if errors:
        raise ValueError("; ".join(errors))
    num_agents = int(NETWORKS[network].num_agents)
    pop = pd.read_feather(paths["pop"])
    info = pd.read_feather(paths["info"])
    metrics = compute_selfish_metrics(
        pop,
        num_agents=num_agents,
        expected_iterations=iterations,
    )
    saved_metrics = pd.read_csv(paths["metrics"])
    if not _metrics_match(saved_metrics, metrics.per_iteration):
        raise ValueError(f"saved metrics do not reproduce from pop.arrow: {run_dir}")
    objective = manifest.get("objective", {}).get("value")
    if not math.isclose(
        float(objective), metrics.objective_value, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"manifest objective does not reproduce: {run_dir}")
    info_iterations, info_events = summarize_info_by_iteration(
        info,
        num_agents=num_agents,
        expected_iterations=iterations,
        intervention_enabled=intervention_enabled,
    )
    identifiers = {
        "run_key": f"{network}:{condition_id}:simseed{simulator_seed}",
        "source": source,
        "network": network,
        "condition_id": condition_id,
        "condition_group": condition_group,
        "condition_role": condition_role,
        "intervention_enabled": intervention_enabled,
        "certainty": certainty,
        "effectiveness": effectiveness,
        "simulator_seed": int(simulator_seed),
        "num_agents": num_agents,
        "iterations": int(iterations),
    }
    iteration_frame = metrics.per_iteration.merge(
        info_iterations,
        on="num_iter",
        validate="one_to_one",
    )
    for position, (field, value) in enumerate(identifiers.items()):
        iteration_frame.insert(position, field, value)

    info_event_frame = info_events.copy()
    for position, (field, value) in enumerate(identifiers.items()):
        info_event_frame.insert(position, field, value)
    pop_event_frame = pop.loc[:, ["num_iter", "t", "num_selfish"]].copy()
    for position, (field, value) in enumerate(identifiers.items()):
        pop_event_frame.insert(position, field, value)

    run_row = {
        **identifiers,
        "objective": metrics.objective_value,
        "manifest_path": paths["manifest"].as_posix(),
        "pop_path": paths["pop"].as_posix(),
        "info_path": paths["info"].as_posix(),
    }
    return iteration_frame, run_row, info_event_frame, pop_event_frame


def load_selected_fixed_raw(
    protocol: dict[str, Any],
    existing: ExistingInformationData,
) -> RawConditionData:
    """Load the selected Stage 4 comparison raw data."""

    stage4_spec = protocol["existing_data"]["stage4_fixed_confirmation"]
    original_root = _resolve_repo_path(stage4_spec["original_experiment_root"])
    override_root = _resolve_repo_path(stage4_spec["prior_high_override_root"])
    selected_conditions = set(stage4_spec["selected_conditions"])
    inventory = existing.stage4_run_inventory[
        existing.stage4_run_inventory["condition_id"].isin(selected_conditions)
    ].copy()
    if len(inventory) != int(stage4_spec["expected_selected_runs"]):
        raise ValueError("Stage 4 selected raw run count is invalid")

    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    info_event_frames: list[pd.DataFrame] = []
    pop_event_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        run_root = override_root if row.condition_id == "prior_high" else original_root
        run_dir = (
            run_root
            / str(row.network)
            / str(row.condition_id)
            / f"simseed_{int(row.simulator_seed)}"
        )
        try:
            loaded = _load_raw_run(
                run_dir=run_dir,
                stage="stage4_fixed_confirmation",
                source="stage4_fixed_confirmation",
                network=str(row.network),
                condition_id=str(row.condition_id),
                condition_group=str(row.condition_id),
                condition_role=str(row.condition_role),
                simulator_seed=int(row.simulator_seed),
                iterations=int(row.iterations),
                certainty=(None if pd.isna(row.certainty) else float(row.certainty)),
                effectiveness=(
                    None if pd.isna(row.effectiveness) else float(row.effectiveness)
                ),
                intervention_enabled=bool(row.intervention_enabled),
                forbid_agent=False,
            )
            iteration, run, info_events, pop_events = loaded
            iteration_frames.append(iteration)
            run_rows.append(run)
            info_event_frames.append(info_events)
            pop_event_frames.append(pop_events)
            audit_rows.append(
                {
                    "source": "stage4_fixed_confirmation",
                    "run_key": run["run_key"],
                    "valid": True,
                    "error": None,
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "source": "stage4_fixed_confirmation",
                    "run_key": str(row.run_key),
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    audit = pd.DataFrame(audit_rows)
    failures = audit[~audit["valid"]]
    if not failures.empty:
        raise ValueError(
            "Stage 4 selected raw audit failed:\n"
            + "\n".join(failures["error"].astype(str).head(20))
        )
    return RawConditionData(
        iterations=pd.concat(iteration_frames, ignore_index=True),
        runs=pd.DataFrame(run_rows),
        info_events=pd.concat(info_event_frames, ignore_index=True),
        pop_events=pd.concat(pop_event_frames, ignore_index=True),
        audit=audit,
    )


def load_stage10_raw(
    experiment_root: str | Path,
    protocol: dict[str, Any],
    *,
    expected_protocol_sha256: str | None = None,
) -> RawConditionData:
    root = Path(experiment_root).resolve()
    specs = build_specs(protocol)
    plan_path = root / EXECUTION_PLAN_NAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"Stage 10 execution plan is missing: {plan_path}")
    plan = _read_json(plan_path)
    counts = plan.get("counts", {})
    if plan.get("experiment_id") != root.name:
        raise ValueError("Stage 10 execution-plan experiment ID is inconsistent")
    if expected_protocol_sha256 is not None and plan.get("protocol", {}).get(
        "sha256"
    ) != expected_protocol_sha256:
        raise ValueError("Stage 10 execution-plan protocol hash is inconsistent")
    if plan.get("stage") != STAGE or counts != {
        "total": 15,
        "completed": 15,
        "pending": 0,
        "other": 0,
    }:
        raise ValueError("Stage 10 execution plan is incomplete")

    iteration_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    info_event_frames: list[pd.DataFrame] = []
    pop_event_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = root / spec.relative_run_dir
        try:
            loaded = _load_raw_run(
                run_dir=run_dir,
                stage=STAGE,
                source="stage10_primary_candidate",
                network=spec.network,
                condition_id=spec.condition_id,
                condition_group="final_candidate",
                condition_role=spec.condition_role,
                simulator_seed=spec.simulator_seed,
                iterations=spec.iterations,
                certainty=spec.certainty,
                effectiveness=spec.effectiveness,
                intervention_enabled=True,
                forbid_agent=True,
            )
            iteration, run, info_events, pop_events = loaded
            iteration_frames.append(iteration)
            run_rows.append(
                {
                    **run,
                    "reporting_role": spec.reporting_role,
                    "selection_mode": spec.selection_mode,
                    "qualified": spec.qualified,
                }
            )
            info_event_frames.append(info_events)
            pop_event_frames.append(pop_events)
            audit_rows.append(
                {
                    "source": "stage10_primary_candidate",
                    "run_key": spec.key,
                    "valid": True,
                    "error": None,
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "source": "stage10_primary_candidate",
                    "run_key": spec.key,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    audit = pd.DataFrame(audit_rows)
    failures = audit[~audit["valid"]]
    if not failures.empty:
        raise ValueError(
            "Stage 10 raw audit failed:\n"
            + "\n".join(failures["error"].astype(str).head(20))
        )
    return RawConditionData(
        iterations=pd.concat(iteration_frames, ignore_index=True),
        runs=pd.DataFrame(run_rows),
        info_events=pd.concat(info_event_frames, ignore_index=True),
        pop_events=pd.concat(pop_event_frames, ignore_index=True),
        audit=audit,
    )


def _spearman_pair(frame: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    data = frame.loc[:, [x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return {"n": int(len(data)), "rho": math.nan, "p_value": math.nan}
    result = stats.spearmanr(data[x], data[y], nan_policy="omit")
    return {
        "n": int(len(data)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _partial_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    controls: Sequence[str],
) -> dict[str, Any]:
    columns = [x, y, *controls]
    data = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) <= len(controls) + 2:
        return {"n": int(len(data)), "rho": math.nan, "p_value": math.nan}
    ranked = data.rank(method="average")
    if ranked[x].nunique() < 2 or ranked[y].nunique() < 2:
        return {"n": int(len(data)), "rho": math.nan, "p_value": math.nan}
    design = np.column_stack(
        [np.ones(len(ranked)), ranked.loc[:, controls].to_numpy(float)]
    )

    def residual(column: str) -> np.ndarray:
        values = ranked[column].to_numpy(float)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coefficients

    x_residual = residual(x)
    y_residual = residual(y)
    if np.isclose(np.std(x_residual), 0) or np.isclose(np.std(y_residual), 0):
        return {"n": int(len(data)), "rho": math.nan, "p_value": math.nan}
    result = stats.pearsonr(x_residual, y_residual)
    return {
        "n": int(len(data)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def add_bh_q_values(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values within named analysis families."""

    result = frame.copy()
    result["q_value_bh"] = np.nan
    if result.empty:
        return result
    for _, group in result.groupby(list(group_columns), dropna=False, sort=False):
        valid = group["p_value"].notna()
        if not valid.any():
            continue
        indices = group.index[valid]
        p_values = result.loc[indices, "p_value"].to_numpy(float)
        order = np.argsort(p_values)
        ranked = p_values[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        restored = np.empty_like(adjusted)
        restored[order] = adjusted
        result.loc[indices, "q_value_bh"] = restored
    return result


def build_correlation_tables(
    existing: ExistingInformationData,
) -> dict[str, pd.DataFrame]:
    """Build exploratory Stage 2 and fixed-design Stage 4 associations."""

    design_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []

    stage2_subsets = {
        "random_search_primary": existing.stage2_trials[
            existing.stage2_trials["method"] == "random"
        ],
        "all_methods_sensitivity": existing.stage2_trials,
    }
    for subset_name, subset in stage2_subsets.items():
        for network, group in subset.groupby("network", sort=True):
            for predictor in ("applied_certainty", "applied_effectiveness"):
                for indicator in INFORMATION_INDICATORS:
                    design_rows.append(
                        {
                            "source": "stage2_saved_trials",
                            "subset": subset_name,
                            "network": network,
                            "predictor": predictor,
                            "indicator": indicator,
                            **_spearman_pair(group, predictor, indicator),
                        }
                    )
            for indicator in INFORMATION_INDICATORS:
                for outcome in ("j_cum", "j_peak"):
                    outcome_rows.append(
                        {
                            "source": "stage2_saved_trials",
                            "subset": subset_name,
                            "network": network,
                            "indicator": indicator,
                            "outcome": outcome,
                            **_spearman_pair(group, indicator, outcome),
                        }
                    )
                    partial_rows.append(
                        {
                            "source": "stage2_saved_trials",
                            "subset": subset_name,
                            "network": network,
                            "indicator": indicator,
                            "outcome": outcome,
                            "controls": "applied_certainty+applied_effectiveness",
                            **_partial_spearman(
                                group,
                                indicator,
                                outcome,
                                ["applied_certainty", "applied_effectiveness"],
                            ),
                        }
                    )

    stage4_outcomes = (
        existing.stage4_iterations.groupby("run_key", sort=True)[
            [OBJECTIVE_NAME, "peak_new_selfish_ratio"]
        ]
        .mean()
        .rename(
            columns={
                OBJECTIVE_NAME: "j_cum",
                "peak_new_selfish_ratio": "j_peak",
            }
        )
        .reset_index()
    )
    stage4 = existing.stage4_info_runs.merge(
        stage4_outcomes,
        on="run_key",
        validate="one_to_one",
    )
    generated_runs = stage4[
        stage4["intervention_enabled"].astype(bool)
        & (stage4["condition_id"] != "prior_high")
    ]
    stage4_group_keys = [
        "network",
        "condition_id",
        "certainty",
        "effectiveness",
    ]
    generated = (
        generated_runs.groupby(stage4_group_keys, dropna=False, sort=True)[
            [*INFORMATION_INDICATORS, "j_cum", "j_peak"]
        ]
        .mean()
        .reset_index()
    )
    for network, group in generated.groupby("network", sort=True):
        for predictor in ("certainty", "effectiveness"):
            for indicator in INFORMATION_INDICATORS:
                design_rows.append(
                    {
                        "source": "stage4_fixed_generated_conditions",
                        "subset": "generated_fixed_conditions",
                        "network": network,
                        "predictor": predictor,
                        "indicator": indicator,
                        **_spearman_pair(group, predictor, indicator),
                    }
                )
        for indicator in INFORMATION_INDICATORS:
            for outcome in ("j_cum", "j_peak"):
                outcome_rows.append(
                    {
                        "source": "stage4_fixed_generated_conditions",
                        "subset": "generated_fixed_conditions",
                        "network": network,
                        "indicator": indicator,
                        "outcome": outcome,
                        **_spearman_pair(group, indicator, outcome),
                    }
                )
                partial_rows.append(
                    {
                        "source": "stage4_fixed_generated_conditions",
                        "subset": "generated_fixed_conditions",
                        "network": network,
                        "indicator": indicator,
                        "outcome": outcome,
                        "controls": "certainty+effectiveness",
                        **_partial_spearman(
                            group,
                            indicator,
                            outcome,
                            ["certainty", "effectiveness"],
                        ),
                    }
                )

    design = add_bh_q_values(
        pd.DataFrame(design_rows),
        group_columns=["source", "subset", "network", "predictor"],
    )
    outcomes = add_bh_q_values(
        pd.DataFrame(outcome_rows),
        group_columns=["source", "subset", "network", "outcome"],
    )
    partial = add_bh_q_values(
        pd.DataFrame(partial_rows),
        group_columns=["source", "subset", "network", "outcome"],
    )
    return {
        "design_information_correlations": design,
        "outcome_information_correlations": outcomes,
        "partial_correlations": partial,
    }


def build_ratio_diagnostics(
    frame: pd.DataFrame,
    *,
    source: str,
    grouping_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(
        list(grouping_columns), dropna=False, sort=True
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identifiers = dict(zip(grouping_columns, group_key, strict=True))
        for metric in INFO_COUNT_COLUMNS:
            numerator_column = f"corrective_{metric}_count"
            denominator_column = f"misinformation_{metric}_count"
            if numerator_column not in group:
                numerator_column = f"corrective_{metric}_sum"
                denominator_column = f"misinformation_{metric}_sum"
            numerator = pd.to_numeric(group[numerator_column], errors="coerce")
            denominator = pd.to_numeric(group[denominator_column], errors="coerce")
            zero = denominator == 0
            total_denominator = float(denominator.sum())
            rows.append(
                {
                    "source": source,
                    **identifiers,
                    "metric": metric,
                    "n_observations": int(len(group)),
                    "zero_denominator_count": int(zero.sum()),
                    "numerator_sum": float(numerator.sum()),
                    "denominator_sum": total_denominator,
                    "ratio_of_sums": (
                        float(numerator.sum()) / total_denominator
                        if total_denominator > 0
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def combine_fixed_raw(
    stage4: RawConditionData,
    stage10: RawConditionData,
) -> RawConditionData:
    iterations = pd.concat([stage4.iterations, stage10.iterations], ignore_index=True)
    runs = pd.concat([stage4.runs, stage10.runs], ignore_index=True)
    info_events = pd.concat(
        [stage4.info_events, stage10.info_events], ignore_index=True
    )
    pop_events = pd.concat([stage4.pop_events, stage10.pop_events], ignore_index=True)
    audit = pd.concat([stage4.audit, stage10.audit], ignore_index=True)
    expected_groups = {"none", "legacy_balance", "prior_high", "final_candidate"}
    if set(iterations["condition_group"].unique()) != expected_groups:
        raise ValueError("combined fixed data does not contain four condition groups")
    iteration_key = ["network", "condition_group", "simulator_seed", "num_iter"]
    if iterations.duplicated(iteration_key).any():
        raise ValueError("combined fixed iteration keys are duplicated")
    if len(runs) != 60 or len(iterations) != 6000:
        raise ValueError("combined fixed data must contain 60 runs and 6000 rows")
    return RawConditionData(
        iterations=iterations.sort_values(iteration_key).reset_index(drop=True),
        runs=runs.sort_values(["network", "condition_group", "simulator_seed"])
        .reset_index(drop=True),
        info_events=info_events.sort_values(
            ["network", "condition_group", "simulator_seed", "num_iter", "t"]
        ).reset_index(drop=True),
        pop_events=pop_events.sort_values(
            ["network", "condition_group", "simulator_seed", "num_iter", "t"]
        ).reset_index(drop=True),
        audit=audit,
    )


def build_fixed_run_summary(iterations: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "run_key",
        "source",
        "network",
        "condition_id",
        "condition_group",
        "condition_role",
        "intervention_enabled",
        "certainty",
        "effectiveness",
        "simulator_seed",
        "num_agents",
        "iterations",
    ]
    mean_columns = [*OUTCOME_COLUMNS, *INFORMATION_INDICATORS]
    non_ratio = [
        column
        for column in mean_columns
        if not column.startswith("corrective_to_misinformation_")
    ]
    result = (
        iterations.groupby(keys, dropna=False, sort=True)[non_ratio]
        .mean()
        .reset_index()
    )
    count_columns = [
        f"{name}_{metric}_count"
        for name in ("misinformation", "corrective")
        for metric in INFO_COUNT_COLUMNS
    ]
    counts = (
        iterations.groupby(keys, dropna=False, sort=True)[count_columns]
        .sum()
        .reset_index()
    )
    result = result.merge(counts, on=keys, validate="one_to_one")
    for metric in INFO_COUNT_COLUMNS:
        numerator = result[f"corrective_{metric}_count"].to_numpy(float)
        denominator = result[f"misinformation_{metric}_count"].to_numpy(float)
        result[f"corrective_to_misinformation_{metric}_ratio"] = np.divide(
            numerator,
            denominator,
            out=np.full(len(result), np.nan),
            where=denominator > 0,
        )
        result[f"misinformation_{metric}_denominator_zero"] = denominator == 0
    return result


def build_fixed_condition_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "network",
        "condition_id",
        "condition_group",
        "condition_role",
        "intervention_enabled",
        "certainty",
        "effectiveness",
        "num_agents",
    ]
    metric_columns = [*OUTCOME_COLUMNS, *INFORMATION_INDICATORS]
    rows: list[dict[str, Any]] = []
    for key, group in run_summary.groupby(keys, dropna=False, sort=True):
        identifiers = dict(zip(keys, key, strict=True))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    **identifiers,
                    "metric": metric,
                    "n_seed_blocks": int(len(values)),
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "between_seed_sd": (
                        float(values.std(ddof=1)) if len(values) > 1 else math.nan
                    ),
                    "min": float(values.min()) if len(values) else math.nan,
                    "max": float(values.max()) if len(values) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _aligned_condition_blocks(
    iterations: pd.DataFrame,
    *,
    network: str,
    reference: str,
    candidate: str,
    columns: Sequence[str],
) -> list[pd.DataFrame]:
    subset = iterations[
        (iterations["network"] == network)
        & iterations["condition_group"].isin([reference, candidate])
    ]
    key = ["simulator_seed", "num_iter"]
    if subset.duplicated([*key, "condition_group"]).any():
        raise ValueError("paired fixed observations are duplicated")
    blocks: list[pd.DataFrame] = []
    for seed, group in subset.groupby("simulator_seed", sort=True):
        wide_parts: list[pd.DataFrame] = []
        for condition in (reference, candidate):
            part = group[group["condition_group"] == condition][[*key, *columns]].copy()
            part = part.rename(
                columns={column: f"{condition}__{column}" for column in columns}
            )
            wide_parts.append(part)
        merged = wide_parts[0].merge(wide_parts[1], on=key, validate="one_to_one")
        if len(merged) != 100:
            raise ValueError(
                f"paired block {network}/{seed} contains {len(merged)} rows"
            )
        blocks.append(merged.sort_values("num_iter").reset_index(drop=True))
    if len(blocks) != 5:
        raise ValueError(f"paired comparison {network} requires five seed blocks")
    return blocks


def _difference_statistic(
    frame: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
    metric: str,
) -> tuple[float, float, float]:
    if metric.startswith("corrective_to_misinformation_"):
        short = metric.removeprefix("corrective_to_misinformation_").removesuffix(
            "_ratio"
        )
        reference_numerator = frame[f"{reference}__corrective_{short}_count"].sum()
        reference_denominator = frame[
            f"{reference}__misinformation_{short}_count"
        ].sum()
        candidate_numerator = frame[f"{candidate}__corrective_{short}_count"].sum()
        candidate_denominator = frame[
            f"{candidate}__misinformation_{short}_count"
        ].sum()
        reference_value = (
            float(reference_numerator / reference_denominator)
            if reference_denominator > 0
            else math.nan
        )
        candidate_value = (
            float(candidate_numerator / candidate_denominator)
            if candidate_denominator > 0
            else math.nan
        )
    else:
        reference_value = float(frame[f"{reference}__{metric}"].mean())
        candidate_value = float(frame[f"{candidate}__{metric}"].mean())
    return reference_value, candidate_value, candidate_value - reference_value


def hierarchical_paired_difference(
    iterations: pd.DataFrame,
    *,
    network: str,
    reference: str,
    candidate: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    ratio = metric.startswith("corrective_to_misinformation_")
    if ratio:
        short = metric.removeprefix("corrective_to_misinformation_").removesuffix(
            "_ratio"
        )
        columns = [
            f"corrective_{short}_count",
            f"misinformation_{short}_count",
        ]
    else:
        columns = [metric]
    blocks = _aligned_condition_blocks(
        iterations,
        network=network,
        reference=reference,
        candidate=candidate,
        columns=columns,
    )
    observed = pd.concat(blocks, ignore_index=True)
    reference_value, candidate_value, difference = _difference_statistic(
        observed,
        reference=reference,
        candidate=candidate,
        metric=metric,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=float)
    n_blocks = len(blocks)
    n_rows = len(blocks[0])
    if any(len(block) != n_rows for block in blocks):
        raise ValueError("paired seed blocks have unequal iteration counts")
    chunk_size = 1000
    if ratio:
        short = metric.removeprefix("corrective_to_misinformation_").removesuffix(
            "_ratio"
        )
        arrays = {
            "reference_numerator": np.stack(
                [
                    block[f"{reference}__corrective_{short}_count"].to_numpy(float)
                    for block in blocks
                ]
            ),
            "reference_denominator": np.stack(
                [
                    block[
                        f"{reference}__misinformation_{short}_count"
                    ].to_numpy(float)
                    for block in blocks
                ]
            ),
            "candidate_numerator": np.stack(
                [
                    block[f"{candidate}__corrective_{short}_count"].to_numpy(float)
                    for block in blocks
                ]
            ),
            "candidate_denominator": np.stack(
                [
                    block[
                        f"{candidate}__misinformation_{short}_count"
                    ].to_numpy(float)
                    for block in blocks
                ]
            ),
        }
    else:
        difference_array = np.stack(
            [
                block[f"{candidate}__{metric}"].to_numpy(float)
                - block[f"{reference}__{metric}"].to_numpy(float)
                for block in blocks
            ]
        )
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        size = stop - start
        block_indices = rng.integers(0, n_blocks, size=(size, n_blocks))
        row_indices = rng.integers(
            0,
            n_rows,
            size=(size, n_blocks, n_rows),
        )
        selected_blocks = block_indices[:, :, None]
        if ratio:
            sampled = {
                name: values[selected_blocks, row_indices].sum(axis=(1, 2))
                for name, values in arrays.items()
            }
            reference_ratio = np.divide(
                sampled["reference_numerator"],
                sampled["reference_denominator"],
                out=np.full(size, np.nan),
                where=sampled["reference_denominator"] > 0,
            )
            candidate_ratio = np.divide(
                sampled["candidate_numerator"],
                sampled["candidate_denominator"],
                out=np.full(size, np.nan),
                where=sampled["candidate_denominator"] > 0,
            )
            bootstrap[start:stop] = candidate_ratio - reference_ratio
        else:
            bootstrap[start:stop] = difference_array[
                selected_blocks, row_indices
            ].mean(axis=(1, 2))
    valid = bootstrap[np.isfinite(bootstrap)]
    if len(valid) != repetitions:
        raise ValueError("paired bootstrap produced a non-finite ratio")
    ci_low, ci_high = np.quantile(valid, [0.025, 0.975])
    return {
        "reference_mean": reference_value,
        "candidate_mean": candidate_value,
        "candidate_minus_reference": difference,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_repetitions": int(repetitions),
    }


def _difference_interpretation(
    metric: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
) -> str:
    if metric in OUTCOME_COLUMNS:
        if ci_high < 0:
            return "outcome_reduction"
        if ci_low > 0:
            return "outcome_increase"
        if estimate < 0:
            return "outcome_reduction_tendency_with_uncertainty"
        if estimate > 0:
            return "outcome_increase_tendency_with_uncertainty"
        return "no_clear_outcome_difference"
    if ci_low > 0:
        return "higher_response_indicator"
    if ci_high < 0:
        return "lower_response_indicator"
    return "no_clear_response_indicator_difference"


def build_primary_paired_effects(
    iterations: pd.DataFrame,
    *,
    primary_indicators: Sequence[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        ("legacy_balance", "none"),
        ("prior_high", "none"),
        ("final_candidate", "none"),
        ("final_candidate", "legacy_balance"),
        ("final_candidate", "prior_high"),
    ]
    metrics = [*OUTCOME_COLUMNS, *primary_indicators]
    rows: list[dict[str, Any]] = []
    offset = 0
    for network in sorted(iterations["network"].unique()):
        for candidate, reference in comparisons:
            comparison_id = f"{candidate}_vs_{reference}"
            for metric in metrics:
                estimate = hierarchical_paired_difference(
                    iterations,
                    network=str(network),
                    reference=reference,
                    candidate=candidate,
                    metric=metric,
                    repetitions=repetitions,
                    seed=seed + offset,
                )
                offset += 1
                rows.append(
                    {
                        "network": network,
                        "comparison_id": comparison_id,
                        "reference": reference,
                        "candidate": candidate,
                        "metric": metric,
                        "metric_role": (
                            "outcome" if metric in OUTCOME_COLUMNS else "response_indicator"
                        ),
                        **estimate,
                        "interpretation": _difference_interpretation(
                            metric,
                            float(estimate["candidate_minus_reference"]),
                            float(estimate["ci_low"]),
                            float(estimate["ci_high"]),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_exploratory_fixed_differences(
    run_summary: pd.DataFrame,
    *,
    primary_indicators: Sequence[str],
) -> pd.DataFrame:
    exploratory = [
        column
        for column in INFORMATION_INDICATORS
        if column not in set(primary_indicators)
    ]
    comparisons = [
        ("legacy_balance", "none"),
        ("prior_high", "none"),
        ("final_candidate", "none"),
        ("final_candidate", "legacy_balance"),
        ("final_candidate", "prior_high"),
    ]
    rows: list[dict[str, Any]] = []
    for network, network_data in run_summary.groupby("network", sort=True):
        means = network_data.groupby("condition_group", sort=True)[exploratory].mean()
        for candidate, reference in comparisons:
            for metric in exploratory:
                reference_value = float(means.loc[reference, metric])
                candidate_value = float(means.loc[candidate, metric])
                rows.append(
                    {
                        "network": network,
                        "comparison_id": f"{candidate}_vs_{reference}",
                        "reference": reference,
                        "candidate": candidate,
                        "metric": metric,
                        "reference_mean": reference_value,
                        "candidate_mean": candidate_value,
                        "candidate_minus_reference": candidate_value
                        - reference_value,
                        "inference_role": "exploratory_point_estimate",
                    }
                )
    return pd.DataFrame(rows)


def _seed_curve_interval(
    seed_curves: pd.DataFrame,
    *,
    value_column: str,
    grouping_columns: Sequence[str],
) -> pd.DataFrame:
    grouped = seed_curves.groupby(list(grouping_columns), sort=True)[value_column]
    result = grouped.agg(n_seed_blocks="size", mean="mean", between_seed_sd="std")
    result = result.reset_index()
    sem = result["between_seed_sd"] / np.sqrt(result["n_seed_blocks"])
    critical = stats.t.ppf(0.975, result["n_seed_blocks"] - 1)
    half_width = critical * sem
    result["ci_low"] = result["mean"] - half_width
    result["ci_high"] = result["mean"] + half_width
    return result


def build_behavior_time_series(pop_events: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive cumulative selfish-action curves by seed block."""

    seed_frames: list[pd.DataFrame] = []
    for network, network_data in pop_events.groupby("network", sort=True):
        max_t = int(network_data["t"].max())
        for identifiers, group in network_data.groupby(
            [
                "condition_group",
                "condition_id",
                "simulator_seed",
                "num_agents",
            ],
            sort=True,
        ):
            condition_group, condition_id, simulator_seed, num_agents = identifiers
            iteration_curves: list[np.ndarray] = []
            for _, iteration in group.groupby("num_iter", sort=True):
                counts = (
                    iteration.groupby("t")["num_selfish"]
                    .sum()
                    .reindex(range(max_t + 1), fill_value=0)
                    .to_numpy(float)
                )
                iteration_curves.append(np.cumsum(counts) / int(num_agents))
            mean_curve = np.mean(np.vstack(iteration_curves), axis=0)
            seed_frames.append(
                pd.DataFrame(
                    {
                        "network": network,
                        "condition_group": condition_group,
                        "condition_id": condition_id,
                        "simulator_seed": int(simulator_seed),
                        "t": np.arange(max_t + 1, dtype=int),
                        "cumulative_selfish_fraction": mean_curve,
                    }
                )
            )
    seed_curves = pd.concat(seed_frames, ignore_index=True)
    summary = _seed_curve_interval(
        seed_curves,
        value_column="cumulative_selfish_fraction",
        grouping_columns=["network", "condition_group", "condition_id", "t"],
    )
    return summary


def build_information_time_series(info_events: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative first-access curves without treating them as unique users."""

    seed_frames: list[pd.DataFrame] = []
    for network, network_data in info_events.groupby("network", sort=True):
        max_t = int(network_data["t"].max())
        for identifiers, group in network_data.groupby(
            [
                "condition_group",
                "condition_id",
                "simulator_seed",
                "num_agents",
            ],
            sort=True,
        ):
            condition_group, condition_id, simulator_seed, num_agents = identifiers
            for info_label, info_name in INFO_NAMES.items():
                label_data = group[group["info_label"] == info_label]
                iteration_curves: list[np.ndarray] = []
                for num_iter in range(int(group["iterations"].iloc[0])):
                    iteration = label_data[label_data["num_iter"] == num_iter]
                    counts = (
                        iteration.groupby("t")["num_fst_viewed"]
                        .sum()
                        .reindex(range(max_t + 1), fill_value=0)
                        .to_numpy(float)
                    )
                    iteration_curves.append(np.cumsum(counts) / int(num_agents))
                mean_curve = np.mean(np.vstack(iteration_curves), axis=0)
                seed_frames.append(
                    pd.DataFrame(
                        {
                            "network": network,
                            "condition_group": condition_group,
                            "condition_id": condition_id,
                            "simulator_seed": int(simulator_seed),
                            "info_label": int(info_label),
                            "info_name": info_name,
                            "t": np.arange(max_t + 1, dtype=int),
                            "cumulative_first_access_per_agent": mean_curve,
                        }
                    )
                )
    seed_curves = pd.concat(seed_frames, ignore_index=True)
    return _seed_curve_interval(
        seed_curves,
        value_column="cumulative_first_access_per_agent",
        grouping_columns=[
            "network",
            "condition_group",
            "condition_id",
            "info_label",
            "info_name",
            "t",
        ],
    )


def build_direction_consistency(
    correlations: dict[str, pd.DataFrame],
    primary_indicators: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    design = correlations["design_information_correlations"]
    outcome = correlations["outcome_information_correlations"]
    partial = correlations["partial_correlations"]
    selections = [
        (
            "stage2_random_design",
            design[
                (design["source"] == "stage2_saved_trials")
                & (design["subset"] == "random_search_primary")
                & design["indicator"].isin(primary_indicators)
            ],
            ["predictor", "indicator"],
        ),
        (
            "stage2_random_outcome",
            outcome[
                (outcome["source"] == "stage2_saved_trials")
                & (outcome["subset"] == "random_search_primary")
                & outcome["indicator"].isin(primary_indicators)
            ],
            ["outcome", "indicator"],
        ),
        (
            "stage2_random_partial",
            partial[
                (partial["source"] == "stage2_saved_trials")
                & (partial["subset"] == "random_search_primary")
                & partial["indicator"].isin(primary_indicators)
            ],
            ["outcome", "indicator"],
        ),
    ]
    for analysis, frame, keys in selections:
        for key, group in frame.groupby(keys, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            identifiers = dict(zip(keys, key, strict=True))
            signs = np.sign(pd.to_numeric(group["rho"], errors="coerce").dropna())
            rows.append(
                {
                    "analysis": analysis,
                    **identifiers,
                    "network_count": int(len(signs)),
                    "positive_networks": int((signs > 0).sum()),
                    "negative_networks": int((signs < 0).sum()),
                    "zero_networks": int((signs == 0).sum()),
                    "same_nonzero_direction_all_networks": bool(
                        len(signs) == 3 and abs(int(signs.sum())) == 3
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_decision(
    paired_effects: pd.DataFrame,
    direction_consistency: pd.DataFrame,
    stage10_runs: pd.DataFrame,
) -> dict[str, Any]:
    networks: list[dict[str, Any]] = []
    primary = paired_effects[
        paired_effects["comparison_id"] == "final_candidate_vs_none"
    ]
    for network, group in primary.groupby("network", sort=True):
        outcome_rows = group[group["metric"].isin(OUTCOME_COLUMNS)]
        info_rows = group[~group["metric"].isin(OUTCOME_COLUMNS)]
        run = stage10_runs[stage10_runs["network"] == network].iloc[0]
        networks.append(
            {
                "network": network,
                "candidate_id": str(run["condition_id"]),
                "candidate_reporting_role": str(run["reporting_role"]),
                "candidate_qualified_before_stage10": bool(run["qualified"]),
                "outcomes": [
                    {
                        "metric": str(row.metric),
                        "candidate_minus_none": float(row.candidate_minus_reference),
                        "ci_low": float(row.ci_low),
                        "ci_high": float(row.ci_high),
                        "interpretation": str(row.interpretation),
                    }
                    for row in outcome_rows.itertuples(index=False)
                ],
                "primary_information_responses": [
                    {
                        "metric": str(row.metric),
                        "candidate_minus_none": float(row.candidate_minus_reference),
                        "ci_low": float(row.ci_low),
                        "ci_high": float(row.ci_high),
                        "interpretation": str(row.interpretation),
                    }
                    for row in info_rows.itertuples(index=False)
                ],
            }
        )
    return {
        "status": "information_diffusion_reanalysis_complete",
        "candidate_selection_changed": False,
        "information_metrics_used_for_candidate_selection": False,
        "causal_claim_supported": False,
        "ratio_promoted_to_design_rule": False,
        "stage2_interpretation": "exploratory_association_with_adaptive_sampling_limit",
        "fixed_comparison_interpretation": "paired_post_intervention_response_description",
        "networks": networks,
        "cross_network_direction_summary_rows": int(len(direction_consistency)),
        "interpretation_limits": [
            "Information indicators were observed after intervention and were not independently manipulated.",
            "The corrective-to-misinformation ratio remains a response indicator.",
            "Lower diffusion does not imply that an information type is unnecessary.",
            "Facebook primary-candidate evidence remains exploratory.",
        ],
    }
