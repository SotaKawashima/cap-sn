"""Audit and compare the network-similarity optimization experiment.

The primary cross-network comparison uses the 300 random-search evaluations.
Those evaluations share optimizer seeds and therefore evaluate the same applied
certainty/effectiveness coordinates on every network in a family.  Adaptive
BO-GP and CMA-ES evaluations remain available as descriptive search output but
are not used to estimate cross-network surface similarity.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from analysis.optimization_metrics import (
    LEGACY_METRIC_NAME,
    OBJECTIVE_DEFINITION_VERSION,
    OBJECTIVE_NAME,
    compute_selfish_metrics_from_arrow,
)
from experiment_runtime import REPO_ROOT, sha256_file


PRIMARY_METHOD = "random_search"
GOOD_FRACTION = 0.10
REQUIRED_TRIAL_COLUMNS = {
    "trial",
    "state",
    "value",
    "proposed_certainty",
    "proposed_effectiveness",
    "applied_certainty",
    "applied_effectiveness",
    OBJECTIVE_NAME,
    "peak_new_selfish_ratio",
    LEGACY_METRIC_NAME,
    "n_iterations",
    "simulation_sec",
    "metric_calculation_sec",
    "trial_total_sec",
    "raw_dir",
}


@dataclass(frozen=True)
class SimilaritySubsetData:
    """Validated added-network data and one audit row per run."""

    trials: pd.DataFrame
    runs: pd.DataFrame
    fixed: pd.DataFrame
    audit: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_matches(
    observed: Any,
    expected: Any,
    tolerance: float = 1e-12,
) -> bool:
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


def _metric_frame_errors(
    saved: pd.DataFrame,
    recalculated: pd.DataFrame,
    *,
    expected_iterations: int,
    tolerance: float,
) -> list[str]:
    if len(saved) != expected_iterations:
        return [
            f"metrics row count={len(saved)}, expected={expected_iterations}"
        ]
    if "num_iter" not in saved.columns:
        return ["saved metrics have no num_iter column"]
    observed_ids = set(
        pd.to_numeric(saved["num_iter"], errors="coerce").dropna().astype(int)
    )
    if observed_ids != set(range(expected_iterations)):
        return ["metrics iteration IDs do not match the expected range"]

    left = saved.sort_values("num_iter").reset_index(drop=True)
    right = recalculated.sort_values("num_iter").reset_index(drop=True)
    missing = [column for column in right.columns if column not in left.columns]
    if missing:
        return [f"saved metrics are missing columns: {missing}"]

    errors: list[str] = []
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


def _validate_common_manifest(
    manifest: Mapping[str, Any],
    spec: Any,
    *,
    expected_stage: str,
) -> list[str]:
    expected_run_type = (
        "single_objective_optimization"
        if spec.run_type == "optimization"
        else "fixed_condition"
    )
    expected_config = (REPO_ROOT / spec.network_config).resolve()
    observed = {
        "status": manifest.get("status"),
        "stage": manifest.get("stage"),
        "run_type": manifest.get("run_type"),
        "network": manifest.get("network", {}).get("id"),
        "num_agents": manifest.get("network", {}).get("num_agents"),
        "network_sha256": manifest.get("network", {}).get("sha256"),
        "simulator_seed": manifest.get("runtime", {}).get("simulator_seed"),
        "iterations": manifest.get("runtime", {}).get("iteration_count"),
        "objective_name": manifest.get("objective", {}).get("name"),
        "objective_version": manifest.get("objective", {}).get(
            "definition_version"
        ),
    }
    expected = {
        "status": "completed",
        "stage": expected_stage,
        "run_type": expected_run_type,
        "network": spec.network,
        "num_agents": int(spec.num_agents),
        "network_sha256": sha256_file(expected_config),
        "simulator_seed": int(spec.simulator_seed),
        "iterations": int(spec.iterations),
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_DEFINITION_VERSION,
    }
    return [
        f"{field}={observed[field]!r}, expected={value!r}"
        for field, value in expected.items()
        if observed[field] != value
    ]


def _validate_optimization_manifest(
    manifest: Mapping[str, Any],
    spec: Any,
) -> list[str]:
    optimization = manifest.get("optimization", {})
    expected_startup = int(spec.startup_trials) if spec.method == "bo_gp" else None
    observed = {
        "method": optimization.get("method"),
        "optimizer_replicate": optimization.get("optimizer_replicate"),
        "optimizer_seed": optimization.get("optimizer_seed"),
        "trials": optimization.get("n_trials_requested"),
        "startup_trials": optimization.get("startup_trials"),
        "raw_level": optimization.get("raw_level"),
        "direction": manifest.get("objective", {}).get("direction"),
        "application_precision": manifest.get("intervention", {}).get(
            "application_precision_decimal_places"
        ),
    }
    expected = {
        "method": spec.method,
        "optimizer_replicate": int(spec.optimizer_replicate),
        "optimizer_seed": int(spec.optimizer_seed),
        "trials": int(spec.trials),
        "startup_trials": expected_startup,
        "raw_level": spec.raw_level,
        "direction": "minimize",
        "application_precision": 4,
    }
    errors = [
        f"{field}={observed[field]!r}, expected={value!r}"
        for field, value in expected.items()
        if observed[field] != value
    ]
    expected_counts = {"complete": int(spec.trials), "failed": 0, "pruned": 0}
    if manifest.get("counts") != expected_counts:
        errors.append(
            f"counts={manifest.get('counts')!r}, expected={expected_counts!r}"
        )
    bounds = manifest.get("intervention", {}).get("parameter_bounds", {})
    for name in ("certainty", "effectiveness"):
        if bounds.get(name) != [0.5, 1.0]:
            errors.append(f"invalid {name} bounds: {bounds.get(name)!r}")
    return errors


def _validate_fixed_manifest(
    manifest: Mapping[str, Any],
    spec: Any,
    *,
    tolerance: float,
) -> list[str]:
    intervention = manifest.get("intervention", {})
    errors: list[str] = []
    if intervention.get("condition_id") != spec.condition:
        errors.append(
            f"condition_id={intervention.get('condition_id')!r}, "
            f"expected={spec.condition!r}"
        )
    expected_enabled = spec.condition != "none"
    if intervention.get("enabled") is not expected_enabled:
        errors.append(
            f"intervention enabled={intervention.get('enabled')!r}, "
            f"expected={expected_enabled!r}"
        )
    applied = intervention.get("applied_parameters")
    if expected_enabled:
        for name in ("certainty", "effectiveness"):
            if not isinstance(applied, Mapping) or not _float_matches(
                applied.get(name), 1.0, tolerance
            ):
                errors.append(f"simple_max applied {name} is not 1.0")
    elif applied is not None:
        errors.append("no-intervention run has applied parameters")
    return errors


def _validate_trial_table(
    trials: pd.DataFrame,
    spec: Any,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_TRIAL_COLUMNS.difference(trials.columns))
    if missing:
        raise ValueError(f"trials.csv is missing columns: {missing}")
    expected_count = int(spec.trials)
    if len(trials) != expected_count:
        raise ValueError(f"trial count={len(trials)}, expected={expected_count}")
    if set(trials["state"].astype(str)) != {"COMPLETE"}:
        raise ValueError(f"unexpected trial states: {sorted(set(trials['state']))}")
    ids = pd.to_numeric(trials["trial"], errors="raise").astype(int)
    if set(ids) != set(range(expected_count)):
        raise ValueError("trial IDs do not match the expected range")

    numeric = sorted(
        REQUIRED_TRIAL_COLUMNS
        - {"state", "raw_dir"}
    )
    converted = trials.copy()
    for column in numeric:
        converted[column] = pd.to_numeric(converted[column], errors="raise")
    if not np.isfinite(converted[numeric].to_numpy(float)).all():
        raise ValueError("trial table contains non-finite numeric values")
    for name in ("certainty", "effectiveness"):
        proposed = converted[f"proposed_{name}"]
        applied = converted[f"applied_{name}"]
        if not proposed.between(0.5, 1.0).all():
            raise ValueError(f"proposed_{name} lies outside [0.5, 1.0]")
        if not applied.between(0.5, 1.0).all():
            raise ValueError(f"applied_{name} lies outside [0.5, 1.0]")
        if not np.allclose(
            applied.to_numpy(float),
            proposed.round(4).to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"applied_{name} does not reproduce rounding")
    if not np.allclose(
        converted["value"],
        converted[OBJECTIVE_NAME],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("value and cumulative_selfish_fraction differ")
    return converted.sort_values("trial").reset_index(drop=True)


def load_similarity_subset(
    experiment_root: str | Path,
    *,
    expected_specs: Sequence[Any],
    expected_stage: str = "network_similarity_optimization",
    tolerance: float = 1e-12,
) -> SimilaritySubsetData:
    """Validate every selected run and reproduce all objectives from raw data."""

    root = Path(experiment_root).resolve()
    trial_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in expected_specs:
        run_dir = root / spec.relative_run_dir
        audit: dict[str, Any] = {
            "run_key": spec.key,
            "network": spec.network,
            "run_type": spec.run_type,
            "method_or_condition": spec.method or spec.condition,
            "files_present": False,
            "manifest_valid": False,
            "table_valid": False,
            "raw_metrics_reproduced": False,
            "valid": False,
            "error": None,
        }
        try:
            manifest_path = run_dir / "manifest.json"
            runtime_path = run_dir / "runtime.toml"
            if not manifest_path.is_file() or not runtime_path.is_file():
                raise FileNotFoundError("manifest.json or runtime.toml is missing")
            manifest = _read_json(manifest_path)
            errors = _validate_common_manifest(
                manifest,
                spec,
                expected_stage=expected_stage,
            )
            if spec.run_type == "optimization":
                errors.extend(_validate_optimization_manifest(manifest, spec))
            else:
                errors.extend(
                    _validate_fixed_manifest(
                        manifest,
                        spec,
                        tolerance=tolerance,
                    )
                )
            if errors:
                raise ValueError("; ".join(errors))
            audit["manifest_valid"] = True

            if spec.run_type == "optimization":
                required = {
                    "trials": run_dir / "trials.csv",
                    "summary": run_dir / "summary.json",
                    "study": run_dir / "study.db",
                }
                missing = [name for name, path in required.items() if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"missing optimization files: {missing}")
                audit["files_present"] = True
                trials = _validate_trial_table(
                    pd.read_csv(required["trials"]),
                    spec,
                )
                audit["table_valid"] = True

                raw_bytes = 0
                for row in trials.itertuples(index=False):
                    trial_dir = run_dir / str(row.raw_dir)
                    trial_paths = {
                        "pop": trial_dir / "pop.arrow",
                        "metrics": trial_dir / "metrics.csv",
                        "summary": trial_dir / "metrics_summary.json",
                        "strategy": trial_dir / "strategy.toml",
                        "opinion": trial_dir / "inhibition_opinion.csv",
                    }
                    missing = [
                        name for name, path in trial_paths.items() if not path.is_file()
                    ]
                    if missing:
                        raise FileNotFoundError(
                            f"trial {row.trial} is missing files: {missing}"
                        )
                    result = compute_selfish_metrics_from_arrow(
                        trial_paths["pop"],
                        num_agents=int(spec.num_agents),
                        expected_iterations=int(spec.iterations),
                    )
                    raw_bytes += trial_paths["pop"].stat().st_size
                    for field in (
                        OBJECTIVE_NAME,
                        "peak_new_selfish_ratio",
                        LEGACY_METRIC_NAME,
                    ):
                        if not _float_matches(
                            getattr(row, field),
                            result.summary[field],
                            tolerance,
                        ):
                            raise ValueError(
                                f"trial {row.trial} does not reproduce {field}"
                            )
                    if not _float_matches(
                        row.value,
                        result.objective_value,
                        tolerance,
                    ):
                        raise ValueError(
                            f"trial {row.trial} objective does not reproduce"
                        )
                    metric_errors = _metric_frame_errors(
                        pd.read_csv(trial_paths["metrics"]),
                        result.per_iteration,
                        expected_iterations=int(spec.iterations),
                        tolerance=tolerance,
                    )
                    if metric_errors:
                        raise ValueError(
                            f"trial {row.trial}: {'; '.join(metric_errors)}"
                        )
                    saved_summary = _read_json(trial_paths["summary"])
                    if not _float_matches(
                        saved_summary.get(OBJECTIVE_NAME),
                        result.objective_value,
                        tolerance,
                    ):
                        raise ValueError(
                            f"trial {row.trial} metrics summary does not reproduce"
                        )
                audit["raw_metrics_reproduced"] = True

                summary = _read_json(required["summary"])
                best_row = trials.loc[trials["value"].idxmin()]
                for label, record in (
                    ("manifest", manifest.get("best") or {}),
                    ("summary", summary.get("best") or {}),
                ):
                    if int(record.get("trial", -1)) != int(best_row["trial"]):
                        raise ValueError(f"{label} best trial does not reproduce")
                    if not _float_matches(
                        record.get("value"),
                        best_row["value"],
                        tolerance,
                    ):
                        raise ValueError(f"{label} best value does not reproduce")

                identifiers = {
                    "run_key": spec.key,
                    "network": spec.network,
                    "method": spec.method,
                    "optimizer_replicate": int(spec.optimizer_replicate),
                    "optimizer_seed": int(spec.optimizer_seed),
                    "simulator_seed": int(spec.simulator_seed),
                    "iterations_per_evaluation": int(spec.iterations),
                    "num_agents": int(spec.num_agents),
                }
                frame = trials.copy()
                frame["evaluation"] = frame["trial"].astype(int) + 1
                for position, (field, value) in enumerate(identifiers.items()):
                    frame.insert(position, field, value)
                trial_frames.append(frame)
                timing = manifest.get("timing_sec", {})
                run_rows.append(
                    {
                        **identifiers,
                        "evaluations": int(spec.trials),
                        "optimization_total_sec": float(
                            timing["optimization_total"]
                        ),
                        "simulation_total_sec": float(timing["simulation_total"]),
                        "metric_calculation_total_sec": float(
                            timing["metric_calculation_total"]
                        ),
                        "optimizer_overhead_sec": float(
                            timing["optimizer_overhead"]
                        ),
                        "raw_pop_bytes": int(raw_bytes),
                    }
                )
            else:
                paths = {
                    "pop": run_dir / "pop.arrow",
                    "metrics": run_dir / "metrics.csv",
                    "summary": run_dir / "metrics_summary.json",
                }
                missing = [name for name, path in paths.items() if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"missing fixed-run files: {missing}")
                audit["files_present"] = True
                result = compute_selfish_metrics_from_arrow(
                    paths["pop"],
                    num_agents=int(spec.num_agents),
                    expected_iterations=int(spec.iterations),
                )
                metric_errors = _metric_frame_errors(
                    pd.read_csv(paths["metrics"]),
                    result.per_iteration,
                    expected_iterations=int(spec.iterations),
                    tolerance=tolerance,
                )
                if metric_errors:
                    raise ValueError("; ".join(metric_errors))
                saved_summary = _read_json(paths["summary"])
                if not _float_matches(
                    saved_summary.get(OBJECTIVE_NAME),
                    result.objective_value,
                    tolerance,
                ) or not _float_matches(
                    manifest.get("objective", {}).get("value"),
                    result.objective_value,
                    tolerance,
                ):
                    raise ValueError("fixed-run objective does not reproduce")
                audit["table_valid"] = True
                audit["raw_metrics_reproduced"] = True
                fixed_rows.append(
                    {
                        "run_key": spec.key,
                        "network": spec.network,
                        "condition": spec.condition,
                        "simulator_seed": int(spec.simulator_seed),
                        "iterations": int(spec.iterations),
                        "num_agents": int(spec.num_agents),
                        "value": result.objective_value,
                        "peak_new_selfish_ratio": result.summary[
                            "peak_new_selfish_ratio"
                        ],
                        "simulation_sec": float(
                            manifest.get("timing_sec", {})["simulation"]
                        ),
                        "pop_bytes": int(paths["pop"].stat().st_size),
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
            "network-similarity subset failed formal validation:\n"
            + preview
            + suffix
        )
    return SimilaritySubsetData(
        trials=pd.concat(trial_frames, ignore_index=True),
        runs=pd.DataFrame(run_rows).sort_values("run_key").reset_index(drop=True),
        fixed=pd.DataFrame(fixed_rows)
        .sort_values(["network", "condition"])
        .reset_index(drop=True),
        audit=pd.DataFrame(audit_rows).sort_values("run_key").reset_index(drop=True),
    )


def load_stage6_trial_inventory(
    analysis_root: str | Path,
    *,
    networks: Sequence[str],
) -> pd.DataFrame:
    """Load the already-formally-audited Stage 6 trial inventory."""

    root = Path(analysis_root).resolve()
    manifest_path = root / "analysis_manifest.json"
    table_path = root / "tables" / "trial_inventory.parquet"
    if not manifest_path.is_file() or not table_path.is_file():
        raise FileNotFoundError("Stage 6 analysis manifest or trial inventory is missing")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("Stage 6 analysis is not completed")
    entry = manifest.get("outputs", {}).get("trial_inventory", {})
    if entry.get("rows") != 2700 or entry.get("sha256") != sha256_file(table_path):
        raise ValueError("Stage 6 trial inventory hash or row count is inconsistent")

    trials = pd.read_parquet(table_path)
    selected = trials.loc[trials["network"].isin(networks)].copy()
    required = {
        "run_key",
        "network",
        "method",
        "optimizer_replicate",
        "optimizer_seed",
        "simulator_seed",
        "evaluation",
        "applied_certainty",
        "applied_effectiveness",
        "state",
        "value",
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"Stage 6 trial inventory is missing columns: {missing}")
    expected_counts = {
        (network, method): 300
        for network in networks
        for method in ("bo_gp", "cma_es", "random_search")
    }
    counts = selected.groupby(["network", "method"]).size().to_dict()
    if counts != expected_counts:
        raise ValueError(f"unexpected Stage 6 selected counts: {counts}")
    if set(selected["state"].astype(str)) != {"COMPLETE"}:
        raise ValueError("Stage 6 selected trials are not all COMPLETE")
    if selected[list(required - {"state", "run_key", "network", "method"})].isna().any().any():
        raise ValueError("Stage 6 selected trials contain missing values")
    selected["source_experiment"] = "stage6_reoptimization"
    return selected.reset_index(drop=True)


def build_shared_random_points(
    trials: pd.DataFrame,
    *,
    family_networks: Mapping[str, Sequence[str]],
    good_fraction: float = GOOD_FRACTION,
) -> pd.DataFrame:
    """Validate and label common random-search coordinates within each family."""

    if not 0.0 < good_fraction < 1.0:
        raise ValueError("good_fraction must lie between zero and one")
    random = trials.loc[trials["method"].eq(PRIMARY_METHOD)].copy()
    rows: list[pd.DataFrame] = []
    for family, networks in family_networks.items():
        family_data = random.loc[random["network"].isin(networks)].copy()
        if not networks:
            raise ValueError(f"family {family!r} contains no networks")
        expected_points: set[tuple[int, int]] | None = None
        coordinate_reference: pd.DataFrame | None = None
        for network in networks:
            network_data = family_data.loc[family_data["network"].eq(network)].copy()
            if len(network_data) != 300:
                raise ValueError(
                    f"{network} has {len(network_data)} random evaluations, expected 300"
                )
            points = set(
                zip(
                    network_data["optimizer_seed"].astype(int),
                    network_data["evaluation"].astype(int),
                    strict=True,
                )
            )
            if len(points) != 300:
                raise ValueError(f"{network} random point IDs are not unique")
            coordinates = (
                network_data.assign(
                    optimizer_seed=network_data["optimizer_seed"].astype(int),
                    evaluation=network_data["evaluation"].astype(int),
                )
                .set_index(["optimizer_seed", "evaluation"])[
                    ["applied_certainty", "applied_effectiveness"]
                ]
                .sort_index()
            )
            if expected_points is None:
                expected_points = points
                coordinate_reference = coordinates
            else:
                if points != expected_points:
                    raise ValueError(f"{network} random point IDs do not align")
                if coordinate_reference is None or not np.array_equal(
                    coordinates.to_numpy(float),
                    coordinate_reference.to_numpy(float),
                ):
                    raise ValueError(
                        f"{network} random applied coordinates do not align"
                    )

            network_data["family"] = family
            network_data["point_id"] = (
                network_data["optimizer_seed"].astype(int).astype(str)
                + ":"
                + network_data["evaluation"].astype(int).astype(str)
            )
            network_data["objective_rank"] = network_data["value"].rank(
                method="average", ascending=True
            )
            network_data["objective_percentile"] = (
                network_data["objective_rank"] - 1.0
            ) / (len(network_data) - 1) * 100.0
            good_count = int(round(len(network_data) * good_fraction))
            ordered = network_data.sort_values(
                ["value", "point_id"], kind="stable"
            )
            good_ids = set(ordered.head(good_count)["point_id"])
            network_data["is_good_point"] = network_data["point_id"].isin(good_ids)
            rows.append(network_data)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["family", "network", "optimizer_seed", "evaluation"]
    ).reset_index(drop=True)


def build_pairwise_similarity(shared: pd.DataFrame) -> pd.DataFrame:
    """Summarize objective ordering and bottom-decile overlap for each pair."""

    rows: list[dict[str, Any]] = []
    for family, family_data in shared.groupby("family", sort=True):
        networks = sorted(family_data["network"].unique())
        for left_network, right_network in combinations(networks, 2):
            left = family_data.loc[
                family_data["network"].eq(left_network)
            ].set_index("point_id")
            right = family_data.loc[
                family_data["network"].eq(right_network)
            ].set_index("point_id")
            joined = left[["value"]].join(
                right[["value"]],
                lsuffix="_left",
                rsuffix="_right",
                how="inner",
                validate="one_to_one",
            )
            left_good = set(left.index[left["is_good_point"]])
            right_good = set(right.index[right["is_good_point"]])
            overlap = len(left_good & right_good)
            union = len(left_good | right_good)
            expected_overlap = len(left_good) * len(right_good) / len(joined)
            rows.append(
                {
                    "family": family,
                    "left_network": left_network,
                    "right_network": right_network,
                    "n_shared_points": int(len(joined)),
                    "spearman_objective": float(
                        joined["value_left"].corr(
                            joined["value_right"], method="spearman"
                        )
                    ),
                    "good_points_per_network": int(len(left_good)),
                    "good_overlap_count": int(overlap),
                    "good_overlap_fraction": float(overlap / len(left_good)),
                    "good_overlap_jaccard": float(overlap / union),
                    "chance_expected_overlap_count": float(expected_overlap),
                    "overlap_enrichment_vs_chance": float(
                        overlap / expected_overlap
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_good_region_summary(shared: pd.DataFrame) -> pd.DataFrame:
    """Describe the bottom-decile parameter region and monotone trends."""

    rows: list[dict[str, Any]] = []
    for (family, network), group in shared.groupby(
        ["family", "network"], sort=True
    ):
        good = group.loc[group["is_good_point"]]
        rows.append(
            {
                "family": family,
                "network": network,
                "n_points": int(len(group)),
                "n_good_points": int(len(good)),
                "certainty_objective_spearman": float(
                    group["applied_certainty"].corr(
                        group["value"], method="spearman"
                    )
                ),
                "effectiveness_objective_spearman": float(
                    group["applied_effectiveness"].corr(
                        group["value"], method="spearman"
                    )
                ),
                "good_certainty_q1": float(good["applied_certainty"].quantile(0.25)),
                "good_certainty_median": float(good["applied_certainty"].median()),
                "good_certainty_q3": float(good["applied_certainty"].quantile(0.75)),
                "good_effectiveness_q1": float(
                    good["applied_effectiveness"].quantile(0.25)
                ),
                "good_effectiveness_median": float(
                    good["applied_effectiveness"].median()
                ),
                "good_effectiveness_q3": float(
                    good["applied_effectiveness"].quantile(0.75)
                ),
                "good_objective_minimum": float(good["value"].min()),
                "good_objective_median": float(good["value"].median()),
                "good_objective_maximum": float(good["value"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_good_region_transfer(shared: pd.DataFrame) -> pd.DataFrame:
    """Measure where one network's bottom-decile points rank on another."""

    rows: list[dict[str, Any]] = []
    for family, family_data in shared.groupby("family", sort=True):
        networks = sorted(family_data["network"].unique())
        indexed = {
            network: family_data.loc[
                family_data["network"].eq(network)
            ].set_index("point_id")
            for network in networks
        }
        for source in networks:
            source_good = set(
                indexed[source].index[indexed[source]["is_good_point"]]
            )
            for target in networks:
                if source == target:
                    continue
                target_rows = indexed[target].loc[sorted(source_good)]
                percentiles = target_rows["objective_percentile"]
                rows.append(
                    {
                        "family": family,
                        "source_network": source,
                        "target_network": target,
                        "n_source_good_points": int(len(source_good)),
                        "target_good_overlap_count": int(
                            target_rows["is_good_point"].sum()
                        ),
                        "target_percentile_q1": float(percentiles.quantile(0.25)),
                        "target_percentile_median": float(percentiles.median()),
                        "target_percentile_q3": float(percentiles.quantile(0.75)),
                    }
                )
    return pd.DataFrame(rows)


def build_reference_comparisons(
    added_trials: pd.DataFrame,
    fixed: pd.DataFrame,
) -> pd.DataFrame:
    """Compare evaluated points with same-seed none and simple-max references."""

    reference = fixed.pivot(
        index="network", columns="condition", values="value"
    )
    if set(reference.columns) != {"none", "simple_max"}:
        raise ValueError("fixed references must contain none and simple_max")
    rows: list[dict[str, Any]] = []
    for network, group in added_trials.groupby("network", sort=True):
        if network not in reference.index:
            raise ValueError(f"fixed references are missing for {network}")
        for subset_name, subset in (
            ("all_adaptive_evaluations", group),
            ("shared_random_search", group.loc[group["method"].eq(PRIMARY_METHOD)]),
        ):
            none = float(reference.loc[network, "none"])
            simple_max = float(reference.loc[network, "simple_max"])
            best = float(subset["value"].min())
            rows.append(
                {
                    "network": network,
                    "subset": subset_name,
                    "n_evaluations": int(len(subset)),
                    "none_value": none,
                    "simple_max_value": simple_max,
                    "best_observed_value": best,
                    "count_better_than_none": int((subset["value"] < none).sum()),
                    "fraction_better_than_none": float(
                        (subset["value"] < none).mean()
                    ),
                    "count_better_than_simple_max": int(
                        (subset["value"] < simple_max).sum()
                    ),
                    "fraction_better_than_simple_max": float(
                        (subset["value"] < simple_max).mean()
                    ),
                    "best_relative_suppression_vs_none": float(
                        (none - best) / none
                    ),
                    "best_relative_suppression_vs_simple_max": float(
                        (simple_max - best) / simple_max
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_family_rank_landscape(
    shared: pd.DataFrame,
    *,
    family: str,
    networks: Sequence[str],
    labels: Mapping[str, str],
    output_path: str | Path,
) -> None:
    """Plot common coordinates colored by within-network objective percentile."""

    data = shared.loc[shared["family"].eq(family)]
    if set(data["network"].unique()) != set(networks):
        raise ValueError(f"plot networks for {family} do not match shared data")
    figure, axes = plt.subplots(
        1,
        len(networks),
        figsize=(4.8 * len(networks), 4.8),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes_array = np.atleast_1d(axes)
    normalization = Normalize(vmin=0.0, vmax=100.0)
    scatter = None
    for axis, network in zip(axes_array, networks, strict=True):
        network_data = data.loc[data["network"].eq(network)].sort_values(
            "objective_percentile", ascending=False, kind="stable"
        )
        scatter = axis.scatter(
            network_data["applied_certainty"],
            network_data["applied_effectiveness"],
            c=network_data["objective_percentile"],
            cmap="viridis",
            norm=normalization,
            s=25,
            alpha=0.78,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(labels.get(network, network), pad=10)
        axis.set_xlim(0.49, 1.01)
        axis.set_ylim(0.49, 1.01)
        axis.set_xticks(np.arange(0.5, 1.01, 0.1))
        axis.set_yticks(np.arange(0.5, 1.01, 0.1))
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Certainty")
        axis.grid(True, color="#D9DEE3", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes_array[0].set_ylabel("Effectiveness")
    figure.suptitle(
        f"{family} family: common random-search objective ranks",
        fontsize=15,
    )
    if scatter is None:
        raise RuntimeError("no points were plotted")
    colorbar = figure.colorbar(scatter, ax=axes_array, fraction=0.025, pad=0.02)
    colorbar.set_label("Within-network objective percentile (0 = best)")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
