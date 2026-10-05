#!/usr/bin/env python3
"""Plot added-network objective values using the Stage 6 visual encoding."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

from plot_final_retest_relative_suppression import (
    REPO_ROOT,
    configure_japanese_font,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments"
    / "summer_2026"
    / "network_similarity_optimization"
    / "20260923_113148_network_similarity_optimization_v01"
    / "network_similarity_analysis_v01"
    / "tables"
    / "combined_trial_inventory.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notes" / "figures"


@dataclass(frozen=True)
class NetworkSpec:
    network: str
    label: str
    colorbar_format: str


BA_SPECS = (
    NetworkSpec("ba1000_seed2", "BA1000 seed 2", "%.3f"),
    NetworkSpec("ba1000_seed3", "BA1000 seed 3", "%.3f"),
    NetworkSpec("ba1000_seed4", "BA1000 seed 4", "%.3f"),
)
BA_ORIGINAL_SPEC = NetworkSpec("ba1000", "BA1000 seed 1", "%.3f")
BA_ALL_SPECS = (BA_ORIGINAL_SPEC, *BA_SPECS)
FACEBOOK_SPECS = (
    NetworkSpec("facebook_brandeis99", "Brandeis99", "%.4f"),
    NetworkSpec("facebook_bucknell39", "Bucknell39", "%.4f"),
    NetworkSpec("facebook_rice31", "Rice31", "%.4f"),
)
FACEBOOK_ORIGINAL_SPEC = NetworkSpec(
    "facebook",
    "SNAP ego-Facebook",
    "%.4f",
)
FACEBOOK_ALL_SPECS = (FACEBOOK_ORIGINAL_SPEC, *FACEBOOK_SPECS)
WIKI_SPEC = NetworkSpec("wiki_rfa_post2008", "Wiki-RfA", "%.4f")
WIKI_ORIGINAL_SPEC = NetworkSpec("wiki_vote", "Wiki-vote", "%.4f")
WIKI_ALL_SPECS = (WIKI_ORIGINAL_SPEC, WIKI_SPEC)
ALL_SPECS = (*BA_ALL_SPECS, *FACEBOOK_ALL_SPECS, *WIKI_ALL_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_trials(path: Path) -> pd.DataFrame:
    trials = pd.read_parquet(path)
    required = {
        "network",
        "method",
        "optimizer_seed",
        "evaluation",
        "applied_certainty",
        "applied_effectiveness",
        "state",
        "value",
    }
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"trial inventory is missing columns: {missing}")
    expected_counts = {
        (spec.network, method): 300
        for spec in ALL_SPECS
        for method in ("bo_gp", "cma_es", "random_search")
    }
    actual_counts = (
        trials.loc[trials["network"].isin(spec.network for spec in ALL_SPECS)]
        .groupby(["network", "method"])
        .size()
        .to_dict()
    )
    if actual_counts != expected_counts:
        raise ValueError(f"unexpected network-method counts: {actual_counts}")
    selected = trials.loc[
        trials["network"].isin(spec.network for spec in ALL_SPECS)
    ].copy()
    if len(selected) != 9000:
        raise ValueError(f"expected 9000 combined evaluations, found {len(selected)}")
    if set(selected["state"].astype(str)) != {"COMPLETE"}:
        raise ValueError(f"unexpected trial states: {sorted(set(selected['state']))}")
    plot_columns = [
        "applied_certainty",
        "applied_effectiveness",
        "value",
    ]
    if selected[plot_columns].isna().any().any():
        raise ValueError("trial inventory contains missing plot values")
    return selected


def _style_axis(axis: plt.Axes, *, title: str, show_ylabel: bool) -> None:
    axis.set_xlim(0.49, 1.01)
    axis.set_ylim(0.49, 1.01)
    axis.set_xticks(np.arange(0.5, 1.01, 0.1))
    axis.set_yticks(np.arange(0.5, 1.01, 0.1))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("確実性")
    if show_ylabel:
        axis.set_ylabel("有効性")
    axis.set_title(title, pad=10)
    axis.grid(True, color="#D9DEE3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7A848C")
    axis.spines["bottom"].set_color("#7A848C")
    axis.tick_params(colors="#37444D")


def _plot_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    trials: pd.DataFrame,
    spec: NetworkSpec,
    *,
    show_ylabel: bool,
) -> None:
    network_trials = (
        trials.loc[trials["network"].eq(spec.network)]
        .sort_values("value", ascending=False, kind="stable")
        .copy()
    )
    if len(network_trials) != 900:
        raise ValueError(
            f"expected 900 evaluations for {spec.network}, "
            f"found {len(network_trials)}"
        )
    normalization = Normalize(
        vmin=float(network_trials["value"].min()),
        vmax=float(network_trials["value"].max()),
    )
    scatter = axis.scatter(
        network_trials["applied_certainty"],
        network_trials["applied_effectiveness"],
        c=network_trials["value"],
        cmap="viridis_r",
        norm=normalization,
        s=16,
        alpha=0.70,
        edgecolors="none",
        rasterized=True,
    )
    _style_axis(axis, title=spec.label, show_ylabel=show_ylabel)
    colorbar = figure.colorbar(
        scatter,
        ax=axis,
        fraction=0.047,
        pad=0.025,
        aspect=24,
    )
    colorbar.set_label(r"$J_{\mathrm{cum}}$（低いほど良い）", fontsize=10)
    colorbar.ax.yaxis.set_major_formatter(
        FormatStrFormatter(spec.colorbar_format)
    )


def plot_networks(
    trials: pd.DataFrame,
    specs: tuple[NetworkSpec, ...],
    *,
    title: str,
    output_stem: str,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(specs),
        figsize=(5.65 * len(specs), 5.6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes_array = np.atleast_1d(axes)
    for index, (axis, spec) in enumerate(zip(axes_array, specs, strict=True)):
        _plot_panel(
            figure,
            axis,
            trials,
            spec,
            show_ylabel=index == 0,
        )
    figure.suptitle(title, fontsize=17)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / output_stem
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    configure_japanese_font()
    trials = load_trials(args.input)
    plot_networks(
        trials,
        BA_SPECS,
        title="BA1000系列の目的関数値のパラメータ空間分布",
        output_stem="ba1000_added_objective_parameter_space_comparison",
        output_dir=args.output_dir,
    )
    plot_networks(
        trials,
        FACEBOOK_SPECS,
        title="Facebook系列の目的関数値のパラメータ空間分布",
        output_stem="facebook_added_objective_parameter_space_comparison",
        output_dir=args.output_dir,
    )
    plot_networks(
        trials,
        (WIKI_SPEC,),
        title="Wiki-RfAの目的関数値のパラメータ空間分布",
        output_stem="wiki_rfa_objective_parameter_space",
        output_dir=args.output_dir,
    )
    plot_networks(
        trials,
        BA_ALL_SPECS,
        title="BA1000 seed 1～4の目的関数値のパラメータ空間分布",
        output_stem="ba1000_seed1_to_seed4_objective_parameter_space_comparison",
        output_dir=args.output_dir,
    )
    plot_networks(
        trials,
        FACEBOOK_ALL_SPECS,
        title="Facebook系列4グラフの目的関数値のパラメータ空間分布",
        output_stem="facebook_original_and_added_objective_parameter_space_comparison",
        output_dir=args.output_dir,
    )
    plot_networks(
        trials,
        WIKI_ALL_SPECS,
        title="Wiki-voteとWiki-RfAの目的関数値のパラメータ空間分布",
        output_stem="wiki_vote_rfa_objective_parameter_space_comparison",
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
