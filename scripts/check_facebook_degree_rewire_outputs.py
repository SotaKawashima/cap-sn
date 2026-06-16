#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.ipc as ipc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_BASE = REPO_ROOT / "experiments/2026-06-16_facebook_degree_rewire/strategy_runs"
DEFAULT_GENERATION_SUMMARY = (
    REPO_ROOT / "v2/test_2/network/facebook_degree_rewire/generation_summary.csv"
)
TOTAL_AGENTS = 4039
STRATEGIES = ["balance", "effective_high", "certainty_high"]
INFO_NAMES = {
    0: "misinformation",
    1: "corrective",
    2: "observational",
    3: "behavior_guiding",
}
KEY_METRICS = [
    "peak_mean_selfish_ratio",
    "mean_total_shared_ratio_misinformation",
    "mean_total_shared_ratio_corrective",
    "mean_total_shared_ratio_observational",
    "mean_total_shared_ratio_behavior_guiding",
    "mean_total_fst_viewed_ratio_misinformation",
    "mean_total_fst_viewed_ratio_corrective",
    "mean_total_fst_viewed_ratio_observational",
    "mean_total_fst_viewed_ratio_behavior_guiding",
]


def latest_run_root(base: Path) -> Path:
    candidates = [p for p in base.glob("degree_rewire_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"degree_rewire_* ディレクトリが見つかりません: {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_arrow(path: Path) -> list[dict]:
    with ipc.open_file(path) as reader:
        return reader.read_all().to_pylist()


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def discover_seeds(run_root: Path) -> list[int]:
    seeds = []
    for path in run_root.glob("seed_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"seed_(\d+)", path.name)
        if match:
            seeds.append(int(match.group(1)))
    if not seeds:
        raise FileNotFoundError(f"seed_* ディレクトリが見つかりません: {run_root}")
    return sorted(seeds)


def discover_networks(run_root: Path, seeds: list[int]) -> list[str]:
    networks = set()
    for seed in seeds:
        seed_dir = run_root / f"seed_{seed}"
        for path in seed_dir.glob("fbdeg_*"):
            if path.is_dir():
                networks.add(path.name)
    if not networks:
        raise FileNotFoundError(f"fbdeg_* ディレクトリが見つかりません: {run_root}")
    return sorted(networks)


def load_generation_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["network"]: row for row in csv.DictReader(f)}


def summarize_run(
    run_root: Path,
    seed: int,
    network: str,
    strategy: str,
    metadata: dict[str, dict[str, str]],
) -> dict[str, float | str | int]:
    result_dir = run_root / f"seed_{seed}" / network / strategy / "result"
    identifier = f"{network}_seed{seed}_{strategy}"
    info_path = result_dir / f"{identifier}_info.arrow"
    pop_path = result_dir / f"{identifier}_pop.arrow"
    agent_path = result_dir / f"{identifier}_agent.arrow"

    missing = [p for p in [info_path, pop_path, agent_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("必要なArrowが見つかりません: " + ", ".join(str(p) for p in missing))
    empty = [p for p in [info_path, pop_path, agent_path] if p.stat().st_size == 0]
    if empty:
        raise ValueError("空のArrowがあります: " + ", ".join(str(p) for p in empty))

    info_rows = read_arrow(info_path)
    pop_rows = read_arrow(pop_path)

    pop_by_iter: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in pop_rows:
        pop_by_iter[int(row["num_iter"])].append((int(row["t"]), int(row["num_selfish"])))

    final_ratios = []
    time_selfish: dict[int, list[int]] = defaultdict(list)
    for rows in pop_by_iter.values():
        rows.sort(key=lambda item: item[0])
        final_ratios.append(rows[-1][1] / TOTAL_AGENTS)
        for t, num_selfish in rows:
            time_selfish[t].append(num_selfish)

    peak_mean_selfish_ratio = max(
        mean(values) / TOTAL_AGENTS for values in time_selfish.values()
    )

    by_iter_label: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"num_shared": 0, "num_viewed": 0, "num_fst_viewed": 0}
    )
    for row in info_rows:
        key = (int(row["num_iter"]), int(row["info_label"]))
        by_iter_label[key]["num_shared"] += int(row["num_shared"])
        by_iter_label[key]["num_viewed"] += int(row["num_viewed"])
        by_iter_label[key]["num_fst_viewed"] += int(row["num_fst_viewed"])

    meta = metadata.get(network, {})
    summary: dict[str, float | str | int] = {
        "network": network,
        "variant": meta.get("variant", network),
        "swap_ratio": meta.get("swap_ratio", ""),
        "seed_state": seed,
        "strategy": strategy,
        "total_agents": TOTAL_AGENTS,
        "iteration_count_observed": len(pop_by_iter),
        "mean_final_selfish_ratio": mean(final_ratios),
        "max_final_selfish_ratio": max(final_ratios) if final_ratios else float("nan"),
        "peak_mean_selfish_ratio": peak_mean_selfish_ratio,
    }

    for info_label, info_name in INFO_NAMES.items():
        values = [
            totals
            for (_, label), totals in by_iter_label.items()
            if label == info_label
        ]
        shared = mean([v["num_shared"] for v in values])
        viewed = mean([v["num_viewed"] for v in values])
        fst_viewed = mean([v["num_fst_viewed"] for v in values])
        summary[f"mean_total_shared_ratio_{info_name}"] = shared / TOTAL_AGENTS
        summary[f"mean_total_viewed_ratio_{info_name}"] = viewed / TOTAL_AGENTS
        summary[f"mean_total_fst_viewed_ratio_{info_name}"] = fst_viewed / TOTAL_AGENTS

    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["network"], row["variant"], row["strategy"]) for row in rows})
    for network, variant, strategy in keys:
        group = [
            row
            for row in rows
            if row["network"] == network and row["strategy"] == strategy
        ]
        record: dict[str, float | str | int] = {
            "network": network,
            "variant": variant,
            "strategy": strategy,
            "seed_count": len(group),
        }
        for metric in KEY_METRICS:
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = sample_std(values)
            record[f"{metric}_min"] = min(values)
            record[f"{metric}_max"] = max(values)
        output.append(record)
    return output


def vs_original_rows(rows: list[dict], original_network: str = "fbdeg_original") -> list[dict]:
    by_key = {
        (row["seed_state"], row["network"], row["strategy"]): row
        for row in rows
    }
    output = []
    for row in rows:
        if row["network"] == original_network:
            continue
        base = by_key.get((row["seed_state"], original_network, row["strategy"]))
        if base is None:
            continue
        for metric in KEY_METRICS:
            current = float(row[metric])
            original = float(base[metric])
            output.append(
                {
                    "seed_state": row["seed_state"],
                    "network": row["network"],
                    "variant": row["variant"],
                    "strategy": row["strategy"],
                    "metric": metric,
                    "current": current,
                    "original": original,
                    "diff": current - original,
                    "abs_diff": abs(current - original),
                }
            )
    return output


def strategy_contrast_rows(rows: list[dict]) -> list[dict]:
    by_key = {
        (row["seed_state"], row["network"], row["strategy"]): row
        for row in rows
    }
    seed_networks = sorted({(row["seed_state"], row["network"], row["variant"]) for row in rows})
    pair_specs = [
        ("effective_high_minus_balance", "effective_high", "balance"),
        ("certainty_high_minus_balance", "certainty_high", "balance"),
        ("certainty_high_minus_effective_high", "certainty_high", "effective_high"),
    ]
    output = []

    for seed, network, variant in seed_networks:
        for contrast, left_strategy, right_strategy in pair_specs:
            left = by_key.get((seed, network, left_strategy))
            right = by_key.get((seed, network, right_strategy))
            if left is None or right is None:
                continue
            for metric in KEY_METRICS:
                left_value = float(left[metric])
                right_value = float(right[metric])
                output.append(
                    {
                        "seed_state": seed,
                        "network": network,
                        "variant": variant,
                        "contrast": contrast,
                        "metric": metric,
                        "left_strategy": left_strategy,
                        "right_strategy": right_strategy,
                        "left_value": left_value,
                        "right_value": right_value,
                        "diff": left_value - right_value,
                        "abs_diff": abs(left_value - right_value),
                    }
                )

        for metric in KEY_METRICS:
            strategy_values = []
            for strategy in STRATEGIES:
                row = by_key.get((seed, network, strategy))
                if row is not None:
                    strategy_values.append((strategy, float(row[metric])))
            if len(strategy_values) != len(STRATEGIES):
                continue
            min_strategy, min_value = min(strategy_values, key=lambda item: item[1])
            max_strategy, max_value = max(strategy_values, key=lambda item: item[1])
            output.append(
                {
                    "seed_state": seed,
                    "network": network,
                    "variant": variant,
                    "contrast": "max_minus_min_strategy",
                    "metric": metric,
                    "left_strategy": max_strategy,
                    "right_strategy": min_strategy,
                    "left_value": max_value,
                    "right_value": min_value,
                    "diff": max_value - min_value,
                    "abs_diff": max_value - min_value,
                }
            )

    return output


def aggregate_contrasts(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["network"], row["variant"], row["contrast"], row["metric"]) for row in rows})
    for network, variant, contrast, metric in keys:
        group = [
            row
            for row in rows
            if row["network"] == network
            and row["contrast"] == contrast
            and row["metric"] == metric
        ]
        diffs = [float(row["diff"]) for row in group]
        abs_diffs = [float(row["abs_diff"]) for row in group]
        output.append(
            {
                "network": network,
                "variant": variant,
                "contrast": contrast,
                "metric": metric,
                "seed_count": len(group),
                "mean_diff": mean(diffs),
                "std_diff": sample_std(diffs),
                "min_diff": min(diffs),
                "max_diff": max(diffs),
                "mean_abs_diff": mean(abs_diffs),
                "positive_seed_count": sum(1 for value in diffs if value > 0),
                "negative_seed_count": sum(1 for value in diffs if value < 0),
                "zero_seed_count": sum(1 for value in diffs if value == 0),
            }
        )
    return output


def validate_master_log(run_root: Path, seeds: list[int], networks: list[str]) -> list[str]:
    logs = sorted((run_root / "logs").glob("run_facebook_degree_rewire_*.log"))
    if not logs:
        return ["master log が見つかりません"]
    text = logs[-1].read_text(errors="replace")
    messages = []
    for seed in seeds:
        for network in networks:
            for strategy in STRATEGIES:
                start = f"Start: seed={seed}, network={network}, strategy={strategy}"
                finish = f"Finished: seed={seed}, network={network}, strategy={strategy}"
                if start not in text:
                    messages.append(f"Start missing: seed={seed}, network={network}, strategy={strategy}")
                if finish not in text:
                    messages.append(f"Finished missing: seed={seed}, network={network}, strategy={strategy}")
    if "=== All facebook degree-rewire runs finished ===" not in text:
        messages.append("all-finished marker missing")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Facebook次数保存rewiring実験のArrow出力を確認し、構造改変ごとの結果を集計します。"
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    parser.add_argument("--generation-summary", type=Path, default=DEFAULT_GENERATION_SUMMARY)
    args = parser.parse_args()

    run_root = args.run_root if args.run_root else latest_run_root(args.run_base)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root

    seeds = discover_seeds(run_root)
    networks = discover_networks(run_root, seeds)
    log_messages = validate_master_log(run_root, seeds, networks)
    if log_messages:
        raise RuntimeError("ログ確認で未完了の可能性があります: " + "; ".join(log_messages))

    metadata = load_generation_metadata(args.generation_summary)
    rows = [
        summarize_run(run_root, seed, network, strategy, metadata)
        for seed in seeds
        for network in networks
        for strategy in STRATEGIES
    ]
    aggregate = aggregate_rows(rows)
    diffs = vs_original_rows(rows)
    strategy_diffs = strategy_contrast_rows(rows)
    strategy_diff_aggregate = aggregate_contrasts(strategy_diffs)

    analysis_dir = run_root / "analysis"
    summary_path = analysis_dir / "facebook_degree_rewire_summary.csv"
    aggregate_path = analysis_dir / "facebook_degree_rewire_aggregate.csv"
    diff_path = analysis_dir / "facebook_degree_rewire_vs_original.csv"
    strategy_diff_path = analysis_dir / "facebook_degree_rewire_strategy_contrasts.csv"
    strategy_diff_aggregate_path = analysis_dir / "facebook_degree_rewire_strategy_contrast_aggregate.csv"
    write_csv(summary_path, rows)
    write_csv(aggregate_path, aggregate)
    write_csv(diff_path, diffs)
    write_csv(strategy_diff_path, strategy_diffs)
    write_csv(strategy_diff_aggregate_path, strategy_diff_aggregate)

    print(f"[OK] run_root: {run_root}")
    print(f"seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"networks: {', '.join(networks)}")
    print(f"saved: {summary_path}")
    print(f"saved: {aggregate_path}")
    print(f"saved: {diff_path}")
    print(f"saved: {strategy_diff_path}")
    print(f"saved: {strategy_diff_aggregate_path}")


if __name__ == "__main__":
    main()
