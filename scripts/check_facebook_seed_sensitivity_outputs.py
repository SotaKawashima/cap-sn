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
DEFAULT_RUN_BASE = REPO_ROOT / "experiments/2026-06-10_facebook_seed_sensitivity/strategy_runs"
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
    candidates = [p for p in base.glob("seed_sensitivity_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"seed_sensitivity_* ディレクトリが見つかりません: {base}")
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


def summarize_strategy(run_root: Path, seed: int, strategy: str) -> dict[str, float | str | int]:
    result_dir = run_root / f"seed_{seed}" / "facebook" / strategy / "result"
    identifier = f"facebook_seed{seed}_{strategy}"
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

    summary: dict[str, float | str | int] = {
        "network": "facebook",
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


def aggregate_by_strategy(rows: list[dict]) -> list[dict]:
    output = []
    for strategy in STRATEGIES:
        strategy_rows = [r for r in rows if r["strategy"] == strategy]
        if not strategy_rows:
            continue
        record: dict[str, float | str | int] = {
            "network": "facebook",
            "strategy": strategy,
            "seed_count": len(strategy_rows),
        }
        for metric in KEY_METRICS:
            values = [float(r[metric]) for r in strategy_rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = sample_std(values)
            record[f"{metric}_min"] = min(values)
            record[f"{metric}_max"] = max(values)
            record[f"{metric}_spread"] = max(values) - min(values)
        output.append(record)
    return output


def contrast_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_seed_strategy = {
        (int(row["seed_state"]), str(row["strategy"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed_state"]) for row in rows})
    per_seed = []
    for seed in seeds:
        balance = by_seed_strategy.get((seed, "balance"))
        if not balance:
            continue
        for strategy in ["effective_high", "certainty_high"]:
            row = by_seed_strategy.get((seed, strategy))
            if not row:
                continue
            for metric in KEY_METRICS:
                current = float(row[metric])
                base = float(balance[metric])
                per_seed.append(
                    {
                        "seed_state": seed,
                        "contrast": f"{strategy}_minus_balance",
                        "metric": metric,
                        "current": current,
                        "balance": base,
                        "diff": current - base,
                        "abs_diff": abs(current - base),
                    }
                )

    aggregate = []
    contrast_names = sorted({str(r["contrast"]) for r in per_seed})
    for contrast in contrast_names:
        for metric in KEY_METRICS:
            values = [
                float(r["diff"])
                for r in per_seed
                if r["contrast"] == contrast and r["metric"] == metric
            ]
            if not values:
                continue
            aggregate.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean_diff": mean(values),
                    "std_diff": sample_std(values),
                    "min_diff": min(values),
                    "max_diff": max(values),
                    "positive_seed_count": sum(1 for v in values if v > 0),
                    "negative_seed_count": sum(1 for v in values if v < 0),
                    "zero_seed_count": sum(1 for v in values if v == 0),
                }
            )
    return per_seed, aggregate


def validate_master_log(run_root: Path, seeds: list[int]) -> list[str]:
    logs = sorted((run_root / "logs").glob("run_facebook_seed_sensitivity_*.log"))
    if not logs:
        return ["master log が見つかりません"]
    text = logs[-1].read_text(errors="replace")
    messages = []
    for seed in seeds:
        for strategy in STRATEGIES:
            start = f"Start: seed={seed}, network=facebook, strategy={strategy}"
            finish = f"Finished: seed={seed}, network=facebook, strategy={strategy}"
            if start not in text:
                messages.append(f"Start missing: seed={seed}, strategy={strategy}")
            if finish not in text:
                messages.append(f"Finished missing: seed={seed}, strategy={strategy}")
    if "=== All facebook seed sensitivity runs finished ===" not in text:
        messages.append("all-finished marker missing")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Facebook複数seed再実験のArrow出力を確認し、seed間の頑健性を集計します。"
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    args = parser.parse_args()

    run_root = args.run_root if args.run_root else latest_run_root(args.run_base)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root

    seeds = discover_seeds(run_root)
    log_messages = validate_master_log(run_root, seeds)
    if log_messages:
        raise RuntimeError("ログ確認で未完了の可能性があります: " + "; ".join(log_messages))

    rows = [
        summarize_strategy(run_root, seed, strategy)
        for seed in seeds
        for strategy in STRATEGIES
    ]
    aggregate_rows = aggregate_by_strategy(rows)
    contrast_per_seed, contrast_aggregate = contrast_rows(rows)

    analysis_dir = run_root / "analysis"
    summary_path = analysis_dir / "facebook_seed_sensitivity_summary.csv"
    aggregate_path = analysis_dir / "facebook_seed_sensitivity_aggregate.csv"
    contrast_path = analysis_dir / "facebook_seed_sensitivity_contrasts.csv"
    contrast_aggregate_path = analysis_dir / "facebook_seed_sensitivity_contrast_aggregate.csv"

    write_csv(summary_path, rows)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(contrast_path, contrast_per_seed)
    write_csv(contrast_aggregate_path, contrast_aggregate)

    print(f"[OK] run_root: {run_root}")
    print(f"seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"saved: {summary_path}")
    print(f"saved: {aggregate_path}")
    print(f"saved: {contrast_path}")
    print(f"saved: {contrast_aggregate_path}")


if __name__ == "__main__":
    main()
