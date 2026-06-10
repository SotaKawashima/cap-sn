#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import pyarrow.ipc as ipc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_BASE = REPO_ROOT / "experiments/2026-06-10_facebook_recheck/strategy_runs"
DEFAULT_BASELINE = (
    REPO_ROOT
    / "experiments/2026-05_real_network_strategy/strategy_runs/analysis/comparison_summary.csv"
)

TOTAL_AGENTS = 4039
STRATEGIES = ["balance", "effective_high", "certainty_high"]
INFO_NAMES = {
    0: "misinformation",
    1: "corrective",
    2: "observational",
    3: "behavior_guiding",
}


def latest_run_root(base: Path) -> Path:
    candidates = [p for p in base.glob("recheck_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"recheck_* ディレクトリが見つかりません: {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_arrow(path: Path) -> list[dict]:
    with ipc.open_file(path) as reader:
        return reader.read_all().to_pylist()


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def summarize_strategy(run_root: Path, strategy: str) -> dict[str, float | str | int]:
    result_dir = run_root / "facebook" / strategy / "result"
    identifier = f"facebook_{strategy}"
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
        "strategy": strategy,
        "total_agents": TOTAL_AGENTS,
        "mean_final_selfish_ratio": mean(final_ratios),
        "max_final_selfish_ratio": max(final_ratios) if final_ratios else float("nan"),
        "peak_mean_selfish_ratio": peak_mean_selfish_ratio,
    }

    for info_label, info_name in INFO_NAMES.items():
        values = [
            totals
            for (num_iter, label), totals in by_iter_label.items()
            if label == info_label
        ]
        shared = mean([v["num_shared"] for v in values])
        viewed = mean([v["num_viewed"] for v in values])
        fst_viewed = mean([v["num_fst_viewed"] for v in values])
        summary[f"mean_total_shared_ratio_{info_name}"] = shared / TOTAL_AGENTS
        summary[f"mean_total_viewed_ratio_{info_name}"] = viewed / TOTAL_AGENTS
        summary[f"mean_total_fst_viewed_ratio_{info_name}"] = fst_viewed / TOTAL_AGENTS

    return summary


def load_baseline(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["strategy"]: row
        for row in rows
        if row.get("network") == "facebook"
    }


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


def compare_to_baseline(rows: list[dict], baseline_path: Path) -> list[dict]:
    baseline = load_baseline(baseline_path)
    compare_cols = [
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
    output = []
    for row in rows:
        strategy = row["strategy"]
        base = baseline.get(strategy)
        if not base:
            continue
        for col in compare_cols:
            current = float(row[col])
            previous = float(base[col])
            output.append(
                {
                    "strategy": strategy,
                    "metric": col,
                    "recheck": current,
                    "baseline": previous,
                    "diff": current - previous,
                    "abs_diff": abs(current - previous),
                }
            )
    return output


def validate_master_log(run_root: Path) -> list[str]:
    logs = sorted((run_root / "logs").glob("run_facebook_recheck_*.log"))
    if not logs:
        return ["master log が見つかりません"]
    text = logs[-1].read_text(errors="replace")
    messages = []
    for strategy in STRATEGIES:
        if f"Start: network=facebook, strategy={strategy}" not in text:
            messages.append(f"Start missing: {strategy}")
        if f"Finished: network=facebook, strategy={strategy}" not in text:
            messages.append(f"Finished missing: {strategy}")
    if "=== All facebook recheck runs finished ===" not in text:
        messages.append("all-finished marker missing")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Facebook再実験のArrow出力を確認し、既存Facebook結果と比較します。"
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    run_root = args.run_root if args.run_root else latest_run_root(args.run_base)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root

    log_messages = validate_master_log(run_root)
    if log_messages:
        raise RuntimeError("ログ確認で未完了の可能性があります: " + "; ".join(log_messages))

    rows = [summarize_strategy(run_root, strategy) for strategy in STRATEGIES]
    analysis_dir = run_root / "analysis"
    summary_path = analysis_dir / "facebook_recheck_summary.csv"
    compare_path = analysis_dir / "facebook_recheck_vs_baseline.csv"
    write_csv(summary_path, rows)
    write_csv(compare_path, compare_to_baseline(rows, args.baseline))

    print(f"[OK] run_root: {run_root}")
    print(f"saved: {summary_path}")
    print(f"saved: {compare_path}")


if __name__ == "__main__":
    main()
