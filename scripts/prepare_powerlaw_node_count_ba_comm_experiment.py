#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import networkx as nx
except ModuleNotFoundError:
    print("networkx が必要です。例: pip install networkx", file=sys.stderr)
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "v2/test_2/network"
DEFAULT_COMM_SOURCE = NETWORK_ROOT / "ba/ba1000/comm.csv"


@dataclass(frozen=True)
class Level:
    name: str
    attachment_edges: int
    triad_probability: float


DEFAULT_LEVELS = [
    Level("deg40_low", 20, 0.0),
    Level("deg40_high", 20, 1.0),
]


def read_comm_levels(path: Path) -> list[float]:
    with path.open(newline="") as f:
        return [float(row["level"]) for row in csv.DictReader(f)]


def connected_powerlaw_cluster_graph(
    *,
    num_nodes: int,
    attachment_edges: int,
    triad_probability: float,
    seed: int,
    max_retries: int,
) -> tuple[nx.Graph, int]:
    for offset in range(max_retries):
        trial_seed = seed + offset
        graph = nx.powerlaw_cluster_graph(
            n=num_nodes,
            m=attachment_edges,
            p=triad_probability,
            seed=trial_seed,
        )
        graph.remove_edges_from(nx.selfloop_edges(graph))
        if nx.is_connected(graph):
            return graph, trial_seed

    raise RuntimeError(
        "connected graph を生成できませんでした: "
        f"n={num_nodes}, m={attachment_edges}, p={triad_probability}, "
        f"seed={seed}, retries={max_retries}"
    )


def write_edgelist(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u, v in sorted((min(u, v), max(u, v)) for u, v in graph.edges()):
            f.write(f"{u} {v}\n")


def write_resampled_comm(path: Path, source_levels: list[float], num_nodes: int, seed: int) -> None:
    rng = random.Random(seed)
    sampled_levels = rng.choices(source_levels, k=num_nodes)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "agent_idx"])
        for agent_idx, level in enumerate(sampled_levels):
            writer.writerow([level, agent_idx])


def write_network_config(path: Path, network_dir_name: str, output_subdir: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'path = "./{output_subdir}/{network_dir_name}/"',
                'graph = "edgelist.txt"',
                "directed = false",
                "transposed = false",
                'community = "comm.csv"',
                "",
            ]
        )
    )


def graph_metrics(graph: nx.Graph) -> dict[str, object]:
    degrees = [degree for _, degree in graph.degree()]
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "avg_degree": sum(degrees) / len(degrees),
        "avg_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
        "largest_connected_component_ratio": 1.0,
        "max_degree": max(degrees),
    }


def comm_metrics(path: Path) -> dict[str, object]:
    levels = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            levels.append(float(row["level"]))

    sorted_levels = sorted(levels)
    n = len(sorted_levels)

    def quantile(q: float) -> float:
        return sorted_levels[round((n - 1) * q)]

    return {
        "comm_mean": sum(levels) / n,
        "comm_std": statistics.pstdev(levels),
        "comm_min": sorted_levels[0],
        "comm_q10": quantile(0.10),
        "comm_q25": quantile(0.25),
        "comm_median": quantile(0.50),
        "comm_q75": quantile(0.75),
        "comm_q90": quantile(0.90),
        "comm_max": sorted_levels[-1],
        "comm_count_zero": sum(level == 0.0 for level in levels),
        "comm_count_one": sum(level == 1.0 for level in levels),
    }


def parse_levels(raw_levels: list[str] | None) -> list[Level]:
    if not raw_levels:
        return DEFAULT_LEVELS

    levels = []
    for raw in raw_levels:
        try:
            name, m_value, p_value = raw.split(":", 2)
            levels.append(
                Level(
                    name=name,
                    attachment_edges=int(m_value),
                    triad_probability=float(p_value),
                )
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--level は name:m:p 形式で指定してください: {raw}"
            ) from exc
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ノード数を変えた Powerlaw cluster graph を生成し、"
            "BA1000由来の comm.csv 分布を復元抽出して実験セットを用意します。"
        )
    )
    parser.add_argument("--num-nodes", type=int, nargs="+", default=[2000, 3000, 4000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--level",
        action="append",
        help="水準を name:m:p 形式で指定。例: --level deg40_high:20:1.0",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260525,
        help="グラフ生成seed。既存 powerlaw_node_count と同じデフォルト。",
    )
    parser.add_argument("--comm-seed-base", type=int, default=20260526)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--output-subdir", default="powerlaw_node_count_ba_comm")
    parser.add_argument("--comm-source", type=Path, default=DEFAULT_COMM_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.level)
    source_levels = read_comm_levels(args.comm_source)
    graph_root = NETWORK_ROOT / args.output_subdir
    summary_rows = []
    config_paths = []

    for num_nodes in args.num_nodes:
        for level in levels:
            for seed_index in args.seeds:
                network_name = f"pncba_n{num_nodes}_{level.name}_seed{seed_index}"
                network_dir = graph_root / network_name
                config_path = NETWORK_ROOT / f"network-{network_name}.toml"
                generation_seed = (
                    args.seed_base
                    + num_nodes
                    + seed_index * 1000
                    + level.attachment_edges * 10
                    + int(level.triad_probability * 100)
                )
                comm_seed = args.comm_seed_base + num_nodes + seed_index * 1000

                graph, actual_seed = connected_powerlaw_cluster_graph(
                    num_nodes=num_nodes,
                    attachment_edges=level.attachment_edges,
                    triad_probability=level.triad_probability,
                    seed=generation_seed,
                    max_retries=args.max_retries,
                )

                graph_path = network_dir / "edgelist.txt"
                comm_path = network_dir / "comm.csv"
                write_edgelist(graph, graph_path)
                write_resampled_comm(comm_path, source_levels, num_nodes, comm_seed)
                write_network_config(config_path, network_name, args.output_subdir)
                config_paths.append(config_path)

                row = {
                    "network": network_name,
                    "node_level": f"n{num_nodes}",
                    "degree_level": level.name.split("_", 1)[0],
                    "cluster_level": level.name.split("_", 1)[1],
                    "attachment_edges_m": level.attachment_edges,
                    "target_avg_degree": level.attachment_edges * 2,
                    "triad_probability": level.triad_probability,
                    "requested_seed": generation_seed,
                    "actual_seed": actual_seed,
                    "comm_seed": comm_seed,
                    "comm_source": str(args.comm_source.relative_to(REPO_ROOT)),
                    "config_path": str(config_path.relative_to(REPO_ROOT)),
                    "graph_path": str(graph_path.relative_to(REPO_ROOT)),
                    "comm_path": str(comm_path.relative_to(REPO_ROOT)),
                    **graph_metrics(graph),
                    **comm_metrics(comm_path),
                }
                summary_rows.append(row)

    summary_path = graph_root / "generation_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} networks")
    print(f"Summary: {summary_path.relative_to(REPO_ROOT)}")
    print("Configs:")
    for path in config_paths:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
