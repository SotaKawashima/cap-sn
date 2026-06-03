#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

import networkx as nx


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETWORK_DIR = REPO_ROOT / "v2/test_2/network/facebook"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/2026-06-02_lfr_facebook_pool/strategy_runs/analysis"
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def read_levels(path: Path) -> list[float]:
    levels_by_node = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            levels_by_node[int(row["agent_idx"])] = float(row["level"])
    return [levels_by_node[i] for i in range(max(levels_by_node) + 1)]


def read_graph(path: Path, node_count: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            u, v = stripped.split()[:2]
            graph.add_edge(int(u), int(v))
    return graph


def rust_indexes_by_level(levels: list[float]) -> list[int]:
    # Rust: levels.iter().enumerate().sorted_by(level desc).
    # itertools::sorted_by collects to Vec and calls stable Vec::sort_by,
    # so equal levels keep CSV/agent-index order.
    return [
        node
        for node, _level in sorted(
            enumerate(levels),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def previous_target_indexes_by_level(levels: list[float]) -> list[int]:
    # This reproduces the Facebook target values used on 2026-06-02:
    # level desc, and agent index desc for equal levels.
    return [
        node
        for node, _level in sorted(
            enumerate(levels),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
    ]


def support_pools_from_indexes(levels: list[float], indexes_by_level: list[int]) -> dict[str, list[int]]:
    n = len(indexes_by_level)
    k = round(n * 0.2)
    center = n // 2
    if n % 2 == 1:
        median = levels[indexes_by_level[center]]
    else:
        median = (levels[indexes_by_level[center]] + levels[indexes_by_level[center - 1]]) / 2

    window = range(max(0, center - k), min(n, center + k))
    middle_positions = sorted(
        window,
        key=lambda i: abs(levels[indexes_by_level[i]] - median),
    )[:k]

    return {
        "top20": indexes_by_level[:k],
        "middle20": [indexes_by_level[i] for i in middle_positions],
        "bottom20": list(reversed(indexes_by_level))[:k],
    }


def community_ids(graph: nx.Graph, seed: int) -> dict[int, int]:
    communities = nx.algorithms.community.louvain_communities(graph, seed=seed)
    return {
        node: community_idx
        for community_idx, community in enumerate(communities)
        for node in community
    }


def node_boundary_metrics(graph: nx.Graph, comm_id: dict[int, int]) -> dict[int, dict[str, float]]:
    metrics = {}
    for node in graph.nodes:
        degree = graph.degree(node)
        external_degree = 0
        neighbor_community_counts = Counter()
        for neighbor in graph.neighbors(node):
            neighbor_comm = comm_id[neighbor]
            neighbor_community_counts[neighbor_comm] += 1
            if neighbor_comm != comm_id[node]:
                external_degree += 1

        if degree:
            external_ratio = external_degree / degree
            participation = 1.0 - sum(
                (count / degree) ** 2 for count in neighbor_community_counts.values()
            )
        else:
            external_ratio = 0.0
            participation = 0.0

        metrics[node] = {
            "degree": float(degree),
            "external_degree": float(external_degree),
            "external_ratio": float(external_ratio),
            "participation": float(participation),
        }
    return metrics


def summarize_pool(
    *,
    order_name: str,
    pool_name: str,
    nodes: list[int],
    boundary: dict[int, dict[str, float]],
    betweenness: dict[int, float],
    comm_id: dict[int, int],
) -> dict[str, object]:
    community_counts = Counter(comm_id[node] for node in nodes)
    row: dict[str, object] = {
        "order": order_name,
        "pool": pool_name,
        "node_count": len(nodes),
        "communities_touched": len(community_counts),
        "largest_comm_count": max(community_counts.values()),
        "largest_comm_ratio": max(community_counts.values()) / len(nodes),
    }

    metric_values = {
        "degree": [boundary[node]["degree"] for node in nodes],
        "external_degree": [boundary[node]["external_degree"] for node in nodes],
        "external_ratio": [boundary[node]["external_ratio"] for node in nodes],
        "participation": [boundary[node]["participation"] for node in nodes],
        "betweenness": [float(betweenness[node]) for node in nodes],
    }
    for metric, values in metric_values.items():
        row[f"{metric}_mean"] = statistics.fmean(values)
        row[f"{metric}_median"] = percentile(values, 50)
        row[f"{metric}_p90"] = percentile(values, 90)
        row[f"{metric}_max"] = max(values)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Facebookのsupport level候補プールをRustのSupportLevelTable順序で再定義し、"
            "コミュニティ集中度・外部接続性を保存します。"
        )
    )
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--louvain-seed", type=int, default=0)
    parser.add_argument("--betweenness-samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = read_levels(args.network_dir / "comm.csv")
    graph = read_graph(args.network_dir / "edgelist.txt", len(levels))
    comm_id = community_ids(graph, args.louvain_seed)
    boundary = node_boundary_metrics(graph, comm_id)
    betweenness = nx.betweenness_centrality(
        graph,
        k=min(args.betweenness_samples, graph.number_of_nodes()),
        seed=args.louvain_seed,
        normalized=True,
    )

    order_specs = {
        "rust_indexes_by_level": rust_indexes_by_level(levels),
        "previous_target_order": previous_target_indexes_by_level(levels),
    }

    rows = []
    for order_name, indexes_by_level in order_specs.items():
        pools = support_pools_from_indexes(levels, indexes_by_level)
        for pool_name, nodes in pools.items():
            rows.append(
                summarize_pool(
                    order_name=order_name,
                    pool_name=pool_name,
                    nodes=nodes,
                    boundary=boundary,
                    betweenness=betweenness,
                    comm_id=comm_id,
                )
            )

    metrics_path = args.output_dir / "facebook_rust_candidate_pool_metrics.csv"
    write_csv(metrics_path, rows)

    rust_rows = [row for row in rows if row["order"] == "rust_indexes_by_level"]
    rust_path = args.output_dir / "facebook_rust_candidate_pool_targets.csv"
    write_csv(rust_path, rust_rows)

    print("saved:", metrics_path)
    print("saved:", rust_path)
    for row in rust_rows:
        print(
            row["pool"],
            "largest_comm_ratio=",
            round(float(row["largest_comm_ratio"]), 3),
            "external_ratio_mean=",
            round(float(row["external_ratio_mean"]), 3),
            "degree_mean=",
            round(float(row["degree_mean"]), 3),
        )


if __name__ == "__main__":
    main()
