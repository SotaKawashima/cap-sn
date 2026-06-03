#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETWORK_ROOT = REPO_ROOT / "v2/test_2/network/lfr_community"
DEFAULT_TARGETS_CSV = (
    REPO_ROOT
    / "experiments/2026-06-02_lfr_facebook_pool/strategy_runs/analysis/facebook_rust_candidate_pool_targets.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/2026-06-02_lfr_facebook_pool/strategy_runs/analysis"
)
POOL_ORDER = ["top20", "middle20", "bottom20"]


@dataclass(frozen=True)
class PoolTarget:
    pool: str
    largest_comm_ratio: float
    communities_touched: int
    degree_mean: float
    external_ratio_mean: float
    participation_mean: float
    betweenness_mean: float


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


def read_targets(path: Path) -> dict[str, PoolTarget]:
    targets = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pool = row["pool"]
            targets[pool] = PoolTarget(
                pool=pool,
                largest_comm_ratio=float(row["largest_comm_ratio"]),
                communities_touched=int(row["communities_touched"]),
                degree_mean=float(row["degree_mean"]),
                external_ratio_mean=float(row["external_ratio_mean"]),
                participation_mean=float(row["participation_mean"]),
                betweenness_mean=float(row["betweenness_mean"]),
            )
    return targets


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


def read_lfr_communities(path: Path) -> dict[int, int]:
    comm_id = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            comm_id[int(row["agent_idx"])] = int(row["lfr_community"])
    return comm_id


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


def node_fit_score(node: int, metrics: dict[int, dict[str, float]], target: PoolTarget, betweenness: dict[int, float]) -> float:
    node_metrics = metrics[node]
    return (
        20.0 * abs(node_metrics["external_ratio"] - target.external_ratio_mean)
        + 10.0 * abs(node_metrics["participation"] - target.participation_mean)
        + abs(node_metrics["degree"] - target.degree_mean) / max(target.degree_mean, 1.0)
        + 100.0 * abs(float(betweenness[node]) - target.betweenness_mean)
    )


def balanced_counts(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def select_best_nodes(
    nodes: list[int],
    count: int,
    metrics: dict[int, dict[str, float]],
    target: PoolTarget,
    betweenness: dict[int, float],
) -> list[int] | None:
    if len(nodes) < count:
        return None
    ordered = sorted(nodes, key=lambda node: node_fit_score(node, metrics, target, betweenness))
    return ordered[:count]


def summarize_selection(
    *,
    network: str,
    pool: str,
    nodes: list[int],
    comm_id: dict[int, int],
    metrics: dict[int, dict[str, float]],
    betweenness: dict[int, float],
    target: PoolTarget,
) -> dict[str, object]:
    counts = Counter(comm_id[node] for node in nodes)
    degree_values = [metrics[node]["degree"] for node in nodes]
    external_degree_values = [metrics[node]["external_degree"] for node in nodes]
    external_ratio_values = [metrics[node]["external_ratio"] for node in nodes]
    participation_values = [metrics[node]["participation"] for node in nodes]
    betweenness_values = [float(betweenness[node]) for node in nodes]
    largest_ratio = max(counts.values()) / len(nodes)
    external_ratio_mean = statistics.fmean(external_ratio_values)
    participation_mean = statistics.fmean(participation_values)

    return {
        "network": network,
        "pool": pool,
        "node_count": len(nodes),
        "target_largest_comm_ratio": target.largest_comm_ratio,
        "largest_comm_ratio": largest_ratio,
        "largest_comm_ratio_abs_error": abs(largest_ratio - target.largest_comm_ratio),
        "target_communities_touched": target.communities_touched,
        "communities_touched": len(counts),
        "target_degree_mean": target.degree_mean,
        "degree_mean": statistics.fmean(degree_values),
        "degree_median": percentile(degree_values, 50),
        "target_external_ratio_mean": target.external_ratio_mean,
        "external_ratio_mean": external_ratio_mean,
        "external_ratio_median": percentile(external_ratio_values, 50),
        "external_ratio_abs_error": abs(external_ratio_mean - target.external_ratio_mean),
        "external_degree_mean": statistics.fmean(external_degree_values),
        "target_participation_mean": target.participation_mean,
        "participation_mean": participation_mean,
        "participation_abs_error": abs(participation_mean - target.participation_mean),
        "target_betweenness_mean": target.betweenness_mean,
        "betweenness_mean": statistics.fmean(betweenness_values),
        "anchor_community": counts.most_common(1)[0][0],
        "anchor_count": counts.most_common(1)[0][1],
    }


def pool_score(row: dict[str, object]) -> float:
    return (
        10.0 * float(row["largest_comm_ratio_abs_error"])
        + 30.0 * float(row["external_ratio_abs_error"])
        + 10.0 * float(row["participation_abs_error"])
        + abs(float(row["degree_mean"]) - float(row["target_degree_mean"]))
        / max(float(row["target_degree_mean"]), 1.0)
    )


def try_build_pool(
    *,
    network: str,
    pool: str,
    target: PoolTarget,
    available: set[int],
    comm_to_nodes: dict[int, list[int]],
    comm_id: dict[int, int],
    metrics: dict[int, dict[str, float]],
    betweenness: dict[int, float],
    pool_size: int,
) -> tuple[dict[str, object], list[int]] | None:
    target_anchor_count = round(pool_size * target.largest_comm_ratio)
    touched = min(target.communities_touched, len(comm_to_nodes))
    if touched < 1:
        return None

    best: tuple[float, dict[str, object], list[int]] | None = None
    community_ids = sorted(comm_to_nodes)
    for anchor in community_ids:
        anchor_available = [node for node in comm_to_nodes[anchor] if node in available]
        if len(anchor_available) < target_anchor_count:
            continue

        secondary_candidates = [cid for cid in community_ids if cid != anchor]
        for secondaries in itertools.combinations(secondary_candidates, touched - 1):
            remaining = pool_size - target_anchor_count
            secondary_counts = balanced_counts(remaining, touched - 1) if touched > 1 else []
            selected = select_best_nodes(anchor_available, target_anchor_count, metrics, target, betweenness)
            if selected is None:
                continue

            ok = True
            selected_nodes = list(selected)
            for cid, count in zip(secondaries, secondary_counts):
                candidates = [node for node in comm_to_nodes[cid] if node in available]
                picked = select_best_nodes(candidates, count, metrics, target, betweenness)
                if picked is None:
                    ok = False
                    break
                selected_nodes.extend(picked)

            if not ok or len(selected_nodes) != pool_size:
                continue

            row = summarize_selection(
                network=network,
                pool=pool,
                nodes=selected_nodes,
                comm_id=comm_id,
                metrics=metrics,
                betweenness=betweenness,
                target=target,
            )
            score = pool_score(row)
            if best is None or score < best[0]:
                best = (score, row, selected_nodes)

    if best is None:
        return None
    return best[1], best[2]


def build_assignment(
    *,
    network: str,
    targets: dict[str, PoolTarget],
    graph: nx.Graph,
    comm_id: dict[int, int],
    metrics: dict[int, dict[str, float]],
    betweenness: dict[int, float],
) -> tuple[list[dict[str, object]], dict[str, list[int]]] | None:
    comm_to_nodes: dict[int, list[int]] = {}
    for node, cid in comm_id.items():
        comm_to_nodes.setdefault(cid, []).append(node)

    pool_size = round(graph.number_of_nodes() * 0.2)
    best: tuple[float, list[dict[str, object]], dict[str, list[int]]] | None = None
    for order in itertools.permutations(POOL_ORDER):
        available = set(graph.nodes)
        rows = []
        selections = {}
        ok = True
        for pool in order:
            result = try_build_pool(
                network=network,
                pool=pool,
                target=targets[pool],
                available=available,
                comm_to_nodes=comm_to_nodes,
                comm_id=comm_id,
                metrics=metrics,
                betweenness=betweenness,
                pool_size=pool_size,
            )
            if result is None:
                ok = False
                break
            row, nodes = result
            rows.append(row)
            selections[pool] = nodes
            available.difference_update(nodes)

        if not ok:
            continue
        score = sum(pool_score(row) for row in rows)
        if best is None or score < best[0]:
            best = (score, rows, selections)

    if best is None:
        return None
    return best[1], best[2]


def lower_bound_rows(
    *,
    network: str,
    targets: dict[str, PoolTarget],
    graph: nx.Graph,
    metrics: dict[int, dict[str, float]],
) -> list[dict[str, object]]:
    pool_size = round(graph.number_of_nodes() * 0.2)
    rows = []
    nodes = list(graph.nodes)
    for pool, target in targets.items():
        lowest = sorted(nodes, key=lambda node: metrics[node]["external_ratio"])[:pool_size]
        mean_external = statistics.fmean(metrics[node]["external_ratio"] for node in lowest)
        rows.append(
            {
                "network": network,
                "pool": pool,
                "target_external_ratio_mean": target.external_ratio_mean,
                "lowest_possible_external_ratio_mean_without_comm_constraints": mean_external,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="既存LFR strongでFacebook Rust実順序の候補プール目標を組めるか探索します。"
    )
    parser.add_argument("--network-root", type=Path, default=DEFAULT_NETWORK_ROOT)
    parser.add_argument("--level-name", default="strong")
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="lfr_rust_target_pool")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--betweenness-samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = read_targets(args.targets_csv)
    feasibility_rows = []
    selected_node_rows = []
    lower_rows = []

    for seed in args.seeds:
        network = f"lfr_{args.level_name}_seed{seed}"
        network_dir = args.network_root / network
        comm_id = read_lfr_communities(network_dir / "lfr_communities.csv")
        graph = read_graph(network_dir / "edgelist.txt", len(comm_id))
        metrics = node_boundary_metrics(graph, comm_id)
        betweenness = nx.betweenness_centrality(
            graph,
            k=min(args.betweenness_samples, graph.number_of_nodes()),
            seed=20260603 + seed,
            normalized=True,
        )
        lower_rows.extend(lower_bound_rows(network=network, targets=targets, graph=graph, metrics=metrics))

        result = build_assignment(
            network=network,
            targets=targets,
            graph=graph,
            comm_id=comm_id,
            metrics=metrics,
            betweenness=betweenness,
        )
        if result is None:
            print("[NG]", network, "assignment not found")
            continue
        rows, selections = result
        feasibility_rows.extend(sorted(rows, key=lambda row: POOL_ORDER.index(str(row["pool"]))))
        for pool, nodes in selections.items():
            for node in nodes:
                selected_node_rows.append(
                    {
                        "network": network,
                        "pool": pool,
                        "agent_idx": node,
                        "lfr_community": comm_id[node],
                        "degree": metrics[node]["degree"],
                        "external_ratio": metrics[node]["external_ratio"],
                        "participation": metrics[node]["participation"],
                        "betweenness": float(betweenness[node]),
                    }
                )
        print("[OK]", network)

    feasibility_path = args.output_dir / f"{args.output_prefix}_feasibility.csv"
    nodes_path = args.output_dir / f"{args.output_prefix}_selected_nodes.csv"
    lower_path = args.output_dir / f"{args.output_prefix}_external_ratio_lower_bound.csv"
    write_csv(feasibility_path, feasibility_rows)
    write_csv(nodes_path, selected_node_rows)
    write_csv(lower_path, lower_rows)
    print("saved:", feasibility_path)
    print("saved:", nodes_path)
    print("saved:", lower_path)

    for row in feasibility_rows:
        print(
            row["network"],
            row["pool"],
            "largest=",
            round(float(row["largest_comm_ratio"]), 3),
            "target_largest=",
            round(float(row["target_largest_comm_ratio"]), 3),
            "external=",
            round(float(row["external_ratio_mean"]), 3),
            "target_external=",
            round(float(row["target_external_ratio_mean"]), 3),
        )


if __name__ == "__main__":
    main()
