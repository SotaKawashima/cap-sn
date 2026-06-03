#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import modularity


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "v2/test_2/network"
ANALYSIS_ROOT = (
    REPO_ROOT / "experiments/2026-06-02_lfr_facebook_pool/strategy_runs/analysis"
)
DEFAULT_TARGETS_CSV = ANALYSIS_ROOT / "facebook_rust_candidate_pool_targets.csv"
DEFAULT_SELECTED_CSVS = [
    ANALYSIS_ROOT / "lfr_mu02_rust_target_pool_selected_nodes.csv",
    ANALYSIS_ROOT / "lfr_mu02_extra_rust_target_pool_selected_nodes.csv",
]


@dataclass(frozen=True)
class SourceSpec:
    network: str
    source_subdir: str
    seed_index: int


@dataclass(frozen=True)
class PoolTarget:
    pool: str
    node_count: int
    communities_touched: int
    largest_comm_ratio: float
    degree_mean: float
    external_ratio_mean: float
    participation_mean: float
    betweenness_mean: float


DEFAULT_SOURCES = [
    SourceSpec("lfr_mu02_seed3", "lfr_low_external_candidate", 3),
    SourceSpec("lfr_mu02_seed4", "lfr_low_external_mu02_extra", 4),
    SourceSpec("lfr_mu02_seed5", "lfr_low_external_mu02_extra", 5),
]
DEFAULT_METHODS = ["original", "random", "rust_target"]
POOL_ORDER = ["top20", "middle20", "bottom20"]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[f])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def read_levels(path: Path, num_nodes: int) -> list[float]:
    levels = [math.nan] * num_nodes
    with path.open() as f:
        for row in csv.DictReader(f):
            agent_idx = int(row["agent_idx"])
            levels[agent_idx] = float(row["level"])
    if any(math.isnan(level) for level in levels):
        raise ValueError(f"comm.csv does not cover all agent indexes: {path}")
    return levels


def write_comm_csv(path: Path, levels: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "agent_idx"])
        for agent_idx, level in enumerate(levels):
            writer.writerow([level, agent_idx])


def read_lfr_communities(path: Path) -> dict[int, int]:
    comm_id = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            comm_id[int(row["agent_idx"])] = int(row["lfr_community"])
    return comm_id


def read_graph(path: Path, num_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            u, v = stripped.split()[:2]
            graph.add_edge(int(u), int(v))
    return graph


def read_targets(path: Path) -> dict[str, PoolTarget]:
    targets = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            pool = row["pool"]
            targets[pool] = PoolTarget(
                pool=pool,
                node_count=int(row["node_count"]),
                communities_touched=int(row["communities_touched"]),
                largest_comm_ratio=float(row["largest_comm_ratio"]),
                degree_mean=float(row["degree_mean"]),
                external_ratio_mean=float(row["external_ratio_mean"]),
                participation_mean=float(row["participation_mean"]),
                betweenness_mean=float(row["betweenness_mean"]),
            )
    missing = [pool for pool in POOL_ORDER if pool not in targets]
    if missing:
        raise ValueError(f"target CSV is missing pools: {missing}")
    return targets


def read_selected_nodes(paths: list[Path]) -> dict[str, dict[str, list[int]]]:
    selected: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as f:
            for row in csv.DictReader(f):
                selected[row["network"]][row["pool"]].append(int(row["agent_idx"]))

    for network, pools in selected.items():
        for pool in POOL_ORDER:
            if pool not in pools:
                continue
            nodes = pools[pool]
            if len(nodes) != len(set(nodes)):
                raise ValueError(f"duplicate selected nodes: {network} {pool}")
    return {network: dict(pools) for network, pools in selected.items()}


def community_sets(comm_id: dict[int, int]) -> list[set[int]]:
    groups: dict[int, set[int]] = defaultdict(set)
    for node, cid in comm_id.items():
        groups[cid].add(node)
    return [groups[cid] for cid in sorted(groups)]


def graph_metrics(graph: nx.Graph) -> dict[str, object]:
    degrees = [degree for _, degree in graph.degree()]
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "avg_degree": statistics.fmean(degrees),
        "median_degree": percentile(degrees, 50),
        "p90_degree": percentile(degrees, 90),
        "p99_degree": percentile(degrees, 99),
        "avg_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
        "max_degree": max(degrees),
    }


def lfr_metrics(graph: nx.Graph, comm_id: dict[int, int]) -> dict[str, object]:
    communities = community_sets(comm_id)
    internal_edges = 0
    external_edges = 0
    for u, v in graph.edges():
        if comm_id[u] == comm_id[v]:
            internal_edges += 1
        else:
            external_edges += 1
    edge_count = graph.number_of_edges()
    sizes = [len(community) for community in communities]
    return {
        "lfr_communities": len(communities),
        "lfr_modularity": modularity(graph, communities),
        "lfr_internal_edges": internal_edges,
        "lfr_external_edges": external_edges,
        "lfr_internal_edge_ratio": internal_edges / edge_count,
        "lfr_external_edge_ratio": external_edges / edge_count,
        "lfr_comm_size_min": min(sizes),
        "lfr_comm_size_median": percentile(sizes, 50),
        "lfr_comm_size_max": max(sizes),
    }


def node_boundary_metrics(
    graph: nx.Graph,
    comm_id: dict[int, int],
) -> dict[int, dict[str, float]]:
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
                (count / degree) ** 2
                for count in neighbor_community_counts.values()
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


def comm_metrics(levels: list[float]) -> dict[str, object]:
    return {
        "comm_mean": statistics.fmean(levels),
        "comm_std": statistics.pstdev(levels),
        "comm_min": min(levels),
        "comm_q10": percentile(levels, 10),
        "comm_q25": percentile(levels, 25),
        "comm_median": percentile(levels, 50),
        "comm_q75": percentile(levels, 75),
        "comm_q90": percentile(levels, 90),
        "comm_max": max(levels),
        "comm_count_zero": sum(level == 0.0 for level in levels),
        "comm_count_one": sum(level == 1.0 for level in levels),
    }


def rust_indexes_by_level(levels: list[float]) -> list[int]:
    return [
        node
        for node, _level in sorted(
            enumerate(levels),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def support_pools(levels: list[float]) -> dict[str, list[int]]:
    indexes_by_level = rust_indexes_by_level(levels)
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


def support_pool_metrics(
    *,
    graph: nx.Graph,
    levels: list[float],
    comm_id: dict[int, int],
    boundary_metrics: dict[int, dict[str, float]],
    targets: dict[str, PoolTarget],
) -> dict[str, object]:
    rows = {}
    for prefix, nodes in support_pools(levels).items():
        target = targets[prefix]
        degrees = [boundary_metrics[node]["degree"] for node in nodes]
        external_degrees = [boundary_metrics[node]["external_degree"] for node in nodes]
        external_ratios = [boundary_metrics[node]["external_ratio"] for node in nodes]
        participations = [boundary_metrics[node]["participation"] for node in nodes]
        community_ids = [comm_id[node] for node in nodes]
        counts = sorted(
            [community_ids.count(community) for community in set(community_ids)],
            reverse=True,
        )
        largest_ratio = counts[0] / len(nodes)
        degree_mean = statistics.fmean(degrees)
        external_ratio_mean = statistics.fmean(external_ratios)
        participation_mean = statistics.fmean(participations)

        rows[f"{prefix}_count"] = len(nodes)
        rows[f"{prefix}_degree_mean"] = degree_mean
        rows[f"{prefix}_degree_median"] = percentile(degrees, 50)
        rows[f"{prefix}_degree_p90"] = percentile(degrees, 90)
        rows[f"{prefix}_max_degree"] = max(degrees)
        rows[f"{prefix}_external_degree_mean"] = statistics.fmean(external_degrees)
        rows[f"{prefix}_external_ratio_mean"] = external_ratio_mean
        rows[f"{prefix}_external_ratio_median"] = percentile(external_ratios, 50)
        rows[f"{prefix}_external_ratio_p90"] = percentile(external_ratios, 90)
        rows[f"{prefix}_participation_mean"] = participation_mean
        rows[f"{prefix}_participation_median"] = percentile(participations, 50)
        rows[f"{prefix}_participation_p90"] = percentile(participations, 90)
        rows[f"{prefix}_communities_touched"] = len(counts)
        rows[f"{prefix}_largest_comm_count"] = counts[0]
        rows[f"{prefix}_largest_comm_ratio"] = largest_ratio
        rows[f"{prefix}_target_largest_comm_ratio"] = target.largest_comm_ratio
        rows[f"{prefix}_target_degree_mean"] = target.degree_mean
        rows[f"{prefix}_target_external_ratio_mean"] = target.external_ratio_mean
        rows[f"{prefix}_target_participation_mean"] = target.participation_mean
        rows[f"{prefix}_largest_comm_ratio_abs_error"] = abs(
            largest_ratio - target.largest_comm_ratio
        )
        rows[f"{prefix}_degree_mean_abs_error"] = abs(
            degree_mean - target.degree_mean
        )
        rows[f"{prefix}_external_ratio_abs_error"] = abs(
            external_ratio_mean - target.external_ratio_mean
        )
        rows[f"{prefix}_participation_abs_error"] = abs(
            participation_mean - target.participation_mean
        )
    return rows


def levels_from_ranked_nodes(ranked_nodes: list[int], num_nodes: int) -> list[float]:
    if len(ranked_nodes) != num_nodes or len(set(ranked_nodes)) != num_nodes:
        raise ValueError("ranked_nodes must contain every node exactly once")
    levels = [0.0] * num_nodes
    for rank, node in enumerate(ranked_nodes):
        levels[node] = (num_nodes - rank) / num_nodes
    return levels


def random_levels(num_nodes: int, rng: random.Random) -> list[float]:
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)
    return levels_from_ranked_nodes(nodes, num_nodes)


def rust_target_levels(
    *,
    selected: dict[str, list[int]],
    num_nodes: int,
    rng: random.Random,
) -> tuple[list[float], dict[str, object]]:
    missing = [pool for pool in POOL_ORDER if pool not in selected]
    if missing:
        raise ValueError(f"selected nodes missing pools: {missing}")

    pools = {pool: list(selected[pool]) for pool in POOL_ORDER}
    pool_size = round(num_nodes * 0.2)
    for pool, nodes in pools.items():
        if len(nodes) != pool_size:
            raise ValueError(f"{pool} has {len(nodes)} nodes, expected {pool_size}")

    selected_nodes = set()
    for pool, nodes in pools.items():
        overlap = selected_nodes.intersection(nodes)
        if overlap:
            raise ValueError(f"{pool} overlaps previous pools: {sorted(overlap)[:5]}")
        selected_nodes.update(nodes)

    filler = [node for node in range(num_nodes) if node not in selected_nodes]
    if len(filler) != num_nodes - pool_size * 3:
        raise ValueError("unexpected filler size")

    rng.shuffle(filler)
    high_filler = filler[:pool_size]
    low_filler = filler[pool_size:]

    ranked_nodes = []
    pool_positions = {}
    for label, part in [
        ("top20", pools["top20"]),
        ("high_filler", high_filler),
        ("middle20", pools["middle20"]),
        ("low_filler", low_filler),
        ("bottom20", pools["bottom20"]),
    ]:
        rng.shuffle(part)
        start = len(ranked_nodes)
        ranked_nodes.extend(part)
        pool_positions[f"{label}_rank_start"] = start
        pool_positions[f"{label}_rank_end"] = len(ranked_nodes) - 1

    return levels_from_ranked_nodes(ranked_nodes, num_nodes), pool_positions


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


def parse_source(raw: str) -> SourceSpec:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--source must be network:source_subdir:seed_index"
        )
    network, source_subdir, seed_index = parts
    return SourceSpec(network, source_subdir, int(seed_index))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create LFR mu=0.02 variants whose support-level candidate pools "
            "match the Facebook Rust indexes_by_level target as closely as the "
            "precomputed feasibility search allowed."
        )
    )
    parser.add_argument("--output-subdir", default="lfr_rust_target_pool")
    parser.add_argument("--seed-base", type=int, default=20260603)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument(
        "--selected-csv",
        type=Path,
        action="append",
        help="CSV produced by check_lfr_rust_pool_target_feasibility.py.",
    )
    parser.add_argument(
        "--source",
        type=parse_source,
        action="append",
        help="Source as network:source_subdir:seed_index.",
    )
    parser.add_argument(
        "--method",
        choices=DEFAULT_METHODS,
        action="append",
        help="Method to generate. Defaults to original, random, rust_target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.source or DEFAULT_SOURCES
    methods = args.method or DEFAULT_METHODS
    selected_paths = args.selected_csv or DEFAULT_SELECTED_CSVS
    targets = read_targets(args.targets_csv)
    selected_nodes = read_selected_nodes(selected_paths)

    output_root = NETWORK_ROOT / args.output_subdir
    rows = []
    config_paths = []

    for source in sources:
        source_dir = NETWORK_ROOT / source.source_subdir / source.network
        source_graph_path = source_dir / "edgelist.txt"
        source_comm_path = source_dir / "comm.csv"
        source_lfr_path = source_dir / "lfr_communities.csv"
        if not source_graph_path.exists():
            raise FileNotFoundError(source_graph_path)
        if not source_comm_path.exists():
            raise FileNotFoundError(source_comm_path)
        if not source_lfr_path.exists():
            raise FileNotFoundError(source_lfr_path)

        comm_id = read_lfr_communities(source_lfr_path)
        num_nodes = len(comm_id)
        graph = read_graph(source_graph_path, num_nodes)
        original_levels = read_levels(source_comm_path, num_nodes)
        boundary_metrics = node_boundary_metrics(graph, comm_id)

        if "rust_target" in methods and source.network not in selected_nodes:
            raise ValueError(f"no selected nodes for {source.network}")

        for method_index, method in enumerate(methods):
            network_name = f"lfrrustpool_mu02_{method}_seed{source.seed_index}"
            network_dir = output_root / network_name
            config_path = NETWORK_ROOT / f"network-{network_name}.toml"
            comm_path = network_dir / "comm.csv"
            method_seed = args.seed_base + source.seed_index * 1000 + method_index * 100
            rng = random.Random(method_seed)

            network_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_graph_path, network_dir / "edgelist.txt")
            shutil.copyfile(source_lfr_path, network_dir / "lfr_communities.csv")

            method_metadata = {
                "top20_rank_start": None,
                "top20_rank_end": None,
                "high_filler_rank_start": None,
                "high_filler_rank_end": None,
                "middle20_rank_start": None,
                "middle20_rank_end": None,
                "low_filler_rank_start": None,
                "low_filler_rank_end": None,
                "bottom20_rank_start": None,
                "bottom20_rank_end": None,
            }
            if method == "original":
                levels = original_levels
                shutil.copyfile(source_comm_path, comm_path)
            elif method == "random":
                levels = random_levels(num_nodes, rng)
                write_comm_csv(comm_path, levels)
            elif method == "rust_target":
                levels, method_metadata = rust_target_levels(
                    selected=selected_nodes[source.network],
                    num_nodes=num_nodes,
                    rng=rng,
                )
                write_comm_csv(comm_path, levels)
            else:
                raise ValueError(f"unknown method: {method}")

            write_network_config(config_path, network_name, args.output_subdir)
            config_paths.append(config_path)

            rows.append(
                {
                    "network": network_name,
                    "source_network": source.network,
                    "source_subdir": source.source_subdir,
                    "mu": 0.02,
                    "seed_index": source.seed_index,
                    "comm_method": method,
                    "method_seed": method_seed if method != "original" else None,
                    "targets_csv": str(args.targets_csv.relative_to(REPO_ROOT)),
                    "selected_csvs": ";".join(
                        str(path.relative_to(REPO_ROOT)) for path in selected_paths
                    ),
                    "config_path": str(config_path.relative_to(REPO_ROOT)),
                    "graph_path": str((network_dir / "edgelist.txt").relative_to(REPO_ROOT)),
                    "comm_path": str(comm_path.relative_to(REPO_ROOT)),
                    "lfr_community_path": str(
                        (network_dir / "lfr_communities.csv").relative_to(REPO_ROOT)
                    ),
                    **method_metadata,
                    **graph_metrics(graph),
                    **lfr_metrics(graph, comm_id),
                    **comm_metrics(levels),
                    **support_pool_metrics(
                        graph=graph,
                        levels=levels,
                        comm_id=comm_id,
                        boundary_metrics=boundary_metrics,
                        targets=targets,
                    ),
                }
            )

    summary_path = output_root / "generation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} network variants")
    print(f"Summary: {summary_path.relative_to(REPO_ROOT)}")
    print("Configs:")
    for path in config_paths:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
