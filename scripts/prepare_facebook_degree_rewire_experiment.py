#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import modularity


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "v2/test_2/network"
DEFAULT_SOURCE_DIR = NETWORK_ROOT / "facebook"
DEFAULT_OUTPUT_SUBDIR = "facebook_degree_rewire"


@dataclass(frozen=True)
class RewireSpec:
    network: str
    label: str
    swap_ratio: float


DEFAULT_REWIRE_SPECS = [
    RewireSpec("fbdeg_original", "original", 0.0),
    RewireSpec("fbdeg_rewire_0p1", "rewire_0p1", 0.1),
    RewireSpec("fbdeg_rewire_1p0", "rewire_1p0", 1.0),
    RewireSpec("fbdeg_rewire_5p0", "rewire_5p0", 5.0),
]
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


def read_levels(path: Path) -> list[float]:
    levels_by_node = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            levels_by_node[int(row["agent_idx"])] = float(row["level"])
    node_count = max(levels_by_node) + 1
    return [levels_by_node[i] for i in range(node_count)]


def write_edgelist(path: Path, graph: nx.Graph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u, v in sorted((min(u, v), max(u, v)) for u, v in graph.edges()):
            f.write(f"{u} {v}\n")


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


def community_sets(comm_id: dict[int, int]) -> list[set[int]]:
    groups: dict[int, set[int]] = {}
    for node, cid in comm_id.items():
        groups.setdefault(cid, set()).add(node)
    return [groups[cid] for cid in sorted(groups)]


def louvain_community_ids(graph: nx.Graph, seed: int) -> dict[int, int]:
    communities = nx.algorithms.community.louvain_communities(graph, seed=seed)
    return {
        node: community_idx
        for community_idx, community in enumerate(communities)
        for node in community
    }


def graph_metrics(graph: nx.Graph, comm_id: dict[int, int]) -> dict[str, object]:
    degrees = [degree for _, degree in graph.degree()]
    communities = community_sets(comm_id)
    internal_edges = 0
    external_edges = 0
    for u, v in graph.edges():
        if comm_id[u] == comm_id[v]:
            internal_edges += 1
        else:
            external_edges += 1

    edge_count = graph.number_of_edges()
    components = [len(c) for c in nx.connected_components(graph)]
    community_sizes = [len(community) for community in communities]
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": edge_count,
        "avg_degree": statistics.fmean(degrees),
        "median_degree": percentile(degrees, 50),
        "p90_degree": percentile(degrees, 90),
        "p99_degree": percentile(degrees, 99),
        "max_degree": max(degrees),
        "avg_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
        "num_connected_components": len(components),
        "largest_connected_component_ratio": max(components) / graph.number_of_nodes(),
        "louvain_communities": len(communities),
        "louvain_modularity": modularity(graph, communities),
        "louvain_internal_edges": internal_edges,
        "louvain_external_edges": external_edges,
        "louvain_internal_edge_ratio": internal_edges / edge_count,
        "louvain_external_edge_ratio": external_edges / edge_count,
        "louvain_comm_size_min": min(community_sizes),
        "louvain_comm_size_median": percentile(community_sizes, 50),
        "louvain_comm_size_max": max(community_sizes),
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


def support_pool_metrics(
    *,
    levels: list[float],
    comm_id: dict[int, int],
    boundary_metrics: dict[int, dict[str, float]],
) -> dict[str, object]:
    rows = {}
    for prefix, nodes in support_pools(levels).items():
        degrees = [boundary_metrics[node]["degree"] for node in nodes]
        external_degrees = [boundary_metrics[node]["external_degree"] for node in nodes]
        external_ratios = [boundary_metrics[node]["external_ratio"] for node in nodes]
        participations = [boundary_metrics[node]["participation"] for node in nodes]
        community_ids = [comm_id[node] for node in nodes]
        counts = sorted(
            [community_ids.count(community) for community in set(community_ids)],
            reverse=True,
        )
        rows[f"{prefix}_count"] = len(nodes)
        rows[f"{prefix}_degree_mean"] = statistics.fmean(degrees)
        rows[f"{prefix}_degree_median"] = percentile(degrees, 50)
        rows[f"{prefix}_degree_p90"] = percentile(degrees, 90)
        rows[f"{prefix}_max_degree"] = max(degrees)
        rows[f"{prefix}_external_degree_mean"] = statistics.fmean(external_degrees)
        rows[f"{prefix}_external_ratio_mean"] = statistics.fmean(external_ratios)
        rows[f"{prefix}_external_ratio_median"] = percentile(external_ratios, 50)
        rows[f"{prefix}_external_ratio_p90"] = percentile(external_ratios, 90)
        rows[f"{prefix}_participation_mean"] = statistics.fmean(participations)
        rows[f"{prefix}_participation_median"] = percentile(participations, 50)
        rows[f"{prefix}_participation_p90"] = percentile(participations, 90)
        rows[f"{prefix}_communities_touched"] = len(counts)
        rows[f"{prefix}_largest_comm_count"] = counts[0]
        rows[f"{prefix}_largest_comm_ratio"] = counts[0] / len(nodes)
    return rows


def degree_sequence(graph: nx.Graph) -> list[int]:
    return [degree for _node, degree in sorted(graph.degree(), key=lambda item: item[0])]


def rewire_graph(
    source_graph: nx.Graph,
    *,
    swap_ratio: float,
    seed: int,
    max_tries_multiplier: int,
) -> tuple[nx.Graph, int]:
    graph = source_graph.copy()
    requested_swaps = round(source_graph.number_of_edges() * swap_ratio)
    if requested_swaps == 0:
        return graph, 0
    nx.double_edge_swap(
        graph,
        nswap=requested_swaps,
        max_tries=max(requested_swaps * max_tries_multiplier, requested_swaps + 1),
        seed=seed,
    )
    return graph, requested_swaps


def parse_spec(raw: str) -> RewireSpec:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--variant must be network:label:swap_ratio"
        )
    network, label, swap_ratio = parts
    return RewireSpec(network=network, label=label, swap_ratio=float(swap_ratio))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Facebook-derived degree-preserving rewired graph variants. "
            "comm.csv is copied unchanged so the experiment isolates graph edits."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--rewire-seed-base", type=int, default=20260616)
    parser.add_argument("--louvain-seed", type=int, default=20260616)
    parser.add_argument("--max-tries-multiplier", type=int, default=50)
    parser.add_argument(
        "--variant",
        type=parse_spec,
        action="append",
        help="Variant as network:label:swap_ratio. Defaults to original, 0.1m, 1.0m, 5.0m swaps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = args.variant or DEFAULT_REWIRE_SPECS
    output_root = NETWORK_ROOT / args.output_subdir
    source_graph_path = args.source_dir / "edgelist.txt"
    source_comm_path = args.source_dir / "comm.csv"
    levels = read_levels(source_comm_path)
    source_graph = read_graph(source_graph_path, len(levels))
    source_degrees = degree_sequence(source_graph)

    rows = []
    config_paths = []
    for variant_index, spec in enumerate(variants):
        rewire_seed = args.rewire_seed_base + variant_index
        graph, requested_swaps = rewire_graph(
            source_graph,
            swap_ratio=spec.swap_ratio,
            seed=rewire_seed,
            max_tries_multiplier=args.max_tries_multiplier,
        )
        if degree_sequence(graph) != source_degrees:
            raise RuntimeError(f"degree sequence changed unexpectedly: {spec.network}")

        network_dir = output_root / spec.network
        graph_path = network_dir / "edgelist.txt"
        comm_path = network_dir / "comm.csv"
        config_path = NETWORK_ROOT / f"network-{spec.network}.toml"

        network_dir.mkdir(parents=True, exist_ok=True)
        write_edgelist(graph_path, graph)
        shutil.copyfile(source_comm_path, comm_path)
        write_network_config(config_path, spec.network, args.output_subdir)
        config_paths.append(config_path)

        comm_id = louvain_community_ids(graph, args.louvain_seed)
        boundary = node_boundary_metrics(graph, comm_id)
        rows.append(
            {
                "network": spec.network,
                "source_network": "facebook",
                "variant": spec.label,
                "swap_ratio": spec.swap_ratio,
                "requested_swaps": requested_swaps,
                "rewire_seed": rewire_seed if requested_swaps else None,
                "louvain_seed": args.louvain_seed,
                "degree_sequence_preserved": True,
                "comm_source": str(source_comm_path.relative_to(REPO_ROOT)),
                "config_path": str(config_path.relative_to(REPO_ROOT)),
                "graph_path": str(graph_path.relative_to(REPO_ROOT)),
                "comm_path": str(comm_path.relative_to(REPO_ROOT)),
                **graph_metrics(graph, comm_id),
                **support_pool_metrics(
                    levels=levels,
                    comm_id=comm_id,
                    boundary_metrics=boundary,
                ),
            }
        )

    summary_path = output_root / "generation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} Facebook-derived variants")
    print(f"Summary: {summary_path.relative_to(REPO_ROOT)}")
    print("Configs:")
    for path in config_paths:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
