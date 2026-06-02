#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

try:
    import networkx as nx
    from networkx.algorithms.community import modularity
except ModuleNotFoundError:
    print("networkx が必要です。例: pip install networkx", file=sys.stderr)
    raise

try:
    from cdlib import algorithms
except ModuleNotFoundError:
    print("cdlib が必要です。例: pip install cdlib", file=sys.stderr)
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "v2/test_2/network"


@dataclass(frozen=True)
class MuLevel:
    name: str
    mu: float


DEFAULT_LEVELS = [
    MuLevel("strong", 0.05),
    MuLevel("middle", 0.20),
    MuLevel("weak", 0.40),
]


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


def connected_lfr_graph(
    *,
    num_nodes: int,
    tau1: float,
    tau2: float,
    mu: float,
    average_degree: float,
    max_degree: int,
    min_community: int,
    max_community: int,
    seed: int,
    max_iters: int,
    max_retries: int,
) -> tuple[nx.Graph, int]:
    errors = []
    for offset in range(max_retries):
        trial_seed = seed + offset
        try:
            graph = nx.LFR_benchmark_graph(
                n=num_nodes,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=average_degree,
                max_degree=max_degree,
                min_community=min_community,
                max_community=max_community,
                seed=trial_seed,
                max_iters=max_iters,
            )
        except Exception as exc:
            errors.append(f"{trial_seed}: {type(exc).__name__}: {exc}")
            continue

        graph = nx.Graph(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        if nx.is_connected(graph):
            return graph, trial_seed
        errors.append(f"{trial_seed}: disconnected")

    detail = "; ".join(errors[:5])
    raise RuntimeError(
        "connected LFR graph を生成できませんでした: "
        f"n={num_nodes}, tau1={tau1}, tau2={tau2}, mu={mu}, "
        f"average_degree={average_degree}, max_degree={max_degree}, "
        f"min_community={min_community}, max_community={max_community}, "
        f"seed={seed}, retries={max_retries}. first errors: {detail}"
    )


def lfr_communities(graph: nx.Graph) -> list[set[int]]:
    communities = []
    seen = set()
    for node, data in graph.nodes(data=True):
        community = frozenset(data["community"])
        if community not in seen:
            seen.add(community)
            communities.append(set(community))
    return communities


def lfr_community_id(graph: nx.Graph, communities: list[set[int]]) -> dict[int, int]:
    comm_id = {}
    for idx, community in enumerate(communities):
        for node in community:
            comm_id[node] = idx
    if len(comm_id) != graph.number_of_nodes():
        raise ValueError("LFR community assignment does not cover all nodes")
    return comm_id


def allocation_value(comms, node: int, community_index: int) -> float:
    raw = float(comms.allocation_matrix[node][community_index])
    value = raw / 100.0 if raw > 1.0 else raw
    return min(1.0, max(0.0, value))


def write_comm_csv(
    *,
    graph: nx.Graph,
    output_path: Path,
    num_communities: int,
    community_index: int,
) -> list[float]:
    if community_index < 0 or community_index >= num_communities:
        raise ValueError(
            f"community_index must be in [0, {num_communities - 1}], "
            f"got {community_index}"
        )

    comms = algorithms.principled_clustering(graph, num_communities)
    nodes = sorted(graph.nodes())
    levels = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "agent_idx"])
        for agent_idx, node in enumerate(nodes):
            level = allocation_value(comms, node, community_index)
            levels.append(level)
            writer.writerow([level, agent_idx])

    return levels


def write_edgelist(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u, v in sorted((min(u, v), max(u, v)) for u, v in graph.edges()):
            f.write(f"{u} {v}\n")


def write_lfr_communities(path: Path, comm_id: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lfr_community", "agent_idx"])
        for agent_idx in sorted(comm_id):
            writer.writerow([comm_id[agent_idx], agent_idx])


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
        "median_degree": percentile(degrees, 50),
        "p90_degree": percentile(degrees, 90),
        "p99_degree": percentile(degrees, 99),
        "avg_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
        "largest_connected_component_ratio": 1.0,
        "max_degree": max(degrees),
    }


def lfr_metrics(graph: nx.Graph, communities: list[set[int]]) -> dict[str, object]:
    comm_id = lfr_community_id(graph, communities)
    internal_edges = 0
    external_edges = 0
    for u, v in graph.edges():
        if comm_id[u] == comm_id[v]:
            internal_edges += 1
        else:
            external_edges += 1

    sizes = [len(community) for community in communities]
    edge_count = graph.number_of_edges()
    return {
        "lfr_communities": len(communities),
        "lfr_modularity": modularity(graph, communities),
        "lfr_internal_edges": internal_edges,
        "lfr_external_edges": external_edges,
        "lfr_internal_edge_ratio": internal_edges / edge_count,
        "lfr_external_edge_ratio": external_edges / edge_count,
        "lfr_comm_size_min": min(sizes),
        "lfr_comm_size_q10": percentile(sizes, 10),
        "lfr_comm_size_q25": percentile(sizes, 25),
        "lfr_comm_size_median": percentile(sizes, 50),
        "lfr_comm_size_q75": percentile(sizes, 75),
        "lfr_comm_size_q90": percentile(sizes, 90),
        "lfr_comm_size_max": max(sizes),
    }


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


def support_pools(levels: list[float]) -> dict[str, list[int]]:
    ordered = sorted(enumerate(levels), key=lambda item: item[1], reverse=True)
    n = len(ordered)
    k = round(n * 0.2)
    center = n // 2
    median = (
        ordered[center][1]
        if n % 2 == 1
        else (ordered[center][1] + ordered[center - 1][1]) / 2
    )
    window = range(max(0, center - k), min(n, center + k))
    middle = sorted(window, key=lambda i: abs(ordered[i][1] - median))[:k]

    return {
        "top20": [agent_idx for agent_idx, _ in ordered[:k]],
        "middle20": [ordered[i][0] for i in middle],
        "bottom20": [agent_idx for agent_idx, _ in ordered[-k:]],
    }


def support_pool_metrics(
    *,
    graph: nx.Graph,
    levels: list[float],
    comm_id: dict[int, int],
) -> dict[str, object]:
    rows = {}
    for prefix, nodes in support_pools(levels).items():
        degrees = [graph.degree(node) for node in nodes]
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
        rows[f"{prefix}_communities_touched"] = len(counts)
        rows[f"{prefix}_largest_comm_count"] = counts[0]
        rows[f"{prefix}_largest_comm_ratio"] = counts[0] / len(nodes)
    return rows


def parse_levels(raw_levels: list[str] | None) -> list[MuLevel]:
    if not raw_levels:
        return DEFAULT_LEVELS

    levels = []
    for raw in raw_levels:
        try:
            name, value = raw.split(":", 1)
            levels.append(MuLevel(name=name, mu=float(value)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--level は name:mu 形式で指定してください: {raw}"
            ) from exc
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LFR benchmark graph を生成し、既存シミュレータ用の"
            "コミュニティ構造制御実験セットを用意します。"
        )
    )
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--tau1", type=float, default=2.5)
    parser.add_argument("--tau2", type=float, default=1.5)
    parser.add_argument("--average-degree", type=float, default=30.0)
    parser.add_argument("--max-degree", type=int, default=120)
    parser.add_argument("--min-community", type=int, default=40)
    parser.add_argument("--max-community", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--level",
        action="append",
        help="コミュニティ混合度を name:mu 形式で指定。例: --level strong:0.05",
    )
    parser.add_argument("--seed-base", type=int, default=20260602)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--max-retries", type=int, default=30)
    parser.add_argument("--output-subdir", default="lfr_community")
    parser.add_argument("--comm-communities", type=int, default=2)
    parser.add_argument("--comm-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.level)
    graph_root = NETWORK_ROOT / args.output_subdir
    summary_rows = []
    config_paths = []

    for level_index, level in enumerate(levels):
        for seed_index in args.seeds:
            network_name = f"lfr_{level.name}_seed{seed_index}"
            network_dir = graph_root / network_name
            config_path = NETWORK_ROOT / f"network-{network_name}.toml"
            generation_seed = (
                args.seed_base
                + level_index * 100000
                + seed_index * 1000
            )

            graph, actual_seed = connected_lfr_graph(
                num_nodes=args.num_nodes,
                tau1=args.tau1,
                tau2=args.tau2,
                mu=level.mu,
                average_degree=args.average_degree,
                max_degree=args.max_degree,
                min_community=args.min_community,
                max_community=args.max_community,
                seed=generation_seed,
                max_iters=args.max_iters,
                max_retries=args.max_retries,
            )

            communities = lfr_communities(graph)
            comm_id = lfr_community_id(graph, communities)
            graph_path = network_dir / "edgelist.txt"
            comm_path = network_dir / "comm.csv"
            lfr_comm_path = network_dir / "lfr_communities.csv"

            write_edgelist(graph, graph_path)
            write_lfr_communities(lfr_comm_path, comm_id)
            levels_for_nodes = write_comm_csv(
                graph=graph,
                output_path=comm_path,
                num_communities=args.comm_communities,
                community_index=args.comm_index,
            )
            write_network_config(config_path, network_name, args.output_subdir)
            config_paths.append(config_path)

            row = {
                "network": network_name,
                "community_level": level.name,
                "mu": level.mu,
                "tau1": args.tau1,
                "tau2": args.tau2,
                "target_average_degree": args.average_degree,
                "max_degree_parameter": args.max_degree,
                "min_community_parameter": args.min_community,
                "max_community_parameter": args.max_community,
                "requested_seed": generation_seed,
                "actual_seed": actual_seed,
                "config_path": str(config_path.relative_to(REPO_ROOT)),
                "graph_path": str(graph_path.relative_to(REPO_ROOT)),
                "comm_path": str(comm_path.relative_to(REPO_ROOT)),
                "lfr_community_path": str(lfr_comm_path.relative_to(REPO_ROOT)),
                **graph_metrics(graph),
                **lfr_metrics(graph, communities),
                **comm_metrics(levels_for_nodes),
                **support_pool_metrics(
                    graph=graph,
                    levels=levels_for_nodes,
                    comm_id=comm_id,
                ),
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
