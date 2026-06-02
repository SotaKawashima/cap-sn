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


@dataclass(frozen=True)
class PoolTarget:
    name: str
    top_ratio: float | None
    middle_ratio: float | None
    bottom_ratio: float | None
    top_touched: int | None = None
    middle_touched: int | None = None
    bottom_touched: int | None = None


DEFAULT_METHODS = [
    PoolTarget("original", None, None, None),
    PoolTarget("random", None, None, None),
    PoolTarget("half_facebook", 0.45, 0.29, 0.41, 5, 9, 6),
    PoolTarget("facebook_like", 0.645, 0.381, 0.526, 3, 9, 5),
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


def community_sets(comm_id: dict[int, int]) -> list[set[int]]:
    groups: dict[int, set[int]] = defaultdict(set)
    for node, cid in comm_id.items():
        groups[cid].add(node)
    return [groups[cid] for cid in sorted(groups)]


def read_graph(path: Path) -> nx.Graph:
    graph = nx.read_edgelist(path, nodetype=int)
    if graph.number_of_nodes() == 0:
        raise ValueError(f"empty graph: {path}")
    return graph


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


def choose_anchor_communities(
    comm_id: dict[int, int],
    target_counts: dict[str, int],
) -> dict[str, int]:
    counts = Counter(comm_id.values())
    sorted_communities = [cid for cid, _ in counts.most_common()]
    anchors = {}
    used = set()
    for pool_name in ["top20", "bottom20", "middle20"]:
        target_count = target_counts[pool_name]
        for cid in sorted_communities:
            if cid in used:
                continue
            if counts[cid] >= target_count:
                anchors[pool_name] = cid
                used.add(cid)
                break
        if pool_name not in anchors:
            raise ValueError(
                f"no LFR community can supply {target_count} nodes for {pool_name}"
            )
    return anchors


def select_pool(
    *,
    pool_size: int,
    target_count: int,
    target_touched: int,
    anchor_cid: int,
    protected_cids: set[int],
    comm_to_nodes: dict[int, list[int]],
    used_nodes: set[int],
    rng: random.Random,
) -> list[int]:
    anchor_available = [node for node in comm_to_nodes[anchor_cid] if node not in used_nodes]
    if len(anchor_available) < target_count:
        raise ValueError(
            f"anchor community {anchor_cid} has only {len(anchor_available)} "
            f"available nodes, needs {target_count}"
        )

    selected = rng.sample(anchor_available, target_count)
    selected_set = set(selected)
    counts = Counter({anchor_cid: target_count})
    fill_needed = pool_size - target_count
    cap = max(0, target_count - 1)

    candidates_by_comm = {}
    candidate_communities = []
    for cid, nodes in sorted(
        comm_to_nodes.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        if cid == anchor_cid:
            continue
        if cid in protected_cids:
            continue
        candidates = [
            node
            for node in nodes
            if node not in used_nodes and node not in selected_set
        ]
        if candidates:
            rng.shuffle(candidates)
            candidates_by_comm[cid] = candidates
            candidate_communities.append(cid)

    if target_touched < 1:
        raise ValueError("target_touched must be at least 1")
    community_order = candidate_communities[: max(0, target_touched - 1)]
    capacity = sum(min(len(candidates_by_comm[cid]), cap) for cid in community_order)
    if capacity < fill_needed:
        raise RuntimeError(
            f"not enough capacity to fill {pool_size} nodes with "
            f"target_touched={target_touched}; capacity={capacity}, "
            f"remaining={fill_needed}"
        )

    while fill_needed > 0:
        progressed = False
        for cid in community_order:
            if fill_needed == 0:
                break
            if counts[cid] >= cap:
                continue
            candidates = candidates_by_comm[cid]
            if not candidates:
                continue
            node = candidates.pop()
            selected.append(node)
            selected_set.add(node)
            counts[cid] += 1
            fill_needed -= 1
            progressed = True
        if not progressed:
            raise RuntimeError(
                f"could not fill pool anchored at community {anchor_cid}; "
                f"remaining={fill_needed}"
            )

    return selected


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


def targeted_levels(
    *,
    num_nodes: int,
    comm_id: dict[int, int],
    target: PoolTarget,
    rng: random.Random,
) -> tuple[list[float], dict[str, object]]:
    pool_size = round(num_nodes * 0.2)
    target_counts = {
        "top20": round(pool_size * target.top_ratio),
        "middle20": round(pool_size * target.middle_ratio),
        "bottom20": round(pool_size * target.bottom_ratio),
    }
    target_touched = {
        "top20": target.top_touched,
        "middle20": target.middle_touched,
        "bottom20": target.bottom_touched,
    }
    comm_to_nodes: dict[int, list[int]] = defaultdict(list)
    for node, cid in comm_id.items():
        comm_to_nodes[cid].append(node)
    for nodes in comm_to_nodes.values():
        nodes.sort()
    for pool_name, value in list(target_touched.items()):
        if value is None:
            target_touched[pool_name] = len(comm_to_nodes)

    anchors = choose_anchor_communities(comm_id, target_counts)
    protected_cids = set(anchors.values())
    used_nodes: set[int] = set()
    pools = {}
    for pool_name in ["top20", "bottom20", "middle20"]:
        nodes = select_pool(
            pool_size=pool_size,
            target_count=target_counts[pool_name],
            target_touched=target_touched[pool_name],
            anchor_cid=anchors[pool_name],
            protected_cids=protected_cids,
            comm_to_nodes=comm_to_nodes,
            used_nodes=used_nodes,
            rng=rng,
        )
        pools[pool_name] = nodes
        used_nodes.update(nodes)

    filler = [node for node in range(num_nodes) if node not in used_nodes]
    rng.shuffle(filler)
    high_filler = filler[:pool_size]
    low_filler = filler[pool_size:]

    ranked_nodes = []
    for part in [
        pools["top20"],
        high_filler,
        pools["middle20"],
        low_filler,
        pools["bottom20"],
    ]:
        rng.shuffle(part)
        ranked_nodes.extend(part)

    metadata = {
        "target_top20_ratio": target.top_ratio,
        "target_middle20_ratio": target.middle_ratio,
        "target_bottom20_ratio": target.bottom_ratio,
        "target_top20_count": target_counts["top20"],
        "target_middle20_count": target_counts["middle20"],
        "target_bottom20_count": target_counts["bottom20"],
        "target_top20_communities_touched": target_touched["top20"],
        "target_middle20_communities_touched": target_touched["middle20"],
        "target_bottom20_communities_touched": target_touched["bottom20"],
        "anchor_top20_community": anchors["top20"],
        "anchor_middle20_community": anchors["middle20"],
        "anchor_bottom20_community": anchors["bottom20"],
    }
    return levels_from_ranked_nodes(ranked_nodes, num_nodes), metadata


def parse_methods(raw_methods: list[str] | None) -> list[PoolTarget]:
    if not raw_methods:
        return DEFAULT_METHODS
    default_by_name = {method.name: method for method in DEFAULT_METHODS}
    methods = []
    for raw in raw_methods:
        if raw in default_by_name:
            methods.append(default_by_name[raw])
            continue
        try:
            parts = raw.split(":")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--method must be a known name or name:top:middle:bottom[:top_touched:middle_touched:bottom_touched]"
            ) from exc
        if len(parts) not in {4, 7}:
            raise argparse.ArgumentTypeError(
                "--method must be a known name or name:top:middle:bottom[:top_touched:middle_touched:bottom_touched]"
            )
        name, top, middle, bottom = parts[:4]
        touched = [int(value) for value in parts[4:]] if len(parts) == 7 else [None, None, None]
        methods.append(
            PoolTarget(
                name,
                float(top),
                float(middle),
                float(bottom),
                touched[0],
                touched[1],
                touched[2],
            )
        )
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create LFR strong graph variants whose support-level candidate pools "
            "are concentrated like Facebook."
        )
    )
    parser.add_argument("--source-subdir", default="lfr_community")
    parser.add_argument("--source-level", default="strong")
    parser.add_argument("--output-subdir", default="lfr_facebook_pool")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed-base", type=int, default=20260602)
    parser.add_argument(
        "--method",
        action="append",
        help=(
            "Pool method. Use original, random, half_facebook, facebook_like, "
            "or name:top:middle:bottom."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.method)
    output_root = NETWORK_ROOT / args.output_subdir
    source_root = NETWORK_ROOT / args.source_subdir
    rows = []
    config_paths = []

    for seed_index in args.seeds:
        source_network = f"lfr_{args.source_level}_seed{seed_index}"
        source_dir = source_root / source_network
        source_graph_path = source_dir / "edgelist.txt"
        source_comm_path = source_dir / "comm.csv"
        source_lfr_path = source_dir / "lfr_communities.csv"
        if not source_graph_path.exists():
            raise FileNotFoundError(source_graph_path)
        if not source_comm_path.exists():
            raise FileNotFoundError(source_comm_path)
        if not source_lfr_path.exists():
            raise FileNotFoundError(source_lfr_path)

        graph = read_graph(source_graph_path)
        num_nodes = graph.number_of_nodes()
        comm_id = read_lfr_communities(source_lfr_path)
        original_levels = read_levels(source_comm_path, num_nodes)

        for method_index, method in enumerate(methods):
            network_name = f"lfrpool_{args.source_level}_{method.name}_seed{seed_index}"
            network_dir = output_root / network_name
            config_path = NETWORK_ROOT / f"network-{network_name}.toml"
            comm_path = network_dir / "comm.csv"
            method_seed = (
                args.seed_base
                + seed_index * 1000
                + method_index * 100
            )
            rng = random.Random(method_seed)

            network_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_graph_path, network_dir / "edgelist.txt")
            shutil.copyfile(source_lfr_path, network_dir / "lfr_communities.csv")

            method_metadata = {
                "target_top20_ratio": None,
                "target_middle20_ratio": None,
                "target_bottom20_ratio": None,
                "target_top20_count": None,
                "target_middle20_count": None,
                "target_bottom20_count": None,
                "target_top20_communities_touched": None,
                "target_middle20_communities_touched": None,
                "target_bottom20_communities_touched": None,
                "anchor_top20_community": None,
                "anchor_middle20_community": None,
                "anchor_bottom20_community": None,
            }
            if method.name == "original":
                levels = original_levels
                shutil.copyfile(source_comm_path, comm_path)
            elif method.name == "random":
                levels = random_levels(num_nodes, rng)
                write_comm_csv(comm_path, levels)
            else:
                levels, method_metadata = targeted_levels(
                    num_nodes=num_nodes,
                    comm_id=comm_id,
                    target=method,
                    rng=rng,
                )
                write_comm_csv(comm_path, levels)

            write_network_config(config_path, network_name, args.output_subdir)
            config_paths.append(config_path)

            rows.append(
                {
                    "network": network_name,
                    "source_network": source_network,
                    "structure_level": args.source_level,
                    "seed_index": seed_index,
                    "comm_method": method.name,
                    "method_seed": method_seed if method.name != "original" else None,
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
                    ),
                }
            )

    summary_path = output_root / "generation_summary.csv"
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
