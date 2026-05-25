#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

try:
    import networkx as nx
except ModuleNotFoundError:
    print("networkx が必要です。例: pip install networkx", file=sys.stderr)
    raise

def read_graph(path: Path, *, directed: bool) -> nx.Graph:
    create_using = nx.DiGraph if directed else nx.Graph
    graph = nx.read_edgelist(path, nodetype=int, create_using=create_using)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


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
) -> None:
    if community_index < 0 or community_index >= num_communities:
        raise ValueError(
            f"community_index must be in [0, {num_communities - 1}], "
            f"got {community_index}"
        )

    try:
        from cdlib import algorithms
    except ModuleNotFoundError:
        print("cdlib が必要です。例: pip install cdlib", file=sys.stderr)
        raise

    comms = algorithms.principled_clustering(graph, num_communities)
    nodes = sorted(graph.nodes())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "agent_idx"])
        for agent_idx, node in enumerate(nodes):
            writer.writerow([allocation_value(comms, node, community_index), agent_idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "graph_comm.ipynb と同じく cdlib.algorithms.principled_clustering "
            "から comm.csv(level,agent_idx) を生成します。"
        )
    )
    parser.add_argument("--graph", type=Path, required=True, help="入力 edgelist")
    parser.add_argument("--output", type=Path, required=True, help="出力 comm.csv")
    parser.add_argument("--directed", action="store_true", help="有向グラフとして読む")
    parser.add_argument("--communities", type=int, default=2)
    parser.add_argument("--community-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = read_graph(args.graph, directed=args.directed)
    write_comm_csv(
        graph=graph,
        output_path=args.output,
        num_communities=args.communities,
        community_index=args.community_index,
    )
    print(f"Wrote {args.output} for {graph.number_of_nodes()} nodes")


if __name__ == "__main__":
    main()
