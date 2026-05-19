#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
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
DEFAULT_COMMUNITY = NETWORK_ROOT / "ba/ba1000/comm.csv"


@dataclass(frozen=True)
class DegreeLevel:
    name: str
    attachment_edges: int


DEFAULT_LEVELS = [
    DegreeLevel("deg20", 10),
    DegreeLevel("deg30", 15),
    DegreeLevel("deg40", 20),
]


def read_community_size(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


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


def write_network_config(path: Path, network_dir_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'path = "./powerlaw_degree/{network_dir_name}/"',
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


def parse_levels(raw_levels: list[str] | None) -> list[DegreeLevel]:
    if not raw_levels:
        return DEFAULT_LEVELS

    levels = []
    for raw in raw_levels:
        try:
            name, value = raw.split(":", 1)
            levels.append(DegreeLevel(name=name, attachment_edges=int(value)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--level は name:m 形式で指定してください: {raw}"
            ) from exc
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "平均次数を変えたPowerlaw cluster graphを生成し、"
            "既存シミュレータ用のネットワーク実験セットを用意します。"
        )
    )
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--level",
        action="append",
        help="平均次数水準を name:m 形式で指定。例: --level deg40:20",
    )
    parser.add_argument(
        "--triad-probability",
        type=float,
        default=0.0,
        help="NetworkX powerlaw_cluster_graph の p。案Aでは低クラスタ条件として0.0を使う。",
    )
    parser.add_argument("--seed-base", type=int, default=20260519)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--output-subdir", default="powerlaw_degree")
    parser.add_argument(
        "--community",
        type=Path,
        default=DEFAULT_COMMUNITY,
        help="コピーする comm.csv。デフォルトは BA1000 のもの。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.level)
    community_path = args.community.resolve()

    if not community_path.exists():
        raise FileNotFoundError(community_path)

    community_size = read_community_size(community_path)
    if community_size != args.num_nodes:
        raise ValueError(
            f"community size mismatch: comm.csv has {community_size} rows, "
            f"but num_nodes={args.num_nodes}"
        )

    graph_root = NETWORK_ROOT / args.output_subdir
    summary_rows = []
    config_paths = []

    for level in levels:
        for seed_index in args.seeds:
            network_name = f"pld_{level.name}_seed{seed_index}"
            network_dir = graph_root / network_name
            config_path = NETWORK_ROOT / f"network-{network_name}.toml"
            generation_seed = (
                args.seed_base
                + seed_index * 1000
                + level.attachment_edges * 10
                + int(args.triad_probability * 100)
            )

            graph, actual_seed = connected_powerlaw_cluster_graph(
                num_nodes=args.num_nodes,
                attachment_edges=level.attachment_edges,
                triad_probability=args.triad_probability,
                seed=generation_seed,
                max_retries=args.max_retries,
            )

            write_edgelist(graph, network_dir / "edgelist.txt")
            shutil.copyfile(community_path, network_dir / "comm.csv")
            write_network_config(config_path, network_name)
            config_paths.append(config_path)

            row = {
                "network": network_name,
                "degree_level": level.name,
                "attachment_edges_m": level.attachment_edges,
                "target_avg_degree": level.attachment_edges * 2,
                "triad_probability": args.triad_probability,
                "requested_seed": generation_seed,
                "actual_seed": actual_seed,
                "config_path": str(config_path.relative_to(REPO_ROOT)),
                "graph_path": str((network_dir / "edgelist.txt").relative_to(REPO_ROOT)),
                **graph_metrics(graph),
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
