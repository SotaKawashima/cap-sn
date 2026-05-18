#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    print("Python 3.11+ が必要です。", file=sys.stderr)
    raise

try:
    import networkx as nx
except ModuleNotFoundError:
    print("networkx が必要です。例: pip install networkx", file=sys.stderr)
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    REPO_ROOT / "v2/test_2/network/network-ba1000.toml",
    REPO_ROOT / "v2/test_2/network/network-ba1000-seed2.toml",
    REPO_ROOT / "v2/test_2/network/network-ba1000-seed3.toml",
    REPO_ROOT / "v2/test_2/network/network-ba1000-seed4.toml",
    REPO_ROOT / "v2/test_2/network/network-facebook.toml",
    REPO_ROOT / "v2/test_2/network/network-wiki-vote.toml",
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


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def safe_assortativity(graph) -> float:
    try:
        return safe_float(nx.degree_assortativity_coefficient(graph))
    except Exception:
        return math.nan


def read_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        conf = tomllib.load(f)

    base = config_path.parent / conf["path"]
    graph_path = (base / conf["graph"]).resolve()

    return {
        "config_path": config_path.resolve(),
        "name": config_path.stem.replace("network-", ""),
        "graph_path": graph_path,
        "directed": bool(conf["directed"]),
        "transposed": bool(conf["transposed"]),
    }


def read_graph(conf: dict):
    if conf["directed"]:
        graph = nx.read_edgelist(
            conf["graph_path"],
            comments="#",
            nodetype=int,
            create_using=nx.DiGraph,
        )
        if conf["transposed"]:
            graph = graph.reverse(copy=True)
    else:
        graph = nx.read_edgelist(
            conf["graph_path"],
            comments="#",
            nodetype=int,
            create_using=nx.Graph,
        )
    return graph


def summarize_undirected(conf: dict, graph) -> dict:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    components = [len(c) for c in nx.connected_components(graph)]

    return {
        "network": conf["name"],
        "config_path": str(conf["config_path"].relative_to(REPO_ROOT)),
        "graph_path": str(conf["graph_path"].relative_to(REPO_ROOT)),
        "directed": False,
        "transposed": False,
        "num_nodes": n,
        "num_edges": m,
        "density": nx.density(graph),
        "avg_degree": statistics.fmean(degrees),
        "median_degree": percentile(degrees, 50),
        "p90_degree": percentile(degrees, 90),
        "p99_degree": percentile(degrees, 99),
        "max_degree": max(degrees),
        "avg_in_degree": "",
        "median_in_degree": "",
        "p90_in_degree": "",
        "p99_in_degree": "",
        "max_in_degree": "",
        "avg_out_degree": "",
        "median_out_degree": "",
        "p90_out_degree": "",
        "p99_out_degree": "",
        "max_out_degree": "",
        "avg_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
        "assortativity": safe_assortativity(graph),
        "reciprocity": "",
        "num_connected_components": len(components),
        "largest_connected_component_ratio": max(components) / n,
        "num_weak_components": "",
        "largest_weak_component_ratio": "",
        "num_strong_components": "",
        "largest_strong_component_ratio": "",
    }


def summarize_directed(conf: dict, graph) -> dict:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    in_degrees = [degree for _, degree in graph.in_degree()]
    out_degrees = [degree for _, degree in graph.out_degree()]
    total_degrees = [
        graph.in_degree(node) + graph.out_degree(node)
        for node in graph.nodes()
    ]
    weak_components = [len(c) for c in nx.weakly_connected_components(graph)]
    strong_components = [len(c) for c in nx.strongly_connected_components(graph)]
    undirected = graph.to_undirected()

    return {
        "network": conf["name"],
        "config_path": str(conf["config_path"].relative_to(REPO_ROOT)),
        "graph_path": str(conf["graph_path"].relative_to(REPO_ROOT)),
        "directed": True,
        "transposed": conf["transposed"],
        "num_nodes": n,
        "num_edges": m,
        "density": nx.density(graph),
        "avg_degree": statistics.fmean(total_degrees),
        "median_degree": percentile(total_degrees, 50),
        "p90_degree": percentile(total_degrees, 90),
        "p99_degree": percentile(total_degrees, 99),
        "max_degree": max(total_degrees),
        "avg_in_degree": statistics.fmean(in_degrees),
        "median_in_degree": percentile(in_degrees, 50),
        "p90_in_degree": percentile(in_degrees, 90),
        "p99_in_degree": percentile(in_degrees, 99),
        "max_in_degree": max(in_degrees),
        "avg_out_degree": statistics.fmean(out_degrees),
        "median_out_degree": percentile(out_degrees, 50),
        "p90_out_degree": percentile(out_degrees, 90),
        "p99_out_degree": percentile(out_degrees, 99),
        "max_out_degree": max(out_degrees),
        "avg_clustering": nx.average_clustering(undirected),
        "transitivity": nx.transitivity(undirected),
        "assortativity": safe_assortativity(undirected),
        "reciprocity": safe_float(nx.reciprocity(graph)),
        "num_connected_components": "",
        "largest_connected_component_ratio": "",
        "num_weak_components": len(weak_components),
        "largest_weak_component_ratio": max(weak_components) / n,
        "num_strong_components": len(strong_components),
        "largest_strong_component_ratio": max(strong_components) / n,
    }


def summarize(conf_path: Path) -> dict:
    conf = read_config(conf_path)
    graph = read_graph(conf)
    if conf["directed"]:
        return summarize_directed(conf, graph)
    return summarize_undirected(conf, graph)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value == "":
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


def write_markdown(rows: list[dict], path: Path) -> None:
    cols = [
        "network",
        "directed",
        "transposed",
        "num_nodes",
        "num_edges",
        "density",
        "avg_degree",
        "median_degree",
        "p90_degree",
        "p99_degree",
        "max_degree",
        "avg_clustering",
        "transitivity",
        "reciprocity",
        "largest_connected_component_ratio",
        "largest_weak_component_ratio",
        "largest_strong_component_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Network Metrics\n\n")
        f.write("この表は `scripts/calc_network_metrics.py` で再計算したものです。\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row[c]) for c in cols) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate structural metrics for network config files."
    )
    parser.add_argument(
        "--output-dir",
        default="network_metrics",
        help="Directory to write summary.csv and summary.md.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="Network config TOML files. Defaults to BA1000 variants, Facebook, and Wiki-vote.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.configs or DEFAULT_CONFIGS
    rows = [summarize(path.resolve()) for path in configs]

    output_dir = REPO_ROOT / args.output_dir
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"saved: {csv_path.relative_to(REPO_ROOT)}")
    print(f"saved: {md_path.relative_to(REPO_ROOT)}")
    print()
    for row in rows:
        print(
            f"{row['network']}: "
            f"nodes={row['num_nodes']}, edges={row['num_edges']}, "
            f"avg_degree={fmt(row['avg_degree'])}, "
            f"clustering={fmt(row['avg_clustering'])}"
        )


if __name__ == "__main__":
    main()
