#!/usr/bin/env python3
"""Prepare observed Facebook and later-period wiki-RfA simulator inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from datetime import datetime
from gzip import open as gzip_open
from pathlib import Path
from zipfile import ZipFile

import networkx as nx
import numpy as np
from scipy.io import mmread


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments/summer_2026/observed_network_inputs/20260922_v01"
)
FACEBOOK = {
    "Brandeis99": (3898, 137567, "592ba6d2491c2fc61fbe3d9a4aec6ed8d3ccb8a4470febb7350263fd47393926"),
    "Bucknell39": (3826, 158864, "27669f7c55307f920a9bc16ff0028317174e0b324dab74e6981eb829dd814ecf"),
    "Rice31": (4087, 184828, "2ee95e547b536ad28914869ff7179203a52423ca039db9bd49fe8b1f85742278"),
}
WIKI_SHA256 = "88d53196fb2564a2e20286dbba818832f718cc352bb181a2101d23d2556f0862"
WIKI_CUTOFF = datetime(2008, 1, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Source SHA-256 mismatch: {path}: {actual}")


def read_facebook_zip(path: Path, name: str) -> nx.Graph:
    member = f"socfb-{name}.mtx"
    with ZipFile(path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"Missing {member} in {path}")
        with archive.open(member) as handle:
            matrix = mmread(handle).tocsr()
    if matrix.shape[0] != matrix.shape[1] or (matrix != matrix.T).nnz:
        raise ValueError(f"Expected a square, symmetric graph: {path}")
    graph = nx.from_scipy_sparse_array(matrix)
    expected_nodes, expected_edges, _ = FACEBOOK[name]
    if (len(graph), graph.number_of_edges()) != (expected_nodes, expected_edges):
        raise ValueError(f"Unexpected Facebook graph size: {path}")
    if nx.number_of_isolates(graph) or nx.number_of_selfloops(graph):
        raise ValueError(f"Isolates or self-loops in {path}")
    return graph


def parse_vote_date(value: str) -> datetime | None:
    for fmt in ("%H:%M, %d %B %Y", "%H:%M, %d %b %Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def read_wiki_rfa(
    path: Path,
    *,
    expected_records: int = 198275,
    expected_size: tuple[int, int] = (5709, 79476),
) -> tuple[nx.DiGraph, list[str], dict[str, int]]:
    named_edges: set[tuple[str, str]] = set()
    counts = {"records": 0, "unparseable_date": 0, "before_cutoff": 0,
              "after_cutoff": 0, "self_votes": 0}
    record: dict[str, str] = {}

    def accept() -> None:
        if not record:
            return
        counts["records"] += 1
        if not {"SRC", "TGT", "VOT", "DAT"}.issubset(record):
            raise ValueError(f"Missing wiki-RfA fields in record {counts['records']}")
        if record["VOT"] not in {"-1", "0", "1"}:
            raise ValueError(f"Unknown vote in record {counts['records']}")
        date = parse_vote_date(record["DAT"])
        if date is None:
            counts["unparseable_date"] += 1
        elif date < WIKI_CUTOFF:
            counts["before_cutoff"] += 1
        else:
            counts["after_cutoff"] += 1
            if record["SRC"] == record["TGT"]:
                counts["self_votes"] += 1
            else:
                named_edges.add((record["SRC"], record["TGT"]))

    with gzip_open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                accept()
                record = {}
                continue
            key, separator, value = line.partition(":")
            if separator and key in {"SRC", "TGT", "VOT", "DAT"}:
                record[key] = value
    accept()

    names = sorted({node for edge in named_edges for node in edge})
    ids = {name: index for index, name in enumerate(names)}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(names)))
    graph.add_edges_from((ids[src], ids[tgt]) for src, tgt in sorted(named_edges))
    if counts["records"] != expected_records or (len(graph), graph.number_of_edges()) != expected_size:
        raise ValueError("wiki-RfA counts differ from the audited source")
    return graph, names, counts


def allocation_to_level(raw: float) -> float:
    level = float(raw) / 100.0
    if not math.isfinite(level) or not 0 <= level <= 1:
        raise ValueError(f"Invalid support level: {level}")
    return level


def write_comm_csv(graph: nx.Graph, path: Path, seed: int) -> dict[str, float]:
    from cdlib import algorithms

    random.seed(seed)
    np.random.seed(seed)
    communities = algorithms.principled_clustering(graph, 2)
    values = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("level", "agent_idx"))
        for index in range(len(graph)):
            # graph_comm.ipynb divides every allocation percentage by 100.
            level = allocation_to_level(communities.allocation_matrix[index][0])
            writer.writerow((level, index))
            values.append(level)
    return {"minimum": min(values), "maximum": max(values),
            "mean": sum(values) / len(values)}


def prepare_one(
    root: Path,
    name: str,
    graph: nx.Graph,
    source_ids: list[str],
    source_path: Path,
    source_url: str,
    seed: int,
    extra: dict | None = None,
) -> dict:
    if set(graph) != set(range(len(graph))):
        raise ValueError(f"Non-contiguous node IDs: {name}")
    directory = root / name
    directory.mkdir()
    edge_path = directory / "edgelist.txt"
    with edge_path.open("w", encoding="utf-8") as handle:
        for u, v in sorted(graph.edges()):
            handle.write(f"{u} {v}\n")
    mapping_path = directory / "node_ids.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("agent_idx", "source_node_id"))
        writer.writerows(enumerate(source_ids))
    comm_path = directory / "comm.csv"
    levels = write_comm_csv(graph, comm_path, seed)
    directed = graph.is_directed()
    (directory / "network.toml").write_text(
        'path = "."\n'
        'graph = "edgelist.txt"\n'
        f'directed = {str(directed).lower()}\n'
        f'transposed = {str(directed).lower()}\n'
        'community = "comm.csv"\n', encoding="utf-8",
    )
    reread = nx.read_edgelist(edge_path, nodetype=int,
                              create_using=nx.DiGraph if directed else nx.Graph)
    def edge_set(g: nx.Graph) -> set[tuple[int, int]]:
        if directed:
            return set(g.edges())
        return {(min(u, v), max(u, v)) for u, v in g.edges()}

    if set(reread) != set(graph) or edge_set(reread) != edge_set(graph):
        raise ValueError(f"Edge-list round-trip changed the graph: {name}")
    components = (nx.weakly_connected_components(graph) if directed
                  else nx.connected_components(graph))
    largest_component = max(map(len, components))
    summary = {
        "name": name,
        "source_url": source_url,
        "source_archive_sha256": sha256(source_path),
        "nodes": len(graph),
        "edges": graph.number_of_edges(),
        "directed": directed,
        "transposed_in_simulator": directed,
        "largest_weak_or_connected_component": largest_component,
        "comm_method": "cdlib.principled_clustering(graph, 2), allocation_matrix[node][0] / 100",
        "comm_seed": seed,
        "support_levels": levels,
        "files_sha256": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()},
    }
    if directed:
        summary["largest_strong_component"] = max(
            map(len, nx.strongly_connected_components(graph))
        )
    if extra:
        summary.update(extra)
    (directory / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def prepare(facebook_zip_dir: Path, wiki_rfa: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    for name, (_, _, digest) in FACEBOOK.items():
        check_source(facebook_zip_dir / f"socfb-{name}.zip", digest)
    check_source(wiki_rfa, WIKI_SHA256)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="observed_inputs_", dir=output_root.parent) as temporary:
        root = Path(temporary)
        summaries = []
        for index, name in enumerate(FACEBOOK):
            source = facebook_zip_dir / f"socfb-{name}.zip"
            graph = read_facebook_zip(source, name)
            summaries.append(prepare_one(
                root, f"facebook_{name.lower()}", graph,
                [str(node + 1) for node in range(len(graph))], source,
                f"https://networkrepository.com/socfb-{name}.php", 20260922 + index,
                {"source_license": "CC BY-SA; see https://networkrepository.com/policy.php"},
            ))
        wiki, names, vote_counts = read_wiki_rfa(wiki_rfa)
        summaries.append(prepare_one(
            root, "wiki_rfa_post2008", wiki, names, wiki_rfa,
            "https://snap.stanford.edu/data/wiki-RfA.html", 20260925,
            {"cutoff_inclusive": WIKI_CUTOFF.date().isoformat(),
             "vote_handling": "all vote signs; duplicate ordered pairs and self-votes removed",
             "record_counts": vote_counts},
        ))
        result = {
            "status": "prepared_not_simulated",
            "networks": summaries,
            "ba1000_reused_configs": [
                f"v2/test_2/network/network-ba1000-seed{seed}.toml"
                for seed in (2, 3, 4)
            ],
        }
        (root / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        root.rename(output_root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facebook-zip-dir", type=Path, required=True)
    parser.add_argument("--wiki-rfa", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.facebook_zip_dir, args.wiki_rfa, args.output_root)
    print(json.dumps({"output_root": str(args.output_root),
                      "networks": [(r["name"], r["nodes"], r["edges"])
                                   for r in result["networks"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
