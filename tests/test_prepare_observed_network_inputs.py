from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_observed_network_inputs import (
    allocation_to_level,
    parse_vote_date,
    read_wiki_rfa,
)


class PrepareObservedNetworkInputsTests(unittest.TestCase):
    def test_allocation_uses_the_original_percentage_scale(self) -> None:
        self.assertEqual(allocation_to_level(50.0), 0.5)
        self.assertEqual(allocation_to_level(0.5), 0.005)
        with self.assertRaises(ValueError):
            allocation_to_level(101.0)

    def test_vote_dates_and_period_filter(self) -> None:
        self.assertEqual(parse_vote_date("12:00, 4 January 2008").year, 2008)
        self.assertIsNone(parse_vote_date("unknown"))
        records = [
            ("A", "B", "1", "12:00, 3 January 2008"),
            ("B", "A", "0", "12:00, 4 January 2008"),
            ("B", "A", "-1", "12:00, 5 January 2008"),
            ("C", "C", "1", "12:00, 5 January 2008"),
            ("D", "A", "1", "bad date"),
        ]
        text = "\n\n".join(
            f"SRC:{src}\nTGT:{tgt}\nVOT:{vote}\nDAT:{date}\nTXT:sample"
            for src, tgt, vote, date in records
        ) + "\n\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wiki-RfA.txt.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write(text)
            graph, names, counts = read_wiki_rfa(
                source, expected_records=5, expected_size=(2, 1)
            )
        self.assertEqual(names, ["A", "B"])
        self.assertEqual(list(graph.edges()), [(1, 0)])
        self.assertEqual(counts["after_cutoff"], 3)
        self.assertEqual(counts["before_cutoff"], 1)
        self.assertEqual(counts["unparseable_date"], 1)
        self.assertEqual(counts["self_votes"], 1)


if __name__ == "__main__":
    unittest.main()
