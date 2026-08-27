"""Reaggregate the legacy structure experiments with the current objective."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.structure_reanalysis import load_protocol, run_structure_reanalysis
from experiment_runtime import REPO_ROOT, validate_safe_name


DEFAULT_PROTOCOL = (
    REPO_ROOT / "experiment_protocols" / "stage9_structure_reanalysis_v1.json"
)
DEFAULT_OUTPUT_PARENT = (
    REPO_ROOT / "experiments" / "summer_2026" / "stage9_structure_reanalysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and reaggregate the existing structure experiments."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--analysis-id", default="existing_structure_reanalysis_v01"
    )
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    output_parent = args.output_parent
    if not output_parent.is_absolute():
        output_parent = REPO_ROOT / output_parent
    protocol = load_protocol(args.protocol)
    result = run_structure_reanalysis(
        protocol,
        repo_root=REPO_ROOT,
        output_root=output_parent / analysis_id,
        repetitions=args.bootstrap_repetitions,
    )
    print(f"Completed Stage 9 structure reanalysis: {result['output_root']}")


if __name__ == "__main__":
    main()
