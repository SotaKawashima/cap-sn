"""Analyze a completed Stage 9 structural confirmation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.structure_confirmation_analysis import (
    run_structure_confirmation_analysis,
)
from experiment_runtime import REPO_ROOT, validate_safe_name
from run_stage9_structure_confirmation import build_specs, load_protocol


DEFAULT_PROTOCOL = (
    REPO_ROOT / "experiment_protocols" / "stage9_structure_confirmation_v1.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and analyze the Stage 9 structure confirmation."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--analysis-id", default="structure_confirmation_analysis_v01"
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    analysis_id = validate_safe_name(args.analysis_id, "analysis_id")
    experiment_root = args.experiment_root
    if not experiment_root.is_absolute():
        experiment_root = REPO_ROOT / experiment_root
    experiment_root = experiment_root.resolve()
    protocol = load_protocol(args.protocol)
    result = run_structure_confirmation_analysis(
        protocol,
        expected_specs=build_specs(protocol),
        experiment_root=experiment_root,
        output_root=experiment_root / analysis_id,
        repetitions=args.bootstrap_repetitions,
    )
    print(f"Completed Stage 9 structure confirmation analysis: {result['output_root']}")


if __name__ == "__main__":
    main()
