"""Run one fixed condition on a repository-local custom network config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.optimization_metrics import MetricValidationError
from experiment_runtime import (
    REPO_ROOT,
    SUMMER_EXPERIMENT_ROOT,
    ExperimentConfigurationError,
    NetworkSpec,
    SimulationExecutionError,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_safe_name,
)
from run_fixed_condition import run_fixed_condition_for_network


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed condition on a repository-local network config."
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--purpose", default="custom_fixed_condition")
    parser.add_argument("--network-id", required=True)
    parser.add_argument("--network-config", type=Path, required=True)
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--certainty", type=float, default=None)
    parser.add_argument("--effectiveness", type=float, default=None)
    parser.add_argument("--intervention-opinion-csv", type=Path, default=None)
    parser.add_argument("--no-intervention", action="store_true")
    parser.add_argument("--simulator-seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--raw-level", choices=["pop", "info_pop", "all"], default="pop"
    )
    parser.add_argument(
        "--output-root", type=Path, default=SUMMER_EXPERIMENT_ROOT
    )
    return parser.parse_args(argv)


def resolve_custom_network(args: argparse.Namespace) -> NetworkSpec:
    network_id = validate_safe_name(args.network_id, "network_id")
    num_agents = validate_positive_integer(args.num_agents, "num_agents")
    if args.network_seed is not None:
        validate_nonnegative_integer(args.network_seed, "network_seed")

    config_path = args.network_config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_path = config_path.resolve()
    try:
        config_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ExperimentConfigurationError(
            "network_config must be inside the repository"
        ) from exc
    if not config_path.is_file():
        raise ExperimentConfigurationError(
            f"network config is missing: {config_path}"
        )
    if config_path.suffix != ".toml":
        raise ExperimentConfigurationError("network_config must be a TOML file")
    return NetworkSpec(
        id=network_id,
        config_path=config_path,
        num_agents=num_agents,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        network = resolve_custom_network(args)
        run_dir = run_fixed_condition_for_network(
            args,
            network=network,
            network_seed=args.network_seed,
        )
    except (
        ExperimentConfigurationError,
        SimulationExecutionError,
        MetricValidationError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Completed custom fixed-condition run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
