"""CLI entrypoint for cross-benchmark evaluation and reporting."""

from __future__ import annotations

import argparse
import json

from gs_dronegym.baselines import load_behavior_cloning_policy
from gs_dronegym.benchmarks import make_benchmark
from gs_dronegym.cli._scene import normalize_scene_arg


def build_parser() -> argparse.ArgumentParser:
    """Create the evaluation CLI parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Evaluate a policy across supported benchmarks.")
    parser.add_argument(
        "--benchmark",
        choices=["gs_dronegym", "drone", "libero", "lerobot"],
        default="gs_dronegym",
        help="Benchmark adapter name.",
    )
    parser.add_argument("--policy", default=None, help="Optional BC checkpoint path.")
    parser.add_argument(
        "--source",
        default=None,
        help="Dataset source path for dataset-based benchmarks.",
    )
    parser.add_argument("--env-id", default="PointNav-v0", help="Drone environment ID.")
    parser.add_argument(
        "--scene",
        default=None,
        help="Optional drone scene handle. Use None for the mock renderer.",
    )
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes to evaluate.")
    parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    parser.add_argument("--device", default="cpu", help="Device for loading policy checkpoints.")
    parser.add_argument(
        "--include-raw-results",
        action="store_true",
        help=(
            "Include per-step episode records in the report. Observation media is "
            "written as shape/dtype references, but reports still grow quickly."
        ),
    )
    return parser


def main() -> None:
    """Run the benchmark evaluation CLI."""
    parser = build_parser()
    args = parser.parse_args()
    policy = (
        load_behavior_cloning_policy(args.policy, device=args.device)
        if args.policy is not None
        else None
    )

    if args.benchmark in {"gs_dronegym", "drone"}:
        scene = normalize_scene_arg(args.scene)
        benchmark = make_benchmark(args.benchmark, env_id=args.env_id, scene=scene)
        report = benchmark.evaluate_policy(
            policy=policy,
            n_episodes=args.n_episodes,
            include_raw_results=args.include_raw_results,
        )
    elif args.benchmark == "libero":
        benchmark = make_benchmark(args.benchmark)
        report = benchmark.evaluate_policy(
            policy=policy,
            source=args.source,
            max_episodes=args.n_episodes,
            include_raw_results=args.include_raw_results,
        )
    else:
        benchmark = make_benchmark(args.benchmark)
        report = benchmark.evaluate_policy(
            policy=policy,
            source=args.source,
            include_raw_results=args.include_raw_results,
        )

    if args.output is not None:
        report.to_json(args.output)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
