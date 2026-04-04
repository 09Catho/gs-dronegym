"""Evaluate a random or learned policy across a supported benchmark adapter."""

from __future__ import annotations

import json

from gs_dronegym import make_benchmark


def main() -> None:
    """Run a small live drone evaluation and print the report."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    report = benchmark.evaluate_policy(policy=None, n_episodes=2)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
