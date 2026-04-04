"""Export a GS-DroneGym rollout into the shared trajectory format."""

from __future__ import annotations

from pathlib import Path

from gs_dronegym import make_benchmark
from gs_dronegym.data import save_dataset


def main() -> None:
    """Run one random PointNav rollout and write it to disk."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    episode = benchmark.collect_episode(seed=0)
    output_path = save_dataset([episode], Path("outputs") / "drone_rollout.json")
    print(f"Saved rollout to {output_path}")


if __name__ == "__main__":
    main()
