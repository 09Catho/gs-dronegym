"""Open a live matplotlib viewer for a GS-DroneGym rollout."""

from __future__ import annotations

from gs_dronegym.cli.live_viewer import run_live_viewer


def main() -> None:
    """Run the live viewer with default mock settings."""
    run_live_viewer(env_id="PointNav-v0", scene=None, steps=60, seed=0, policy="random")


if __name__ == "__main__":
    main()
