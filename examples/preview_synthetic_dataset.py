"""Example preview script for synthetic dataset rollouts."""

from __future__ import annotations

from pathlib import Path

from gs_dronegym.data import preview_dataset_task


def main() -> None:
    """Preview one narrow corridor rollout and save a GIF."""
    summary = preview_dataset_task(
        scene=None,
        stage_name="stage2_flight_skills",
        task_id="narrow_corridor",
        steps=40,
        save_gif=Path("outputs") / "dataset_preview.gif",
        renderer_device="cpu",
        allow_mock_rendering=True,
    )
    print(summary.to_dict())


if __name__ == "__main__":
    main()
