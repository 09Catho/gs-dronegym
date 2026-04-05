"""CLI entrypoint for previewing one synthetic dataset episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gs_dronegym.data import preview_dataset_task


def build_parser() -> argparse.ArgumentParser:
    """Create the preview parser."""
    parser = argparse.ArgumentParser(description="Preview one synthetic dataset task rollout.")
    parser.add_argument("--scene", default=None, help="Scene handle, local .ply path, or None.")
    parser.add_argument(
        "--stage",
        default="stage1_scene_comprehension",
        help="Curriculum stage name to preview.",
    )
    parser.add_argument("--task-id", default=None, help="Optional task id override.")
    parser.add_argument("--steps", type=int, default=60, help="Maximum preview steps.")
    parser.add_argument("--save-gif", default=None, help="Optional GIF output path.")
    parser.add_argument("--renderer-device", default="cpu", help="Render device.")
    parser.add_argument("--seed", type=int, default=0, help="Preview seed.")
    parser.add_argument(
        "--allow-mock-rendering",
        action="store_true",
        help="Permit scene=None or mock:// previews.",
    )
    return parser


def main() -> None:
    """Run the preview CLI."""
    parser = build_parser()
    args = parser.parse_args()
    summary = preview_dataset_task(
        scene=None if args.scene in {None, "None"} else args.scene,
        stage_name=args.stage,
        task_id=args.task_id,
        steps=args.steps,
        save_gif=None if args.save_gif is None else Path(args.save_gif),
        renderer_device=args.renderer_device,
        seed=args.seed,
        allow_mock_rendering=args.allow_mock_rendering,
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
