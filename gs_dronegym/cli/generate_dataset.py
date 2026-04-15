"""CLI entrypoint for synthetic VLA-AN-like dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gs_dronegym.data import (
    CurriculumStageConfig,
    DatasetGenerationConfig,
    SceneSelectionConfig,
    default_curriculum_stages,
    generate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the dataset generation parser."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic VLA-AN-like aerial waypoint dataset."
    )
    parser.add_argument("output_root", help="Output dataset root directory.")
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="Scene handles: builtin names, local .ply paths, or mock:// names.",
    )
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        default=12,
        help="Episodes to generate for each resolved scene.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=8,
        help="Episodes per Parquet shard.",
    )
    parser.add_argument("--dataset-id", default="synthetic_vla_an", help="Dataset identifier.")
    parser.add_argument("--seed", type=int, default=0, help="Global seed.")
    parser.add_argument("--renderer-device", default="cuda", help="Render device.")
    parser.add_argument("--width", type=int, default=224, help="RGB/depth width.")
    parser.add_argument("--height", type=int, default=224, help="RGB/depth height.")
    parser.add_argument(
        "--debug-episodes",
        type=int,
        default=2,
        help="Per-split debug JSON episodes to export.",
    )
    parser.add_argument(
        "--allow-mock-rendering",
        action="store_true",
        help="Permit mock:// scene sources for tests and CPU-only runs.",
    )
    parser.add_argument(
        "--task-filter",
        nargs="+",
        default=None,
        help=(
            "Optional task IDs to keep, e.g. point_nav. Useful for focused "
            "debug datasets before training on the full curriculum."
        ),
    )
    return parser


def _filtered_stages(task_filter: list[str] | None) -> tuple[CurriculumStageConfig, ...]:
    """Build curriculum stages after applying an optional task filter."""
    if not task_filter:
        return tuple()
    allowed = set(task_filter)
    stages: list[CurriculumStageConfig] = []
    for stage in default_curriculum_stages():
        task_ids = tuple(task_id for task_id in stage.task_ids if task_id in allowed)
        if not task_ids:
            continue
        stages.append(
            CurriculumStageConfig(
                name=stage.name,
                weight=stage.weight,
                task_ids=task_ids,
                max_steps=stage.max_steps,
                description=f"{stage.description} Filtered to {', '.join(task_ids)}.",
            )
        )
    if not stages:
        raise ValueError(f"No curriculum stages contain requested task IDs: {sorted(allowed)}")
    return tuple(stages)


def main() -> None:
    """Run the synthetic dataset generation CLI."""
    parser = build_parser()
    args = parser.parse_args()
    config = DatasetGenerationConfig(
        output_root=Path(args.output_root),
        scene_selection=SceneSelectionConfig(
            sources=tuple(str(item) for item in args.scenes),
        ),
        episodes_per_scene=args.episodes_per_scene,
        shard_size_episodes=args.shard_size,
        dataset_id=args.dataset_id,
        seed=args.seed,
        renderer_device=args.renderer_device,
        image_size=(args.width, args.height),
        debug_export_episodes_per_split=args.debug_episodes,
        allow_mock_rendering=args.allow_mock_rendering,
        stages=_filtered_stages(args.task_filter),
    )
    manifest = generate_dataset(config)
    print(json.dumps(manifest.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
