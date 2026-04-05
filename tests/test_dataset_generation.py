"""Tests for synthetic VLA-AN-like dataset generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gs_dronegym.baselines import BehaviorCloningConfig, train_behavior_cloning
from gs_dronegym.data import (
    CurriculumStageConfig,
    DatasetGenerationConfig,
    SceneSelectionConfig,
    generate_dataset,
    load_dataset,
    preview_dataset_task,
    validate_generated_dataset,
)


def _generator_config(output_root: Path, scenes: tuple[str, ...]) -> DatasetGenerationConfig:
    """Create a small mock-backed generation config for tests."""
    return DatasetGenerationConfig(
        output_root=output_root,
        scene_selection=SceneSelectionConfig(sources=scenes),
        stages=(
            CurriculumStageConfig(
                name="stage1_scene_comprehension",
                weight=1.0,
                task_ids=("point_nav",),
                max_steps=12,
                description="Fast unit-test curriculum.",
            ),
        ),
        episodes_per_scene=2,
        shard_size_episodes=2,
        renderer_device="cpu",
        allow_mock_rendering=True,
        debug_export_episodes_per_split=1,
        seed=7,
    )


def test_generate_dataset_writes_manifest_and_media(tmp_path: Path) -> None:
    """Synthetic dataset generation should create shards, debug JSON, and media."""
    config = _generator_config(tmp_path / "dataset", ("mock://scene_a",))
    manifest = generate_dataset(config)
    assert manifest.counts["n_episodes"] == 2
    assert (config.output_root / "manifest.json").exists()
    assert (config.output_root / "splits.json").exists()
    assert list((config.output_root / "parquet" / "train").glob("steps-*.parquet"))
    assert list((config.output_root / "episodes_debug" / "train").glob("*.json"))
    rgb_files = list((config.output_root / "media" / "train").glob("**/*.png"))
    depth_files = list((config.output_root / "media" / "train").glob("**/*.npy"))
    assert rgb_files
    assert depth_files


def test_generated_dataset_round_trip_and_bc_training(tmp_path: Path) -> None:
    """Generated Parquet datasets should reload and train the BC baseline."""
    config = _generator_config(tmp_path / "dataset", ("mock://scene_a",))
    generate_dataset(config)
    episodes = load_dataset(config.output_root, format="gs_dronegym")
    assert episodes
    assert "expert_waypoint" in episodes[0].steps[0].info
    report = validate_generated_dataset(config.output_root)
    assert report.valid is True
    policy, summary = train_behavior_cloning(
        episodes=episodes,
        config=BehaviorCloningConfig(epochs=1, batch_size=2, learning_rate=1e-3),
        split="train",
        checkpoint_path=tmp_path / "policy.pt",
    )
    assert summary.n_examples >= len(episodes)
    prediction = policy.predict(episodes[0].steps[0].observation)
    assert prediction.shape == (4,)
    assert np.all(np.isfinite(prediction))


def test_preview_dataset_task_saves_gif(tmp_path: Path) -> None:
    """Preview mode should render a short GIF for one sampled dataset task."""
    gif_path = tmp_path / "preview.gif"
    summary = preview_dataset_task(
        scene=None,
        stage_name="stage2_flight_skills",
        task_id="narrow_corridor",
        steps=10,
        save_gif=gif_path,
        renderer_device="cpu",
        allow_mock_rendering=True,
    )
    assert summary.n_steps >= 1
    assert gif_path.exists()


def test_scene_split_assignment_uses_scene_level_holdouts(tmp_path: Path) -> None:
    """Scene-level splits should keep held-out scenes out of train."""
    config = _generator_config(
        tmp_path / "dataset",
        ("mock://scene_a", "mock://scene_b", "mock://scene_c"),
    )
    manifest = generate_dataset(config)
    splits = manifest.scene_splits
    assert len(splits) == 3
    assert "train" in set(splits.values())
    assert "val" in set(splits.values()) or "test" in set(splits.values())
