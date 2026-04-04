"""Tests for the shared trajectory schema and dataset serialization."""

from __future__ import annotations

import numpy as np

from gs_dronegym.data import (
    ActionSpec,
    ObservationSpec,
    TaskSpec,
    TrajectoryEpisode,
    TrajectoryStep,
    load_common_dataset,
    save_dataset,
)


def _sample_episode() -> TrajectoryEpisode:
    """Construct a small synthetic episode for schema tests.

    Returns:
        Synthetic trajectory episode.
    """
    observation = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth": np.ones((8, 8), dtype=np.float32),
        "state": np.arange(12, dtype=np.float32),
        "instruction": "fly to the marker",
    }
    step = TrajectoryStep(
        observation=observation,
        action=np.asarray([0.1, -0.2, 0.0, 0.3], dtype=np.float32),
        reward=1.5,
        terminated=True,
        truncated=False,
        info={"collision": False},
        step_index=0,
    )
    return TrajectoryEpisode(
        episode_id="episode-0",
        benchmark_name="gs_dronegym",
        embodiment="drone",
        task=TaskSpec(
            task_id="point_nav",
            benchmark_name="gs_dronegym",
            embodiment="drone",
            instruction="fly to the marker",
        ),
        action_spec=ActionSpec(shape=(4,)),
        observation_spec=ObservationSpec(
            modalities=("rgb", "depth", "state", "instruction"),
            rgb_shape=(8, 8, 3),
            depth_shape=(8, 8),
            state_shape=(12,),
        ),
        steps=[step],
        success=True,
        split="train",
        source="unit_test",
    )


def test_episode_round_trip_preserves_arrays() -> None:
    """Trajectory episodes should survive dict serialization round-trips."""
    episode = _sample_episode()
    restored = TrajectoryEpisode.from_dict(episode.to_dict())
    assert restored.episode_id == episode.episode_id
    assert restored.task.task_id == episode.task.task_id
    assert np.array_equal(restored.steps[0].observation["rgb"], episode.steps[0].observation["rgb"])
    assert np.allclose(restored.steps[0].action, episode.steps[0].action)


def test_save_and_load_common_dataset(tmp_path: object) -> None:
    """Normalized datasets should round-trip through disk IO."""
    episode = _sample_episode()
    output = save_dataset([episode], tmp_path)
    loaded = load_common_dataset(output)
    assert len(loaded) == 1
    assert loaded[0].success is True
    assert loaded[0].observation_spec.rgb_shape == (8, 8, 3)
