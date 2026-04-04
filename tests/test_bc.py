"""Tests for the lightweight behavior-cloning baseline."""

from __future__ import annotations

import numpy as np
import torch

from gs_dronegym.baselines import (
    BehaviorCloningConfig,
    evaluate_behavior_cloning,
    train_behavior_cloning,
)
from gs_dronegym.data import (
    ActionSpec,
    ObservationSpec,
    TaskSpec,
    TrajectoryEpisode,
    TrajectoryStep,
)


def _synthetic_episodes() -> list[TrajectoryEpisode]:
    """Create tiny synthetic episodes for behavior-cloning smoke tests.

    Returns:
        List of episodes.
    """
    episodes: list[TrajectoryEpisode] = []
    for episode_idx in range(2):
        steps: list[TrajectoryStep] = []
        for step_idx in range(4):
            state = np.asarray(
                [episode_idx, step_idx, step_idx + 1.0, 0.0],
                dtype=np.float32,
            )
            action = np.asarray([state[0] + state[1], state[2]], dtype=np.float32)
            steps.append(
                TrajectoryStep(
                    observation={
                        "state": state,
                        "instruction": "match the state",
                    },
                    action=action,
                    reward=1.0,
                    terminated=step_idx == 3,
                    truncated=False,
                    step_index=step_idx,
                )
            )
        episodes.append(
            TrajectoryEpisode(
                episode_id=f"episode-{episode_idx}",
                benchmark_name="synthetic",
                embodiment="unit_test",
                task=TaskSpec(
                    task_id="regression",
                    benchmark_name="synthetic",
                    embodiment="unit_test",
                    instruction="match the state",
                ),
                action_spec=ActionSpec(shape=(2,), normalized=False),
                observation_spec=ObservationSpec(
                    modalities=("state", "instruction"),
                    state_shape=(4,),
                ),
                steps=steps,
                success=True,
                split="train",
            )
        )
    return episodes


def test_behavior_cloning_training_and_eval(tmp_path: object) -> None:
    """The baseline should train and emit finite imitation metrics."""
    torch.manual_seed(0)
    episodes = _synthetic_episodes()
    checkpoint = tmp_path / "policy.pt"
    policy, summary = train_behavior_cloning(
        episodes,
        config=BehaviorCloningConfig(epochs=2, batch_size=2, learning_rate=1e-2),
        split="train",
        checkpoint_path=checkpoint,
    )
    metrics = evaluate_behavior_cloning(policy, episodes, split="train")
    assert summary.n_examples == 8
    assert checkpoint.exists()
    assert np.isfinite(metrics["action_mse"])
    assert np.isfinite(metrics["action_mae"])
