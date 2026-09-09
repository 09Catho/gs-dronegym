"""Tests for the lightweight behavior-cloning baseline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from gs_dronegym.baselines import (
    INSTRUCTION_ENCODER_VERSION,
    BehaviorCloningConfig,
    evaluate_behavior_cloning,
    load_behavior_cloning_policy,
    train_behavior_cloning,
)
from gs_dronegym.baselines.behavior_cloning import _hash_instruction
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


def test_behavior_cloning_training_and_eval(tmp_path: Path) -> None:
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


_CROSS_PROCESS_SNIPPET = """
import json
import sys

from gs_dronegym.baselines.behavior_cloning import _hash_instruction

vector = _hash_instruction(sys.argv[1], 64)
print(json.dumps([float(value) for value in vector]))
"""


def _encode_in_subprocess(text: str, hash_seed: str) -> list[float]:
    """Encode an instruction in a fresh interpreter with a given hash seed.

    Args:
        text: Instruction to encode.
        hash_seed: Value for ``PYTHONHASHSEED`` in the child process.

    Returns:
        Encoded instruction feature vector.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        [sys.executable, "-c", _CROSS_PROCESS_SNIPPET, text],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )
    return cast(list[float], json.loads(completed.stdout))


def test_instruction_encoding_is_stable_across_process_hash_seeds() -> None:
    """Instruction features must not depend on PYTHONHASHSEED."""
    text = "fly to the red chair near the window"
    reference = _hash_instruction(text, 64)
    for hash_seed in ("0", "1", "2", "12345"):
        encoded = np.asarray(_encode_in_subprocess(text, hash_seed), dtype=np.float32)
        assert np.array_equal(encoded, reference), f"PYTHONHASHSEED={hash_seed} changed features"
    assert float(reference.sum()) > 0.0


def test_checkpoint_records_instruction_encoder_version(tmp_path: Path) -> None:
    """Saved checkpoints must record the encoder version used for training."""
    episodes = _synthetic_episodes()
    checkpoint = tmp_path / "policy.pt"
    train_behavior_cloning(
        episodes,
        config=BehaviorCloningConfig(epochs=1, batch_size=2),
        split="train",
        checkpoint_path=checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["model_spec"]["instruction_encoder_version"] == INSTRUCTION_ENCODER_VERSION
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        load_behavior_cloning_policy(checkpoint)


def test_legacy_checkpoint_loads_with_explicit_warning(tmp_path: Path) -> None:
    """Checkpoints without an encoder version must warn instead of loading silently."""
    episodes = _synthetic_episodes()
    checkpoint = tmp_path / "legacy.pt"
    train_behavior_cloning(
        episodes,
        config=BehaviorCloningConfig(epochs=1, batch_size=2),
        split="train",
        checkpoint_path=checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    del payload["model_spec"]["instruction_encoder_version"]
    torch.save(payload, checkpoint)

    with pytest.warns(RuntimeWarning, match="instruction_encoder_version"):
        policy = load_behavior_cloning_policy(checkpoint)
    assert policy.instruction_dim == BehaviorCloningConfig().instruction_dim


def test_future_encoder_version_is_rejected(tmp_path: Path) -> None:
    """A checkpoint from a newer encoder must fail loudly rather than mis-encode."""
    episodes = _synthetic_episodes()
    checkpoint = tmp_path / "future.pt"
    train_behavior_cloning(
        episodes,
        config=BehaviorCloningConfig(epochs=1, batch_size=2),
        split="train",
        checkpoint_path=checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    payload["model_spec"]["instruction_encoder_version"] = INSTRUCTION_ENCODER_VERSION + 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="instruction encoder version"):
        load_behavior_cloning_policy(checkpoint)
