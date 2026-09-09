"""The instruction observation must be consumable by standard RL libraries.

A ``spaces.Text`` entry inside a dictionary observation cannot be processed by
common learners, so the environment also exposes the same deterministic
fixed-width encoding the baseline policy trains on.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

import gs_dronegym
from gs_dronegym.utils.instruction_encoder import (
    DEFAULT_INSTRUCTION_DIM,
    encode_instruction,
)


def test_default_mode_preserves_the_text_observation() -> None:
    """The default configuration must stay backwards compatible."""
    env = gs_dronegym.make("PointNav-v0", scene=None)
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["instruction"], str)
    assert "instruction_features" not in obs
    env.close()


def test_feature_mode_yields_an_all_box_observation_space() -> None:
    """Feature mode must remove the text space so learners can consume it."""
    env = gs_dronegym.make("PointNav-v0", scene=None, instruction_mode="features")
    obs, _ = env.reset(seed=0)
    assert all(
        isinstance(space, spaces.Box) for space in env.observation_space.spaces.values()
    )
    assert "instruction" not in obs
    assert obs["instruction_features"].shape == (DEFAULT_INSTRUCTION_DIM,)
    assert env.observation_space.contains(obs)
    env.close()


def test_both_mode_exposes_text_and_features() -> None:
    """Both mode must carry the raw string alongside its encoding."""
    env = gs_dronegym.make("PointNav-v0", scene=None, instruction_mode="both")
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["instruction"], str)
    assert np.array_equal(
        obs["instruction_features"],
        encode_instruction(str(obs["instruction"]), DEFAULT_INSTRUCTION_DIM),
    )
    env.close()


def test_features_match_the_shared_encoder_used_for_training() -> None:
    """Environment features must equal what the baseline encodes at train time."""
    env = gs_dronegym.make("PointNav-v0", scene=None, instruction_mode="both")
    obs, _ = env.reset(seed=0)
    from gs_dronegym.baselines.behavior_cloning import _hash_instruction

    assert np.array_equal(
        obs["instruction_features"],
        _hash_instruction(str(obs["instruction"]), DEFAULT_INSTRUCTION_DIM),
    )
    env.close()


def test_instruction_dim_is_configurable() -> None:
    """A custom feature width must be reflected in the space and the output."""
    env = gs_dronegym.make(
        "PointNav-v0", scene=None, instruction_mode="features", instruction_dim=32
    )
    obs, _ = env.reset(seed=0)
    assert obs["instruction_features"].shape == (32,)
    assert env.observation_space.spaces["instruction_features"].shape == (32,)
    env.close()


def test_feature_mode_works_without_images() -> None:
    """Feature mode must compose with state-only observations."""
    env = gs_dronegym.make(
        "PointNav-v0", scene=None, instruction_mode="features", observation_mode="state"
    )
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"state", "instruction_features"}
    assert env.observation_space.contains(obs)
    env.close()


def test_invalid_instruction_dim_is_rejected() -> None:
    """A non-positive feature width must fail loudly."""
    with pytest.raises(ValueError, match="instruction_dim must be positive"):
        gs_dronegym.make("PointNav-v0", scene=None, instruction_dim=0)


def test_encoder_is_deterministic_for_repeated_calls() -> None:
    """Encoding must be a pure function of the text and width."""
    text = "fly to the red chair near the window"
    assert np.array_equal(encode_instruction(text, 64), encode_instruction(text, 64))
    assert not np.array_equal(encode_instruction(text, 64), encode_instruction("land", 64))
