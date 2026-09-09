"""Determinism guarantees for the environment's public seeding contract.

These tests pin the behaviour an RL library relies on: seeding once and then
calling ``reset()`` repeatedly must replay one reproducible stream of episodes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import gs_dronegym


def _episode_goals(n_episodes: int, seed: int = 0, augmentation: bool = False) -> np.ndarray:
    """Collect goal positions for a seed-once, reset-many episode sequence.

    Args:
        n_episodes: Number of episodes to reset through.
        seed: Seed supplied to the first reset only.
        augmentation: Whether observation augmentation is enabled.

    Returns:
        Array of goal positions, one row per episode.
    """
    env = gs_dronegym.make("PointNav-v0", scene=None, augmentation=augmentation)
    goals: list[np.ndarray] = []
    env.reset(seed=seed)
    goals.append(np.asarray(env.unwrapped.goal_position, dtype=np.float32).copy())
    for _ in range(n_episodes - 1):
        env.reset()
        goals.append(np.asarray(env.unwrapped.goal_position, dtype=np.float32).copy())
    env.close()
    return np.stack(goals)


def _rollout_digest(seed: int, n_steps: int = 60, augmentation: bool = False) -> str:
    """Hash a fixed-action rollout over observations, rewards and termination.

    Args:
        seed: Seed supplied to reset.
        n_steps: Number of steps to run.
        augmentation: Whether observation augmentation is enabled.

    Returns:
        Hex digest covering the whole trajectory.
    """
    env = gs_dronegym.make("PointNav-v0", scene=None, augmentation=augmentation)
    obs, _ = env.reset(seed=seed)
    digest = hashlib.blake2b(digest_size=16)
    action_rng = np.random.default_rng(12345)

    def absorb(observation: dict[str, object], reward: float, flags: tuple[bool, bool]) -> None:
        digest.update(np.asarray(observation["rgb"], dtype=np.uint8).tobytes())
        digest.update(np.asarray(observation["state"], dtype=np.float32).tobytes())
        if "depth" in observation:
            digest.update(np.asarray(observation["depth"], dtype=np.float32).tobytes())
        digest.update(str(observation["instruction"]).encode("utf-8"))
        digest.update(np.float64(reward).tobytes())
        digest.update(bytes(flags))

    absorb(obs, 0.0, (False, False))
    for _ in range(n_steps):
        action = action_rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        absorb(obs, float(reward), (bool(terminated), bool(truncated)))
        if terminated or truncated:
            obs, _ = env.reset()
            absorb(obs, 0.0, (False, False))
    env.close()
    return digest.hexdigest()


def test_seed_once_then_reset_replays_the_same_episode_sequence() -> None:
    """Seeding once must fix every later episode, not only the first."""
    first = _episode_goals(4, seed=0)
    second = _episode_goals(4, seed=0)
    assert np.array_equal(first, second)


def test_successive_episodes_are_not_identical() -> None:
    """A reproducible stream must still advance between episodes."""
    goals = _episode_goals(4, seed=0)
    assert not np.allclose(goals[0], goals[1])


def test_different_seeds_produce_different_streams() -> None:
    """Distinct seeds must not collapse onto the same episode sequence."""
    assert not np.array_equal(_episode_goals(4, seed=0), _episode_goals(4, seed=1))


def test_augmented_observations_are_reproducible() -> None:
    """Augmentation must draw from the seeded stream, not process entropy."""
    assert _rollout_digest(0, n_steps=8, augmentation=True) == _rollout_digest(
        0, n_steps=8, augmentation=True
    )


def test_augmentation_does_not_perturb_task_layout() -> None:
    """Task and augmentation streams must be independent."""
    assert np.array_equal(
        _episode_goals(4, seed=0, augmentation=False),
        _episode_goals(4, seed=0, augmentation=True),
    )


@pytest.mark.parametrize("seed", [0, 7])
def test_rollout_digest_is_stable_across_environment_instances(seed: int) -> None:
    """Two fresh environments at one seed must produce identical trajectories."""
    assert _rollout_digest(seed) == _rollout_digest(seed)


def test_rollout_digest_is_seed_sensitive() -> None:
    """The digest must actually depend on the seed."""
    assert _rollout_digest(0) != _rollout_digest(7)
