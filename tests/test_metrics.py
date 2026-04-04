"""Tests for benchmark metric helpers."""

from __future__ import annotations

import numpy as np

from gs_dronegym.utils.metrics import Episode, path_length, spl, success_rate


def _episode(positions: list[list[float]], success: bool) -> Episode:
    """Create a synthetic episode for testing metrics.

    Args:
        positions: Position waypoints.
        success: Success flag.

    Returns:
        Episode instance.
    """
    pos_arrays = [np.asarray(position, dtype=np.float32) for position in positions]
    return Episode(
        positions=pos_arrays,
        actions=[],
        rewards=[],
        success=success,
        collision=not success,
        goal_position=pos_arrays[-1],
        n_steps=max(len(pos_arrays) - 1, 0),
    )


def test_spl_is_one_for_optimal_path() -> None:
    """Straight-line successful paths should achieve SPL of one."""
    episode = _episode([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], success=True)
    assert spl([episode]) == 1.0


def test_spl_is_zero_when_all_episodes_fail() -> None:
    """Failed episodes should contribute zero to SPL."""
    episodes = [_episode([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], success=False) for _ in range(3)]
    assert spl(episodes) == 0.0


def test_success_rate_half_for_five_of_ten() -> None:
    """Success rate should reflect the fraction of successful runs."""
    episodes = [_episode([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], success=idx < 5) for idx in range(10)]
    assert success_rate(episodes) == 0.5


def test_path_length_is_at_least_euclidean_distance() -> None:
    """Path length should never undershoot the direct start-goal distance."""
    episode = _episode(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]],
        success=True,
    )
    start = np.asarray(episode.positions[0], dtype=np.float32)
    goal = np.asarray(episode.goal_position, dtype=np.float32)
    assert path_length(episode) >= float(np.linalg.norm(goal - start))
