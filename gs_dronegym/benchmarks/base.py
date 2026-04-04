"""Base interfaces and shared helpers for benchmark adapters.

The benchmark layer normalizes data loading, evaluation, and reporting across
drone navigation, manipulation benchmarks such as LIBERO, and dataset-first
formats such as LeRobot. This base module provides the common adapter contract
and utilities shared by concrete benchmark implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from gs_dronegym.data.schema import BenchmarkReport, JsonValue, TrajectoryEpisode

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class PolicyLike(Protocol):
    """Protocol for policies that can produce actions from observations."""

    def predict(self, observation: dict[str, object]) -> np.ndarray:
        """Predict an action for one observation."""


def call_policy(
    policy: PolicyLike | None | object,
    observation: dict[str, object],
    action_dim: int,
) -> np.ndarray:
    """Normalize policy invocation across callables and predictor objects.

    Args:
        policy: Policy object, callable, or ``None`` for a zero policy.
        observation: Current observation dictionary.
        action_dim: Expected action dimension.

    Returns:
        Action vector as ``float32``.
    """
    if policy is None:
        return np.zeros(action_dim, dtype=np.float32)
    if isinstance(policy, PolicyLike):
        return np.asarray(policy.predict(observation), dtype=np.float32)
    if callable(policy):
        return np.asarray(policy(observation), dtype=np.float32)
    raise TypeError(f"Unsupported policy object: {type(policy)!r}")


def compute_core_metrics(episodes: list[TrajectoryEpisode]) -> dict[str, float]:
    """Compute embodiment-agnostic metrics over a set of episodes.

    Args:
        episodes: Episodes to summarize.

    Returns:
        Dictionary of core benchmark metrics.
    """
    if not episodes:
        return {
            "success_rate": 0.0,
            "mean_return": 0.0,
            "mean_episode_length": 0.0,
        }
    successes = float(sum(1 for episode in episodes if episode.success))
    returns = np.asarray([episode.total_reward for episode in episodes], dtype=np.float32)
    lengths = np.asarray([episode.n_steps for episode in episodes], dtype=np.float32)
    return {
        "success_rate": successes / float(len(episodes)),
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def build_task_breakdown(episodes: list[TrajectoryEpisode]) -> dict[str, dict[str, JsonValue]]:
    """Aggregate simple per-task metrics for report generation.

    Args:
        episodes: Episodes to summarize.

    Returns:
        Per-task breakdown dictionary.
    """
    grouped: dict[str, list[TrajectoryEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.task.task_id, []).append(episode)
    breakdown: dict[str, dict[str, JsonValue]] = {}
    for task_id, task_episodes in grouped.items():
        core_metrics = compute_core_metrics(task_episodes)
        breakdown[task_id] = {
            "n_episodes": len(task_episodes),
            "success_rate": core_metrics["success_rate"],
            "mean_return": core_metrics["mean_return"],
            "mean_episode_length": core_metrics["mean_episode_length"],
        }
    return breakdown


class BenchmarkAdapter(ABC):
    """Abstract benchmark adapter."""

    name: str
    embodiment: str

    @abstractmethod
    def load_dataset(self, source: str) -> list[TrajectoryEpisode]:
        """Load benchmark trajectories into the common schema."""

    @abstractmethod
    def evaluate_policy(self, *args: object, **kwargs: object) -> BenchmarkReport:
        """Evaluate a policy or model and return a normalized report."""
