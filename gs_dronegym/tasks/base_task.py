"""Abstract task interfaces for drone navigation benchmarks.

Tasks define reset logic, reward shaping, success criteria, and optional dynamic
scene elements for GS-DroneGym. They are designed to stay lightweight while
providing enough hooks for the environment to combine task logic with dynamics
and rendering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class TaskConfig:
    """Configuration shared by benchmark tasks.

    Attributes:
        max_steps: Maximum environment steps before truncation.
        success_threshold: Euclidean distance threshold to count success.
        collision_penalty: Penalty applied when a collision occurs.
        step_penalty: Per-step penalty to encourage efficiency.
        success_reward: Reward bonus applied on success.
    """

    max_steps: int = 200
    success_threshold: float = 0.5
    collision_penalty: float = -10.0
    step_penalty: float = -0.01
    success_reward: float = 10.0


@dataclass(slots=True)
class CylinderObstacle:
    """Simple cylindrical obstacle used for collision detection."""

    center: np.ndarray
    radius: float
    height: float
    kind: str = field(default="cylinder", init=False)


@dataclass(slots=True)
class BoxObstacle:
    """Axis-aligned box obstacle used for collision detection."""

    min_corner: np.ndarray
    max_corner: np.ndarray
    kind: str = field(default="box", init=False)


Obstacle = CylinderObstacle | BoxObstacle


class BaseTask(ABC):
    """Abstract base class for all GS-DroneGym tasks."""

    def __init__(self, config: TaskConfig | None = None) -> None:
        """Initialize a task instance.

        Args:
            config: Optional task configuration.
        """
        self.config = config or TaskConfig()
        self._rng = np.random.default_rng()
        self.goal_position = np.zeros(3, dtype=np.float32)
        self.instruction = ""

    def seed(self, seed: int | None) -> None:
        """Seed the task RNG.

        Args:
            seed: RNG seed. If ``None``, the generator is reinitialized
                nondeterministically.
        """
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def reset(self, scene_bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Reset task state for a new episode.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.

        Returns:
            Tuple of initial drone state, goal position, and language instruction.
        """

    @abstractmethod
    def is_success(self, state: np.ndarray) -> bool:
        """Return whether the task objective is satisfied.

        Args:
            state: Current drone state.

        Returns:
            ``True`` when the episode should terminate with success.
        """

    def is_failure(self, state: np.ndarray, collision: bool) -> bool:
        """Return whether the episode should terminate with failure.

        Args:
            state: Current drone state.
            collision: Whether a collision was detected.

        Returns:
            ``True`` if the episode should terminate as a failure.
        """
        del state
        return collision

    def compute_reward(
        self,
        state: np.ndarray,
        prev_state: np.ndarray,
        collision: bool,
        step: int,
    ) -> float:
        """Compute potential-based reward shaping.

        Args:
            state: Current drone state.
            prev_state: Previous drone state.
            collision: Whether a collision occurred this step.
            step: Current environment step index.

        Returns:
            Reward for the transition.
        """
        del step
        prev_distance = float(np.linalg.norm(prev_state[:3] - self.get_goal_position()))
        curr_distance = float(np.linalg.norm(state[:3] - self.get_goal_position()))
        reward = prev_distance - curr_distance + self.config.step_penalty
        if collision:
            reward += self.config.collision_penalty
        if self.is_success(state):
            reward += self.config.success_reward
        return float(reward)

    def update(self, state: np.ndarray, step: int, obs_dt: float) -> None:
        """Advance internal task state after an environment step.

        Args:
            state: Current drone state.
            step: Current environment step count.
            obs_dt: Environment observation time step in seconds.
        """
        del state, step, obs_dt

    def get_obstacles(self) -> list[Obstacle]:
        """Return obstacles active for the task.

        Returns:
            A list of obstacles used for collision detection.
        """
        return []

    def get_goal_position(self) -> np.ndarray:
        """Return the current task goal position.

        Returns:
            Goal position as a ``(3,)`` array.
        """
        return self.goal_position.astype(np.float32, copy=True)

    def _sample_position(
        self,
        scene_bbox: np.ndarray,
        margin: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample a position uniformly within a scene bounding box.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.
            margin: Optional margin removed from each side.

        Returns:
            Sampled XYZ position.
        """
        margin_vec = (
            margin.astype(np.float32)
            if margin is not None
            else np.array([0.5, 0.5, 0.5], dtype=np.float32)
        )
        low = scene_bbox[0].astype(np.float32) + margin_vec
        high = scene_bbox[1].astype(np.float32) - margin_vec
        if np.any(high <= low):
            return ((scene_bbox[0] + scene_bbox[1]) / 2.0).astype(np.float32)
        return self._rng.uniform(low=low, high=high).astype(np.float32)

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Return a stable string identifier for the task."""
