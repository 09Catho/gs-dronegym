"""Dynamic target following benchmark for GS-DroneGym."""

from __future__ import annotations

import numpy as np

from gs_dronegym.tasks.base_task import BaseTask, TaskConfig


class DynamicFollowTask(BaseTask):
    """Follow a moving target that travels on a circular path."""

    def __init__(
        self,
        radius: float = 2.0,
        angular_velocity: float = 0.4,
        required_hold_steps: int = 15,
        config: TaskConfig | None = None,
    ) -> None:
        """Initialize the dynamic follow task.

        Args:
            radius: Circular trajectory radius in meters.
            angular_velocity: Target angular velocity in radians per second.
            required_hold_steps: Consecutive in-range steps needed for success.
            config: Optional task configuration.
        """
        super().__init__(config=config)
        self.radius = radius
        self.angular_velocity = angular_velocity
        self.required_hold_steps = required_hold_steps
        self._center = np.zeros(3, dtype=np.float32)
        self._angle = 0.0
        self._close_steps = 0

    def reset(self, scene_bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Reset the moving target trajectory.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.

        Returns:
            Initial drone state, target position, and instruction.
        """
        self._center = self._sample_position(
            scene_bbox,
            margin=np.array([self.radius + 1.0, self.radius + 1.0, 1.0], dtype=np.float32),
        )
        self._angle = 0.0
        self._close_steps = 0
        self.goal_position = self._compute_target_position().astype(np.float32)

        state = np.zeros(12, dtype=np.float32)
        state[:3] = self._center + np.array([-self.radius, 0.0, 0.0], dtype=np.float32)
        self.instruction = "follow the moving target"
        return state, self.goal_position.copy(), self.instruction

    def update(self, state: np.ndarray, step: int, obs_dt: float) -> None:
        """Advance the target trajectory and update proximity counters.

        Args:
            state: Current drone state.
            step: Current environment step count.
            obs_dt: Environment observation time step.
        """
        del step
        self._angle += self.angular_velocity * obs_dt
        self.goal_position = self._compute_target_position().astype(np.float32)
        if float(np.linalg.norm(state[:3] - self.goal_position)) <= 1.0:
            self._close_steps += 1
        else:
            self._close_steps = 0

    def is_success(self, state: np.ndarray) -> bool:
        """Return whether the drone stayed near the target long enough."""
        del state
        return self._close_steps >= self.required_hold_steps

    def _compute_target_position(self) -> np.ndarray:
        """Compute the current moving target position.

        Returns:
            Target position as a ``(3,)`` array.
        """
        offset = np.array(
            [
                self.radius * np.cos(self._angle),
                self.radius * np.sin(self._angle),
                0.0,
            ],
            dtype=np.float32,
        )
        return self._center + offset

    @property
    def task_id(self) -> str:
        """Return the task identifier."""
        return "dynamic_follow"
