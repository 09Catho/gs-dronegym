"""Point-goal navigation benchmark for GS-DroneGym."""

from __future__ import annotations

import numpy as np

from gs_dronegym.tasks.base_task import BaseTask, TaskConfig


class PointNavTask(BaseTask):
    """Navigate to a randomly sampled 3D coordinate within the scene."""

    def __init__(self, config: TaskConfig | None = None) -> None:
        """Initialize the PointNav task.

        Args:
            config: Optional task configuration.
        """
        super().__init__(config=config)

    def reset(self, scene_bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Reset the task for a new episode.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.

        Returns:
            Initial drone state, goal position, and instruction string.
        """
        init_pos = self._sample_position(
            scene_bbox,
            margin=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        goal = self._sample_position(
            scene_bbox,
            margin=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        while float(np.linalg.norm(goal - init_pos)) < 2.0:
            goal = self._sample_position(
                scene_bbox,
                margin=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )

        state = np.zeros(12, dtype=np.float32)
        state[:3] = init_pos
        self.goal_position = goal.astype(np.float32)
        self.instruction = (
            f"fly to position ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})"
        )
        return state, self.goal_position.copy(), self.instruction

    def is_success(self, state: np.ndarray) -> bool:
        """Check whether the drone reached the point goal."""
        distance = float(np.linalg.norm(state[:3] - self.goal_position))
        return distance <= self.config.success_threshold

    @property
    def task_id(self) -> str:
        """Return the task identifier."""
        return "point_nav"
