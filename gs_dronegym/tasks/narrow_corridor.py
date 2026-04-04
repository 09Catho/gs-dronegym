"""Narrow corridor navigation benchmark for GS-DroneGym."""

from __future__ import annotations

import numpy as np

from gs_dronegym.tasks.base_task import BaseTask, BoxObstacle, Obstacle, TaskConfig


class NarrowCorridorTask(BaseTask):
    """Navigate through a tight straight corridor without touching walls."""

    def __init__(self, config: TaskConfig | None = None) -> None:
        """Initialize the corridor task.

        Args:
            config: Optional task configuration.
        """
        super().__init__(config=config)
        self.walls: list[BoxObstacle] = []

    def reset(self, scene_bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Reset the corridor geometry and start/goal states.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.

        Returns:
            Initial drone state, goal position, and instruction.
        """
        center_y = float(np.mean(scene_bbox[:, 1]))
        center_z = float(
            np.clip(scene_bbox[0, 2] + 1.2, scene_bbox[0, 2] + 0.5, scene_bbox[1, 2] - 0.5)
        )
        start_x = float(scene_bbox[0, 0] + 0.5)
        corridor_length = float(
            max(5.0, min(15.0, scene_bbox[1, 0] - scene_bbox[0, 0] - 1.0))
        )
        end_x = start_x + corridor_length
        half_gap = 0.75
        wall_thickness = 0.25
        wall_height = 2.5

        left_min = np.array(
            [start_x, center_y - half_gap - wall_thickness, center_z - wall_height / 2.0],
            dtype=np.float32,
        )
        left_max = np.array(
            [end_x, center_y - half_gap, center_z + wall_height / 2.0],
            dtype=np.float32,
        )
        right_min = np.array(
            [start_x, center_y + half_gap, center_z - wall_height / 2.0],
            dtype=np.float32,
        )
        right_max = np.array(
            [end_x, center_y + half_gap + wall_thickness, center_z + wall_height / 2.0],
            dtype=np.float32,
        )
        self.walls = [
            BoxObstacle(min_corner=left_min, max_corner=left_max),
            BoxObstacle(min_corner=right_min, max_corner=right_max),
        ]

        state = np.zeros(12, dtype=np.float32)
        state[:3] = np.array([start_x + 0.5, center_y, center_z], dtype=np.float32)
        self.goal_position = np.array([end_x - 0.5, center_y, center_z], dtype=np.float32)
        self.instruction = "fly through the narrow corridor"
        return state, self.goal_position.copy(), self.instruction

    def is_success(self, state: np.ndarray) -> bool:
        """Return whether the corridor exit has been reached."""
        distance = float(np.linalg.norm(state[:3] - self.goal_position))
        return distance <= self.config.success_threshold

    def get_obstacles(self) -> list[Obstacle]:
        """Return corridor walls as collision boxes."""
        return list(self.walls)

    @property
    def task_id(self) -> str:
        """Return the task identifier."""
        return "narrow_corridor"
