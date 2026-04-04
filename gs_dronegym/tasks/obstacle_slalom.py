"""Static obstacle slalom benchmark for GS-DroneGym."""

from __future__ import annotations

import numpy as np

from gs_dronegym.tasks.base_task import BaseTask, CylinderObstacle, Obstacle, TaskConfig


class ObstacleSlalomTask(BaseTask):
    """Weave through a sequence of static cylindrical obstacles."""

    def __init__(self, config: TaskConfig | None = None) -> None:
        """Initialize the slalom task.

        Args:
            config: Optional task configuration.
        """
        super().__init__(config=config)
        self.obstacles: list[CylinderObstacle] = []
        self.gate_centers = np.zeros((0, 3), dtype=np.float32)
        self.final_goal = np.zeros(3, dtype=np.float32)
        self.current_gate_index = 0

    def reset(self, scene_bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Reset the obstacle course.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.

        Returns:
            Initial drone state, current goal position, and instruction.
        """
        center_y = float(np.mean(scene_bbox[:, 1]))
        base_z = float(
            np.clip(scene_bbox[0, 2] + 1.2, scene_bbox[0, 2] + 0.5, scene_bbox[1, 2] - 0.5)
        )
        start_x = float(scene_bbox[0, 0] + 1.0)
        usable_span = float(max(5.0, min(10.0, scene_bbox[1, 0] - scene_bbox[0, 0] - 3.0)))
        gate_xs = np.linspace(start_x + 1.5, start_x + usable_span, num=5, dtype=np.float32)
        offsets = np.array([-0.9, 0.9, -0.9, 0.9, -0.9], dtype=np.float32)

        self.obstacles = []
        gate_centers: list[np.ndarray] = []
        for x, y_offset in zip(gate_xs, offsets, strict=True):
            obstacle_center = np.array([x, center_y + y_offset, base_z], dtype=np.float32)
            self.obstacles.append(
                CylinderObstacle(center=obstacle_center, radius=0.3, height=2.0)
            )
            gate_centers.append(np.array([x, center_y, base_z], dtype=np.float32))

        self.gate_centers = np.stack(gate_centers).astype(np.float32)
        self.current_gate_index = 0
        self.final_goal = np.array(
            [min(gate_xs[-1] + 1.5, float(scene_bbox[1, 0] - 0.5)), center_y, base_z],
            dtype=np.float32,
        )
        self.goal_position = self.gate_centers[0].copy()

        state = np.zeros(12, dtype=np.float32)
        state[:3] = np.array([start_x, center_y, base_z], dtype=np.float32)
        self.instruction = "weave through all the obstacles ahead"
        return state, self.goal_position.copy(), self.instruction

    def update(self, state: np.ndarray, step: int, obs_dt: float) -> None:
        """Advance gate progress based on the current drone position.

        Args:
            state: Current drone state.
            step: Current step count.
            obs_dt: Observation step duration.
        """
        del step, obs_dt
        if self.current_gate_index < len(self.gate_centers):
            gate = self.gate_centers[self.current_gate_index]
            if float(np.linalg.norm(state[:3] - gate)) <= 1.5:
                self.current_gate_index += 1

        if self.current_gate_index < len(self.gate_centers):
            self.goal_position = self.gate_centers[self.current_gate_index].copy()
        else:
            self.goal_position = self.final_goal.copy()

    def is_success(self, state: np.ndarray) -> bool:
        """Return whether all gates were cleared and the finish reached."""
        if self.current_gate_index < len(self.gate_centers):
            return False
        distance = float(np.linalg.norm(state[:3] - self.final_goal))
        return distance <= self.config.success_threshold

    def get_obstacles(self) -> list[Obstacle]:
        """Return cylindrical obstacles for collision detection."""
        return list(self.obstacles)

    @property
    def task_id(self) -> str:
        """Return the task identifier."""
        return "obstacle_slalom"
