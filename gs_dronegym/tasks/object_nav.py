"""Language-driven object navigation benchmark for GS-DroneGym."""

from __future__ import annotations

import numpy as np

from gs_dronegym.tasks.base_task import BaseTask, TaskConfig


class ObjectNavTask(BaseTask):
    """Navigate to a named semantic region within the scene."""

    def __init__(
        self,
        regions: dict[str, np.ndarray] | None = None,
        config: TaskConfig | None = None,
    ) -> None:
        """Initialize the ObjectNav task.

        Args:
            regions: Mapping from region label to ``(2, 3)`` bounding box.
            config: Optional task configuration.
        """
        super().__init__(config=config)
        self.regions = regions or {}
        self.current_label = "target region"

    def reset(self, scene_bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Reset the task for a new episode.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.

        Returns:
            Initial drone state, goal position, and instruction string.
        """
        state = np.zeros(12, dtype=np.float32)
        state[:3] = self._sample_position(
            scene_bbox,
            margin=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        if self.regions:
            labels = sorted(self.regions.keys())
            self.current_label = labels[int(self._rng.integers(0, len(labels)))]
            region = self.regions[self.current_label].astype(np.float32)
            low = region[0]
            high = region[1]
            self.goal_position = self._rng.uniform(low=low, high=high).astype(np.float32)
        else:
            self.current_label = "target region"
            self.goal_position = self._sample_position(
                scene_bbox,
                margin=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )

        self.instruction = f"navigate to the {self.current_label}"
        return state, self.goal_position.copy(), self.instruction

    def is_success(self, state: np.ndarray) -> bool:
        """Check whether the drone reached the sampled region goal."""
        distance = float(np.linalg.norm(state[:3] - self.goal_position))
        return distance <= self.config.success_threshold

    @property
    def task_id(self) -> str:
        """Return the task identifier."""
        return "object_nav"
