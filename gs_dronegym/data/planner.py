"""Geometric expert planning utilities for synthetic aerial dataset creation.

This module provides a lightweight oracle that converts task goals and obstacle
geometry into collision-aware waypoint targets for GS-DroneGym dataset
generation. The planner is intentionally simple and deterministic so it can
produce reproducible supervision labels at scale without introducing heavy
external planning dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gs_dronegym.tasks import BaseTask, DynamicFollowTask
from gs_dronegym.tasks.base_task import BoxObstacle, CylinderObstacle, Obstacle


@dataclass(slots=True)
class PlannerConfig:
    """Configuration for the synthetic dataset expert planner."""

    waypoint_spacing_m: float = 0.75
    lookahead_distance_m: float = 1.2
    obstacle_clearance_m: float = 0.75
    imminent_clearance_m: float = 0.6
    recovery_clearance_m: float = 0.35
    dynamic_target_lookahead_steps: int = 3
    max_altitude_step_m: float = 0.6

    def to_dict(self) -> dict[str, float]:
        """Serialize the planner configuration."""
        return {
            "waypoint_spacing_m": float(self.waypoint_spacing_m),
            "lookahead_distance_m": float(self.lookahead_distance_m),
            "obstacle_clearance_m": float(self.obstacle_clearance_m),
            "imminent_clearance_m": float(self.imminent_clearance_m),
            "recovery_clearance_m": float(self.recovery_clearance_m),
            "dynamic_target_lookahead_steps": int(self.dynamic_target_lookahead_steps),
            "max_altitude_step_m": float(self.max_altitude_step_m),
        }


class ExpertPlanner:
    """Deterministic geometric expert for waypoint-supervised flight data."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        """Initialize the planner."""
        self.config = config or PlannerConfig()

    def plan_waypoint(
        self,
        state: np.ndarray,
        goal_position: np.ndarray,
        task: BaseTask,
        scene_bbox: np.ndarray,
        obs_dt: float,
    ) -> tuple[np.ndarray, dict[str, float | bool | list[float]]]:
        """Plan the next expert waypoint and safety labels.

        Args:
            state: Current drone state.
            goal_position: Current task goal position.
            task: Active task instance.
            scene_bbox: Scene bounding box as ``(2, 3)``.
            obs_dt: Environment observation step in seconds.

        Returns:
            Tuple of global expert waypoint ``[x, y, z, yaw]`` and safety label
            dictionary.
        """
        start = np.asarray(state[:3], dtype=np.float32)
        goal = np.asarray(goal_position, dtype=np.float32)
        obstacles = task.get_obstacles()

        if isinstance(task, DynamicFollowTask):
            future_goal = self._predict_dynamic_goal(task, obs_dt)
        else:
            future_goal = goal.astype(np.float32, copy=True)

        path = self._plan_path(start, future_goal, obstacles, scene_bbox)
        waypoint_xyz = self._select_lookahead_waypoint(start, path)
        heading_target = self._heading_target(waypoint_xyz, future_goal, start)
        yaw = float(math.atan2(heading_target[1] - start[1], heading_target[0] - start[0]))
        waypoint = np.concatenate(
            [waypoint_xyz.astype(np.float32), np.array([yaw], dtype=np.float32)]
        )

        min_clearance = self._min_clearance(start, obstacles, scene_bbox)
        path_progress = float(np.linalg.norm(goal - start) - np.linalg.norm(goal - waypoint_xyz))
        labels: dict[str, float | bool | list[float]] = {
            "collision_imminent": bool(min_clearance < self.config.imminent_clearance_m),
            "min_clearance_m": float(min_clearance),
            "recovery_required": bool(
                min_clearance < self.config.recovery_clearance_m or path_progress < -0.05
            ),
            "path_progress_m": float(path_progress),
            "goal_distance_m": float(np.linalg.norm(goal - start)),
            "future_target_position": future_goal.astype(np.float32).tolist(),
            "planner_path_xyz": path.astype(np.float32).tolist(),
        }
        return waypoint.astype(np.float32), labels

    def normalized_waypoint_action(self, state: np.ndarray, waypoint: np.ndarray) -> np.ndarray:
        """Convert a global waypoint into the env's normalized waypoint action.

        Args:
            state: Current drone state.
            waypoint: Global expert waypoint ``[x, y, z, yaw]``.

        Returns:
            Normalized action in ``[-1, 1]`` suitable for ``action_mode="waypoint"``.
        """
        delta_xyz = (waypoint[:3] - state[:3]) / np.array([1.5, 1.5, 1.0], dtype=np.float32)
        yaw_delta = self._wrap_angle(float(waypoint[3] - state[8])) / (np.pi / 4.0)
        action = np.concatenate(
            [delta_xyz.astype(np.float32), np.array([yaw_delta], dtype=np.float32)]
        )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _predict_dynamic_goal(self, task: DynamicFollowTask, obs_dt: float) -> np.ndarray:
        """Predict a short-horizon future goal for dynamic following tasks."""
        angle = float(
            task._angle
            + task.angular_velocity * obs_dt * self.config.dynamic_target_lookahead_steps
        )
        offset = np.array(
            [task.radius * np.cos(angle), task.radius * np.sin(angle), 0.0],
            dtype=np.float32,
        )
        return (task._center + offset).astype(np.float32)

    def _plan_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[Obstacle],
        scene_bbox: np.ndarray,
    ) -> np.ndarray:
        """Plan a collision-aware polyline from start to goal."""
        path_points: list[np.ndarray] = [start.astype(np.float32)]
        current = start.astype(np.float32)
        for obstacle in obstacles:
            if not self._segment_needs_detour(current, goal, obstacle):
                continue
            detour = self._detour_waypoint(current, goal, obstacle, scene_bbox)
            path_points.append(detour.astype(np.float32))
            current = detour.astype(np.float32)
        path_points.append(goal.astype(np.float32))
        polyline = np.stack(path_points).astype(np.float32)
        return self._densify(polyline, scene_bbox)

    def _densify(self, points: np.ndarray, scene_bbox: np.ndarray) -> np.ndarray:
        """Insert intermediate waypoints at a fixed spacing."""
        densified: list[np.ndarray] = [points[0].astype(np.float32)]
        for index in range(1, len(points)):
            start = points[index - 1]
            end = points[index]
            delta = end - start
            distance = float(np.linalg.norm(delta))
            if distance < 1e-6:
                continue
            steps = max(1, int(np.ceil(distance / self.config.waypoint_spacing_m)))
            for step in range(1, steps + 1):
                point = start + delta * (step / steps)
                clipped = np.clip(
                    point,
                    scene_bbox[0] + np.array([0.25, 0.25, 0.25], dtype=np.float32),
                    scene_bbox[1] - np.array([0.25, 0.25, 0.25], dtype=np.float32),
                )
                densified.append(clipped.astype(np.float32))
        return np.stack(densified).astype(np.float32)

    def _select_lookahead_waypoint(self, start: np.ndarray, path: np.ndarray) -> np.ndarray:
        """Pick the next path point at the configured lookahead distance."""
        if len(path) == 1:
            return path[0].astype(np.float32)
        for point in path[1:]:
            if float(np.linalg.norm(point - start)) >= self.config.lookahead_distance_m:
                return point.astype(np.float32)
        return path[-1].astype(np.float32)

    def _heading_target(
        self,
        waypoint_xyz: np.ndarray,
        future_goal: np.ndarray,
        start: np.ndarray,
    ) -> np.ndarray:
        """Return the position used to derive the expert yaw angle."""
        if float(np.linalg.norm(future_goal[:2] - start[:2])) > 1e-5:
            return future_goal.astype(np.float32)
        return waypoint_xyz.astype(np.float32)

    def _segment_needs_detour(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle: Obstacle,
    ) -> bool:
        """Return whether the direct segment passes too close to an obstacle."""
        if isinstance(obstacle, CylinderObstacle):
            distance = self._distance_segment_to_cylinder_xy(start, goal, obstacle)
            return bool(distance < obstacle.radius + self.config.obstacle_clearance_m)
        distance = self._distance_segment_to_box_xy(start, goal, obstacle)
        return bool(distance < self.config.obstacle_clearance_m)

    def _detour_waypoint(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle: Obstacle,
        scene_bbox: np.ndarray,
    ) -> np.ndarray:
        """Construct a simple obstacle-avoiding detour waypoint."""
        if isinstance(obstacle, CylinderObstacle):
            segment = goal[:2] - start[:2]
            norm = float(np.linalg.norm(segment))
            direction = (
                np.array([1.0, 0.0], dtype=np.float32)
                if norm < 1e-6
                else (segment / norm).astype(np.float32)
            )
            lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
            sign = 1.0 if float(np.dot(lateral, obstacle.center[:2] - start[:2])) < 0.0 else -1.0
            offset = lateral * sign * np.float32(obstacle.radius + self.config.obstacle_clearance_m)
            detour_xy = obstacle.center[:2] + offset
            detour_z = np.clip(
                start[2],
                scene_bbox[0, 2] + 0.3,
                scene_bbox[1, 2] - 0.3,
            )
            return np.array([detour_xy[0], detour_xy[1], detour_z], dtype=np.float32)

        center = (obstacle.min_corner + obstacle.max_corner) / 2.0
        sign = -1.0 if start[1] <= center[1] else 1.0
        detour_y = center[1] + sign * (
            (obstacle.max_corner[1] - obstacle.min_corner[1]) / 2.0
            + self.config.obstacle_clearance_m
        )
        detour_x = float(np.clip(center[0], scene_bbox[0, 0] + 0.5, scene_bbox[1, 0] - 0.5))
        detour_z = float(
            np.clip(
                start[2],
                scene_bbox[0, 2] + 0.3,
                scene_bbox[1, 2] - 0.3,
            )
        )
        return np.array([detour_x, detour_y, detour_z], dtype=np.float32)

    def _min_clearance(
        self,
        position: np.ndarray,
        obstacles: list[Obstacle],
        scene_bbox: np.ndarray,
    ) -> float:
        """Compute the minimum clearance to scene bounds and active obstacles."""
        clearances: list[float] = []
        clearances.extend(
            [
                float(position[0] - scene_bbox[0, 0]),
                float(position[1] - scene_bbox[0, 1]),
                float(position[2] - scene_bbox[0, 2]),
                float(scene_bbox[1, 0] - position[0]),
                float(scene_bbox[1, 1] - position[1]),
                float(scene_bbox[1, 2] - position[2]),
            ]
        )
        for obstacle in obstacles:
            if isinstance(obstacle, CylinderObstacle):
                horizontal = float(
                    np.linalg.norm(position[:2] - obstacle.center[:2]) - obstacle.radius
                )
                vertical = abs(float(position[2] - obstacle.center[2])) - obstacle.height / 2.0
                if vertical <= 0.0:
                    clearances.append(horizontal)
                else:
                    clearances.append(float(np.hypot(max(horizontal, 0.0), vertical)))
            else:
                delta = np.maximum(obstacle.min_corner - position, position - obstacle.max_corner)
                delta = np.maximum(delta, 0.0)
                inside_margin = np.min(
                    np.concatenate([position - obstacle.min_corner, obstacle.max_corner - position])
                )
                if inside_margin >= 0.0:
                    clearances.append(-float(inside_margin))
                else:
                    clearances.append(float(np.linalg.norm(delta)))
        return float(min(clearances))

    def _distance_segment_to_cylinder_xy(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle: CylinderObstacle,
    ) -> float:
        """Return XY distance from a segment to a cylindrical obstacle center."""
        point = obstacle.center[:2].astype(np.float32)
        return self._distance_point_to_segment_xy(point, start[:2], goal[:2])

    def _distance_segment_to_box_xy(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle: BoxObstacle,
    ) -> float:
        """Return approximate XY distance from a segment to a box footprint."""
        samples = np.linspace(0.0, 1.0, num=20, dtype=np.float32)
        segment_points = start[None, :2] + (goal[:2] - start[:2])[None, :] * samples[:, None]
        min_corner = obstacle.min_corner[:2]
        max_corner = obstacle.max_corner[:2]
        distances: list[float] = []
        for point in segment_points:
            delta = np.maximum(min_corner - point, point - max_corner)
            delta = np.maximum(delta, 0.0)
            if np.all((point >= min_corner) & (point <= max_corner)):
                distances.append(
                    -float(np.min(np.concatenate([point - min_corner, max_corner - point])))
                )
            else:
                distances.append(float(np.linalg.norm(delta)))
        return float(min(distances))

    def _distance_point_to_segment_xy(
        self,
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        """Return the Euclidean distance from a point to a segment in XY."""
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator < 1e-8:
            return float(np.linalg.norm(point - start))
        t_value = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
        projection = start + t_value * segment
        return float(np.linalg.norm(point - projection))

    def _wrap_angle(self, angle: float) -> float:
        """Wrap an angle to ``[-pi, pi]``."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
