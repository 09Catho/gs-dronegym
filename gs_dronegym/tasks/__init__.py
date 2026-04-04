"""Benchmark navigation tasks for GS-DroneGym."""

from gs_dronegym.tasks.base_task import BaseTask, TaskConfig
from gs_dronegym.tasks.dynamic_follow import DynamicFollowTask
from gs_dronegym.tasks.narrow_corridor import NarrowCorridorTask
from gs_dronegym.tasks.object_nav import ObjectNavTask
from gs_dronegym.tasks.obstacle_slalom import ObstacleSlalomTask
from gs_dronegym.tasks.point_nav import PointNavTask

__all__ = [
    "BaseTask",
    "DynamicFollowTask",
    "NarrowCorridorTask",
    "ObjectNavTask",
    "ObstacleSlalomTask",
    "PointNavTask",
    "TaskConfig",
]
