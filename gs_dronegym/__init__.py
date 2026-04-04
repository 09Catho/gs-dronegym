"""Top-level package exports and environment registration for GS-DroneGym.

GS-DroneGym provides photorealistic drone navigation environments for
vision-language-action research. This module exposes the public factory API,
registers built-in Gymnasium environments, and re-exports the main classes
used in experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register, registry

from gs_dronegym.benchmarks import list_benchmarks, make_benchmark
from gs_dronegym.data import (
    ActionSpec,
    BenchmarkReport,
    ObservationSpec,
    TaskSpec,
    TrajectoryEpisode,
    TrajectoryStep,
    load_dataset,
)
from gs_dronegym.env.drone_env import GSDroneEnv
from gs_dronegym.scene.builtin_scenes import SceneInfo, get_scene, list_scenes
from gs_dronegym.tasks import (
    BaseTask,
    DynamicFollowTask,
    NarrowCorridorTask,
    ObjectNavTask,
    ObstacleSlalomTask,
    PointNavTask,
    TaskConfig,
)
from gs_dronegym.utils.metrics import BenchmarkResult, Episode

__all__ = [
    "ActionSpec",
    "BaseTask",
    "BenchmarkReport",
    "BenchmarkResult",
    "DynamicFollowTask",
    "Episode",
    "GSDroneEnv",
    "NarrowCorridorTask",
    "ObservationSpec",
    "ObjectNavTask",
    "ObstacleSlalomTask",
    "PointNavTask",
    "SceneInfo",
    "TaskSpec",
    "TaskConfig",
    "TrajectoryEpisode",
    "TrajectoryStep",
    "__version__",
    "get_scene",
    "list_benchmarks",
    "load_dataset",
    "list_scenes",
    "make",
    "make_benchmark",
]

__version__ = "0.2.0"

LOGGER = logging.getLogger(__name__)

_REGISTERED_ENVS: dict[str, type[BaseTask]] = {
    "PointNav-v0": PointNavTask,
    "ObjectNav-v0": ObjectNavTask,
    "ObstacleSlalom-v0": ObstacleSlalomTask,
    "DynamicFollow-v0": DynamicFollowTask,
    "NarrowCorridor-v0": NarrowCorridorTask,
}


def _create_env(task_cls: type[BaseTask], **kwargs: object) -> GSDroneEnv:
    """Create a registered environment instance.

    Args:
        task_cls: Task class to instantiate.
        **kwargs: Environment keyword arguments. A ``task`` instance may be
            supplied to override the default task construction.

    Returns:
        A configured :class:`GSDroneEnv`.
    """
    task = kwargs.pop("task", None)
    if task is None:
        task = task_cls()
    scene = kwargs.pop("scene", None)
    if isinstance(scene, str) and scene in list_scenes():
        scene = get_scene(scene)
    elif isinstance(scene, Path):
        scene = scene
    return GSDroneEnv(task=task, scene_path=scene, **kwargs)


def _register_envs() -> None:
    """Register built-in environments once with Gymnasium."""
    for env_id, task_cls in _REGISTERED_ENVS.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="gs_dronegym:_create_env",
            kwargs={"task_cls": task_cls},
        )


def make(env_id: str, scene: str | Path | None = None, **kwargs: object) -> gym.Env:
    """Construct a built-in GS-DroneGym environment.

    Args:
        env_id: Registered Gymnasium environment ID.
        scene: Scene name or local/remote Gaussian scene path.
        **kwargs: Additional environment configuration forwarded to the
            registered environment constructor.

    Returns:
        The created Gymnasium environment.

    Raises:
        gym.error.Error: If the environment ID is unknown.
    """
    _register_envs()
    kwargs["scene"] = scene
    LOGGER.debug("Creating environment %s with kwargs=%s", env_id, kwargs)
    return gym.make(env_id, **kwargs)


_register_envs()
