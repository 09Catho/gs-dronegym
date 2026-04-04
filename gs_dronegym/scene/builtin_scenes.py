"""Built-in public Gaussian splat scene registry for GS-DroneGym.

The registry contains hand-authored metadata for commonly used benchmark scenes
and integrates with :class:`gs_dronegym.scene.scene_loader.SceneLoader` to fetch
and validate scene assets lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gs_dronegym.scene.scene_loader import SceneLoader


@dataclass(frozen=True, slots=True)
class SceneInfo:
    """Metadata for a built-in Gaussian splat scene."""

    name: str
    url: str
    description: str
    bbox: np.ndarray
    suggested_start_pos: np.ndarray
    suggested_goal_pos: np.ndarray


BUILTIN_SCENES: dict[str, SceneInfo] = {
    "garden": SceneInfo(
        name="garden",
        url=(
            "https://huggingface.co/nerfbaselines/nerfbaselines/"
            "resolve/main/gaussian-splatting/mipnerf360/garden/point_cloud.ply?download=true"
        ),
        description="MipNeRF360 outdoor garden scene.",
        bbox=np.array([[-4.0, -4.0, 0.2], [4.0, 4.0, 4.5]], dtype=np.float32),
        suggested_start_pos=np.array([-2.5, 0.0, 1.5], dtype=np.float32),
        suggested_goal_pos=np.array([2.5, 0.0, 1.5], dtype=np.float32),
    ),
    "room": SceneInfo(
        name="room",
        url=(
            "https://huggingface.co/nerfbaselines/nerfbaselines/"
            "resolve/main/gaussian-splatting/mipnerf360/room/point_cloud.ply?download=true"
        ),
        description="MipNeRF360 indoor room scene.",
        bbox=np.array([[-3.0, -3.0, 0.2], [3.0, 3.0, 3.0]], dtype=np.float32),
        suggested_start_pos=np.array([-1.5, 0.0, 1.2], dtype=np.float32),
        suggested_goal_pos=np.array([1.5, 0.0, 1.2], dtype=np.float32),
    ),
    "bicycle": SceneInfo(
        name="bicycle",
        url=(
            "https://huggingface.co/nerfbaselines/nerfbaselines/"
            "resolve/main/gaussian-splatting/mipnerf360/bicycle/point_cloud.ply?download=true"
        ),
        description="MipNeRF360 bicycle scene with outdoor clutter.",
        bbox=np.array([[-4.5, -4.0, 0.2], [4.5, 4.0, 4.0]], dtype=np.float32),
        suggested_start_pos=np.array([-3.0, 0.0, 1.6], dtype=np.float32),
        suggested_goal_pos=np.array([3.0, 0.0, 1.6], dtype=np.float32),
    ),
    "truck": SceneInfo(
        name="truck",
        url=(
            "https://huggingface.co/nerfbaselines/nerfbaselines/"
            "resolve/main/gaussian-splatting/tanksandtemples/truck/point_cloud.ply?download=true"
        ),
        description="Tanks and Temples truck scene.",
        bbox=np.array([[-5.0, -5.0, 0.2], [5.0, 5.0, 4.5]], dtype=np.float32),
        suggested_start_pos=np.array([-3.5, 0.0, 1.6], dtype=np.float32),
        suggested_goal_pos=np.array([3.5, 0.0, 1.6], dtype=np.float32),
    ),
}


def get_scene(name: str) -> Path:
    """Download and return a built-in scene asset path.

    Args:
        name: Built-in scene name.

    Returns:
        Local path to the cached scene asset.

    Raises:
        KeyError: If the scene name is unknown.
    """
    if name not in BUILTIN_SCENES:
        raise KeyError(f"Unknown built-in scene: {name}")
    loader = SceneLoader()
    return loader.load(BUILTIN_SCENES[name].url)


def list_scenes() -> list[str]:
    """List all available built-in scene names.

    Returns:
        Sorted list of scene names.
    """
    return sorted(BUILTIN_SCENES.keys())
