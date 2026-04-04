"""Scene loading and built-in public Gaussian splat registry."""

from gs_dronegym.scene.builtin_scenes import BUILTIN_SCENES, SceneInfo, get_scene, list_scenes
from gs_dronegym.scene.scene_loader import SceneLoader, SceneValidationError

__all__ = [
    "BUILTIN_SCENES",
    "SceneInfo",
    "SceneLoader",
    "SceneValidationError",
    "get_scene",
    "list_scenes",
]
