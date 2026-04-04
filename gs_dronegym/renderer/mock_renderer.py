"""Deterministic CPU fallback renderer for GS-DroneGym.

The mock renderer provides synthetic RGB, depth, and alpha outputs without
requiring GPU hardware or the optional gsplat dependency. It is used for tests,
CI, and basic environment bring-up.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from gs_dronegym.renderer.camera_model import CameraModel

LOGGER = logging.getLogger(__name__)


class MockRenderer:
    """CPU-only renderer with deterministic pseudo-observations."""

    def __init__(
        self,
        scene_path: str | None = None,
        camera: CameraModel | None = None,
        device: str = "cpu",
        near_plane: float = 0.1,
        far_plane: float = 100.0,
    ) -> None:
        """Initialize the mock renderer.

        Args:
            scene_path: Unused placeholder for interface compatibility.
            camera: Camera model used for output shapes.
            device: Device string kept for interface compatibility.
            near_plane: Minimum rendered depth.
            far_plane: Maximum rendered depth.
        """
        del scene_path, device
        self.camera = camera or CameraModel()
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)

    def load_scene(self, scene_path: str | None) -> None:
        """Load a scene for interface compatibility.

        Args:
            scene_path: Scene path, ignored by the mock renderer.
        """
        del scene_path

    def render(self, w2c: np.ndarray) -> dict[str, np.ndarray]:
        """Render a deterministic pseudo-frame from a camera pose.

        Args:
            w2c: World-to-camera transform.

        Returns:
            Dictionary containing ``rgb``, ``depth``, and ``alpha`` outputs.
        """
        pose = np.asarray(w2c, dtype=np.float32)
        seed = self._pose_seed(pose)
        rng = np.random.default_rng(seed)

        height = self.camera.height
        width = self.camera.width
        cam_height = float(np.linalg.inv(pose)[2, 3])
        sky_weight = float(np.clip(cam_height / 5.0, 0.0, 1.0))

        sky = np.array([120.0, 180.0, 235.0], dtype=np.float32)
        ground = np.array([150.0, 105.0, 65.0], dtype=np.float32)
        tint = sky_weight * sky + (1.0 - sky_weight) * ground

        noise = rng.normal(loc=0.0, scale=22.0, size=(height, width, 3)).astype(np.float32)
        rgb = np.clip(tint.reshape(1, 1, 3) + noise, 0.0, 255.0).astype(np.uint8)

        y_grid = np.linspace(-1.0, 1.0, num=height, dtype=np.float32).reshape(height, 1)
        depth_base = np.maximum(cam_height - y_grid, self.near_plane)
        depth_noise = rng.normal(loc=0.0, scale=0.02, size=(height, width)).astype(np.float32)
        depth = np.clip(depth_base + depth_noise, self.near_plane, self.far_plane).astype(
            np.float32
        )
        alpha = np.ones((height, width), dtype=np.float32)

        return {"rgb": rgb, "depth": depth, "alpha": alpha}

    def render_batch(self, w2c_list: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Render a batch of pseudo-frames.

        Args:
            w2c_list: Sequence of world-to-camera transforms.

        Returns:
            List of render dictionaries.
        """
        return [self.render(w2c) for w2c in w2c_list]

    def _pose_seed(self, w2c: np.ndarray) -> int:
        """Derive a deterministic random seed from a camera pose.

        Args:
            w2c: World-to-camera transform.

        Returns:
            Deterministic seed integer.
        """
        digest = hashlib.sha256(np.round(w2c, 5).tobytes()).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)
