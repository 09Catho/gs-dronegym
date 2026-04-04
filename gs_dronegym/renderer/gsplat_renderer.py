"""3D Gaussian Splatting renderer integration for GS-DroneGym.

This module bridges Gaussian splat scene files to rendered RGB/depth
observations using the optional ``gsplat`` Python package. When ``gsplat`` is
unavailable, the class transparently falls back to :class:`MockRenderer`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from gs_dronegym.renderer.camera_model import CameraModel
from gs_dronegym.renderer.mock_renderer import MockRenderer
from gs_dronegym.scene.scene_loader import SceneLoader

LOGGER = logging.getLogger(__name__)

try:
    from gsplat import rasterization

    _GSPLAT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency path
    rasterization = None
    _GSPLAT_AVAILABLE = False


class GSplatRenderer:
    """Render Gaussian splat scenes with optional ``gsplat`` acceleration."""

    def __init__(
        self,
        scene_path: str | Path,
        camera: CameraModel,
        device: str = "cuda",
        near_plane: float = 0.1,
        far_plane: float = 100.0,
    ) -> None:
        """Initialize the renderer and load a Gaussian scene.

        Args:
            scene_path: Path to a Gaussian splat ``.ply`` file.
            camera: Camera model used to build rendering matrices.
            device: Torch device string.
            near_plane: Near clipping plane in meters.
            far_plane: Far clipping plane in meters.
        """
        self.camera = camera
        self.device = device
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.loader = SceneLoader()
        self.fallback = MockRenderer(
            scene_path=str(scene_path),
            camera=camera,
            device="cpu",
            near_plane=near_plane,
            far_plane=far_plane,
        )

        self.scene_path = Path(scene_path)
        self.means = torch.empty((0, 3), dtype=torch.float32)
        self.quats = torch.empty((0, 4), dtype=torch.float32)
        self.scales = torch.empty((0, 3), dtype=torch.float32)
        self.opacities = torch.empty((0,), dtype=torch.float32)
        self.colors = torch.empty((0, 3), dtype=torch.float32)
        self.sh_degree: int | None = None
        self._backend_failed = False

        if not _GSPLAT_AVAILABLE:
            LOGGER.warning(
                "gsplat is not installed. Falling back to MockRenderer for scene %s.",
                scene_path,
            )
        self.load_scene(scene_path)

    def load_scene(self, scene_path: str | Path) -> None:
        """Load Gaussian parameters from a scene file.

        Args:
            scene_path: Local path or URL pointing to a ``.ply`` scene file.
        """
        local_path = self.loader.load(scene_path)
        self.scene_path = local_path
        if not _GSPLAT_AVAILABLE:
            self.fallback.load_scene(str(local_path))
            return

        ply_data = PlyData.read(str(local_path))
        vertex = ply_data["vertex"].data
        means = np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float32),
                np.asarray(vertex["y"], dtype=np.float32),
                np.asarray(vertex["z"], dtype=np.float32),
            ],
            axis=1,
        )

        scales = np.stack(
            [
                np.asarray(vertex["scale_0"], dtype=np.float32),
                np.asarray(vertex["scale_1"], dtype=np.float32),
                np.asarray(vertex["scale_2"], dtype=np.float32),
            ],
            axis=1,
        )
        quats = np.stack(
            [
                np.asarray(vertex["rot_0"], dtype=np.float32),
                np.asarray(vertex["rot_1"], dtype=np.float32),
                np.asarray(vertex["rot_2"], dtype=np.float32),
                np.asarray(vertex["rot_3"], dtype=np.float32),
            ],
            axis=1,
        )
        quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.clip(quat_norm, 1e-8, None)
        opacity_raw = np.asarray(vertex["opacity"], dtype=np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacity_raw))

        properties = [prop.name for prop in ply_data["vertex"].properties]
        dc = np.stack(
            [
                np.asarray(vertex["f_dc_0"], dtype=np.float32),
                np.asarray(vertex["f_dc_1"], dtype=np.float32),
                np.asarray(vertex["f_dc_2"], dtype=np.float32),
            ],
            axis=1,
        )
        rest_names = sorted(
            [name for name in properties if name.startswith("f_rest_")],
            key=lambda item: int(item.split("_")[-1]),
        )
        if rest_names and len(rest_names) % 3 == 0:
            rest = np.stack(
                [np.asarray(vertex[name], dtype=np.float32) for name in rest_names],
                axis=1,
            )
            n_coeff_minus_one = len(rest_names) // 3
            rest = rest.reshape(rest.shape[0], n_coeff_minus_one, 3).astype(np.float32)
            colors = np.concatenate([dc[:, None, :], rest], axis=1).astype(np.float32)
            self.sh_degree = max(0, int(np.sqrt(colors.shape[1]) - 1))
        else:
            colors = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0).astype(np.float32)
            self.sh_degree = None

        target_device = self._resolved_device()
        self.means = torch.from_numpy(means).to(device=target_device, dtype=torch.float32)
        self.quats = torch.from_numpy(quats).to(device=target_device, dtype=torch.float32)
        self.scales = torch.from_numpy(scales).to(device=target_device, dtype=torch.float32)
        self.opacities = torch.from_numpy(opacities).to(device=target_device, dtype=torch.float32)
        self.colors = torch.from_numpy(colors).to(device=target_device, dtype=torch.float32)
        LOGGER.info("Loaded %d gaussians from %s", means.shape[0], local_path)

    def render(self, w2c: np.ndarray) -> dict[str, np.ndarray]:
        """Render a single camera view.

        Args:
            w2c: World-to-camera transform of shape ``(4, 4)``.

        Returns:
            Dictionary containing ``rgb``, ``depth``, and ``alpha``.
        """
        if not _GSPLAT_AVAILABLE or self._backend_failed:
            return self.fallback.render(w2c)

        device = self._resolved_device()
        viewmat = torch.from_numpy(np.asarray(w2c, dtype=np.float32)).unsqueeze(0).to(device)
        k_matrix = torch.from_numpy(self.camera.get_intrinsics()).unsqueeze(0).to(device)
        try:
            renders, alphas, meta = rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=viewmat,
                Ks=k_matrix,
                width=self.camera.width,
                height=self.camera.height,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                render_mode="RGB+D",
                sh_degree=self.sh_degree,
            )
        except Exception as exc:  # pragma: no cover - backend-dependent
            LOGGER.warning(
                "gsplat backend failed for scene %s (%s). Falling back to MockRenderer.",
                self.scene_path,
                exc,
            )
            self._backend_failed = True
            return self.fallback.render(w2c)
        del meta
        render = renders[0].detach().cpu().numpy().astype(np.float32)
        alpha = np.squeeze(alphas[0].detach().cpu().numpy()).astype(np.float32)
        rgb = np.clip(render[..., :3], 0.0, 1.0)
        depth = np.clip(render[..., 3], self.near_plane, self.far_plane).astype(np.float32)
        return {
            "rgb": (rgb * 255.0).astype(np.uint8),
            "depth": depth,
            "alpha": alpha,
        }

    def render_batch(self, w2c_list: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Render a batch of camera poses.

        Args:
            w2c_list: List of world-to-camera transforms.

        Returns:
            List of render dictionaries.
        """
        if not _GSPLAT_AVAILABLE:
            return self.fallback.render_batch(w2c_list)
        return [self.render(w2c) for w2c in w2c_list]

    def _resolved_device(self) -> str:
        """Resolve the torch device used for rendering.

        Returns:
            Safe torch device string.
        """
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested for gsplat rendering but not available; using CPU.")
            return "cpu"
        return self.device
