"""Scene loading, caching, and validation for Gaussian splat scenes.

This module centralizes scene IO so environments can work with local files,
downloadable assets, and validation errors consistently across the package.
"""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from plyfile import PlyData

LOGGER = logging.getLogger(__name__)


class SceneValidationError(RuntimeError):
    """Raised when a Gaussian splat scene file is malformed."""


class SceneLoader:
    """Loader for local and remote Gaussian splat scene files."""

    CACHE_DIR = Path.home() / ".gs_dronegym" / "scenes"
    REQUIRED_PROPERTIES = {"x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0"}

    def load(self, path: str | Path) -> Path:
        """Load a local or remote scene into the cache.

        Args:
            path: Local path or HTTP(S) URL.

        Returns:
            Local path to the validated cached scene.

        Raises:
            FileNotFoundError: If the local file does not exist.
            SceneValidationError: If the scene is missing required properties.
        """
        source = str(path)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if source.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(source)
            filename = Path(parsed.path).name or "scene.ply"
            local_path = self.CACHE_DIR / filename
            if not local_path.exists():
                LOGGER.info("Downloading scene asset from %s", source)
                urllib.request.urlretrieve(source, local_path)
            path_obj = local_path
        else:
            path_obj = Path(source).expanduser().resolve()
            if not path_obj.exists():
                raise FileNotFoundError(f"Scene file not found: {path_obj}")

        self.validate_ply(path_obj)
        return path_obj

    def validate_ply(self, path: Path) -> bool:
        """Validate required Gaussian splat vertex properties.

        Args:
            path: Path to a ``.ply`` file.

        Returns:
            ``True`` if validation passes.

        Raises:
            SceneValidationError: If the file is malformed.
        """
        if path.suffix.lower() != ".ply":
            raise SceneValidationError(f"Expected a .ply file, received: {path}")

        try:
            ply_data = PlyData.read(str(path))
        except Exception as exc:  # pragma: no cover - delegated to library
            raise SceneValidationError(f"Failed to parse PLY file {path}: {exc}") from exc

        if "vertex" not in ply_data:
            raise SceneValidationError(f"PLY file {path} does not contain a vertex element.")
        vertex = ply_data["vertex"]
        properties = {prop.name for prop in vertex.properties}
        missing = sorted(self.REQUIRED_PROPERTIES - properties)
        if missing:
            raise SceneValidationError(
                f"PLY file {path} is missing required Gaussian properties: {missing}"
            )
        return True

    def infer_bbox(self, path: Path) -> np.ndarray:
        """Infer a scene bounding box from vertex positions.

        Args:
            path: Path to a validated PLY scene.

        Returns:
            Scene bounding box as a ``(2, 3)`` float32 array.
        """
        ply_data = PlyData.read(str(path))
        vertex = ply_data["vertex"].data
        xyz = np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float32),
                np.asarray(vertex["y"], dtype=np.float32),
                np.asarray(vertex["z"], dtype=np.float32),
            ],
            axis=1,
        )
        min_corner = np.min(xyz, axis=0).astype(np.float32)
        max_corner = np.max(xyz, axis=0).astype(np.float32)
        return np.stack([min_corner, max_corner]).astype(np.float32)
