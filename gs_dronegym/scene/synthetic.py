"""Generate small synthetic Gaussian scenes with known ground-truth geometry.

Real Gaussian reconstructions are large and rarely redistributable, which makes
them unusable as test fixtures. A synthetic room built from primitives exercises
the same PLY parsing, occupancy extraction and rendering code paths while having
geometry that is known exactly, so occupancy can be checked against truth rather
than against another estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

LOGGER = logging.getLogger(__name__)

#: Logit opacity corresponding to an almost fully opaque Gaussian.
_OPAQUE_LOGIT = np.float32(5.0)


@dataclass(slots=True)
class SyntheticRoomConfig:
    """Geometry of a synthetic rectangular room with one central pillar."""

    size: tuple[float, float, float] = (10.0, 8.0, 3.0)
    spacing: float = 0.1
    pillar_center: tuple[float, float] = (0.0, 0.0)
    pillar_half_extent: float = 0.5
    include_ceiling: bool = False


def _surface_grid(
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a regular grid over a rectangular surface.

    Args:
        u_range: Inclusive range along the first axis.
        v_range: Inclusive range along the second axis.
        spacing: Sample spacing.

    Returns:
        Flattened coordinate arrays for the two axes.
    """
    u = np.arange(u_range[0], u_range[1] + 0.5 * spacing, spacing, dtype=np.float32)
    v = np.arange(v_range[0], v_range[1] + 0.5 * spacing, spacing, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    return grid_u.reshape(-1), grid_v.reshape(-1)


def synthetic_room_points(config: SyntheticRoomConfig | None = None) -> np.ndarray:
    """Build surface sample points for a synthetic room.

    The room is centred on the origin in x and y, with its floor at ``z = 0``.

    Args:
        config: Optional room configuration.

    Returns:
        Array of surface points with shape ``(n, 3)``.
    """
    room = config or SyntheticRoomConfig()
    size_x, size_y, size_z = room.size
    half_x, half_y = 0.5 * size_x, 0.5 * size_y
    spacing = room.spacing
    parts: list[np.ndarray] = []

    grid_x, grid_y = _surface_grid((-half_x, half_x), (-half_y, half_y), spacing)
    parts.append(np.stack([grid_x, grid_y, np.zeros_like(grid_x)], axis=1))
    if room.include_ceiling:
        parts.append(np.stack([grid_x, grid_y, np.full_like(grid_x, size_z)], axis=1))

    grid_x, grid_z = _surface_grid((-half_x, half_x), (0.0, size_z), spacing)
    for y_value in (-half_y, half_y):
        parts.append(np.stack([grid_x, np.full_like(grid_x, y_value), grid_z], axis=1))

    grid_y, grid_z = _surface_grid((-half_y, half_y), (0.0, size_z), spacing)
    for x_value in (-half_x, half_x):
        parts.append(np.stack([np.full_like(grid_y, x_value), grid_y, grid_z], axis=1))

    pillar_x, pillar_y = room.pillar_center
    half = room.pillar_half_extent
    grid_v, grid_z = _surface_grid((-half, half), (0.0, size_z), spacing)
    for offset in (-half, half):
        parts.append(
            np.stack(
                [np.full_like(grid_v, pillar_x + offset), grid_v + pillar_y, grid_z], axis=1
            )
        )
        parts.append(
            np.stack(
                [grid_v + pillar_x, np.full_like(grid_v, pillar_y + offset), grid_z], axis=1
            )
        )

    return np.concatenate(parts, axis=0).astype(np.float32)


def write_synthetic_room_ply(
    path: str | Path,
    config: SyntheticRoomConfig | None = None,
) -> Path:
    """Write a synthetic room as a Gaussian splat PLY file.

    Opacity is stored as a logit and scale as a logarithm, matching the original
    3D Gaussian Splatting export convention, so the file loads through the same
    code path as a real reconstruction.

    Args:
        path: Destination ``.ply`` path.
        config: Optional room configuration.

    Returns:
        Path to the written file.
    """
    room = config or SyntheticRoomConfig()
    points = synthetic_room_points(room)
    n_points = int(points.shape[0])
    log_scale = np.float32(np.log(max(room.spacing, 1e-6) * 0.5))

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    vertex = np.zeros(n_points, dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["opacity"] = _OPAQUE_LOGIT
    for axis in range(3):
        vertex[f"scale_{axis}"] = log_scale
    vertex["rot_0"] = np.float32(1.0)

    height_fraction = points[:, 2] / max(room.size[2], 1e-6)
    vertex["f_dc_0"] = (0.5 + 0.5 * height_fraction).astype(np.float32)
    vertex["f_dc_1"] = np.float32(0.4)
    vertex["f_dc_2"] = (1.0 - 0.5 * height_fraction).astype(np.float32)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output_path))
    LOGGER.info("Wrote synthetic Gaussian room with %d Gaussians to %s", n_points, output_path)
    return output_path
