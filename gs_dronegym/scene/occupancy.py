"""Derive navigation bounds and collision geometry from a Gaussian scene.

Task obstacles have historically been hand-authored primitives with no
relationship to the Gaussians that produce the rendered image, so a policy could
fly through a visible wall or collide with empty space. This module builds a
voxel occupancy grid directly from the scene's own Gaussians, so that geometry
and pixels describe the same world.

Gaussian PLY files store opacity as a logit and scales as logarithms, matching
the original 3D Gaussian Splatting export convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

LOGGER = logging.getLogger(__name__)

#: Gaussians fainter than this contribute nothing solid and are discarded.
DEFAULT_OPACITY_THRESHOLD = 0.5

#: Percentile clip used to reject reconstruction floaters when bounding a scene.
DEFAULT_BOUNDS_PERCENTILE = 1.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    Args:
        values: Input array.

    Returns:
        Element-wise logistic of the input.
    """
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0))),
        np.exp(np.clip(values, -60.0, 60.0)) / (1.0 + np.exp(np.clip(values, -60.0, 60.0))),
    ).astype(np.float32)


@dataclass(slots=True)
class GaussianCloud:
    """Minimal Gaussian scene attributes needed for geometry extraction."""

    positions: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray

    @property
    def n_gaussians(self) -> int:
        """Return the number of Gaussians in the cloud.

        Returns:
            Gaussian count.
        """
        return int(self.positions.shape[0])


def load_gaussian_cloud(path: str | Path) -> GaussianCloud:
    """Read positions, activated opacities and activated scales from a PLY.

    Args:
        path: Path to a Gaussian splat PLY file.

    Returns:
        Parsed Gaussian cloud with activations applied.

    Raises:
        ValueError: If required Gaussian properties are absent.
    """
    ply_data = PlyData.read(str(path))
    vertex = ply_data["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    missing = required - names
    if missing:
        raise ValueError(f"Gaussian PLY {path} is missing properties: {sorted(missing)}")

    positions = np.stack(
        [np.asarray(vertex[axis], dtype=np.float32) for axis in ("x", "y", "z")],
        axis=1,
    )
    opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    scales = np.exp(
        np.stack(
            [np.asarray(vertex[f"scale_{i}"], dtype=np.float32) for i in range(3)],
            axis=1,
        )
    ).astype(np.float32)
    return GaussianCloud(positions=positions, opacities=opacities, scales=scales)


@dataclass(slots=True)
class OccupancyGrid:
    """Voxel occupancy derived from a Gaussian scene."""

    origin: np.ndarray
    voxel_size: float
    occupied: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the voxel grid shape.

        Returns:
            Grid dimensions along x, y and z.
        """
        dims = self.occupied.shape
        return (int(dims[0]), int(dims[1]), int(dims[2]))

    @property
    def bounds(self) -> np.ndarray:
        """Return the world-space bounds spanned by the grid.

        Returns:
            Bounds as a ``(2, 3)`` float32 array.
        """
        extent = np.asarray(self.shape, dtype=np.float32) * np.float32(self.voxel_size)
        return np.stack([self.origin, self.origin + extent]).astype(np.float32)

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        """Convert world points into integer voxel indices.

        Args:
            points: Array of shape ``(n, 3)``.

        Returns:
            Integer indices of shape ``(n, 3)``, unclipped.
        """
        query = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return np.floor((query - self.origin) / np.float32(self.voxel_size)).astype(np.int64)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return whether points fall inside the grid extent.

        Args:
            points: Array of shape ``(n, 3)``.

        Returns:
            Boolean mask of shape ``(n,)``.
        """
        index = self.world_to_index(points)
        shape = np.asarray(self.shape, dtype=np.int64)
        return np.all((index >= 0) & (index < shape), axis=1)

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        """Query occupancy at world points.

        Points outside the grid are reported unoccupied; the environment's own
        bounds check is responsible for out-of-scene positions.

        Args:
            points: Array of shape ``(n, 3)``.

        Returns:
            Boolean mask of shape ``(n,)``.
        """
        query = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        inside = self.contains(query)
        result = np.zeros(query.shape[0], dtype=bool)
        if not np.any(inside):
            return result
        index = self.world_to_index(query[inside])
        result[inside] = self.occupied[index[:, 0], index[:, 1], index[:, 2]]
        return result

    def occupancy_fraction(self) -> float:
        """Return the fraction of voxels marked occupied.

        Returns:
            Occupied fraction in ``[0, 1]``.
        """
        return float(np.count_nonzero(self.occupied) / max(self.occupied.size, 1))


def derive_scene_bounds(
    cloud: GaussianCloud,
    percentile: float = DEFAULT_BOUNDS_PERCENTILE,
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD,
) -> np.ndarray:
    """Estimate robust navigation bounds for a Gaussian scene.

    Raw min/max over Gaussian positions is dominated by reconstruction floaters,
    which produce enormous, mostly empty bounds. Clipping to a percentile of the
    solid Gaussians gives bounds that describe the reconstructed room.

    Args:
        cloud: Parsed Gaussian cloud.
        percentile: Lower percentile to clip at; the upper clip is its mirror.
        opacity_threshold: Minimum activated opacity treated as solid.

    Returns:
        Bounds as a ``(2, 3)`` float32 array.

    Raises:
        ValueError: If no Gaussian passes the opacity threshold.
    """
    solid = cloud.positions[cloud.opacities >= np.float32(opacity_threshold)]
    if solid.size == 0:
        raise ValueError(
            f"No Gaussian exceeded opacity threshold {opacity_threshold}; "
            "the scene may use a different opacity convention."
        )
    lower = np.percentile(solid, percentile, axis=0).astype(np.float32)
    upper = np.percentile(solid, 100.0 - percentile, axis=0).astype(np.float32)
    return np.stack([lower, upper]).astype(np.float32)


def _dilate(occupied: np.ndarray, iterations: int) -> np.ndarray:
    """Grow occupied voxels by a 6-connected structuring element.

    Args:
        occupied: Boolean occupancy volume.
        iterations: Number of dilation passes.

    Returns:
        Dilated occupancy volume.
    """
    grown = occupied
    for _ in range(max(0, iterations)):
        neighbours = grown.copy()
        for axis in range(3):
            for shift in (-1, 1):
                neighbours |= np.roll(grown, shift, axis=axis)
        grown = neighbours
    return grown


def _sphere_offsets(radius: int) -> np.ndarray:
    """Return integer voxel offsets within a given radius.

    Args:
        radius: Radius in voxels.

    Returns:
        Array of offsets with shape ``(n, 3)``.
    """
    if radius <= 0:
        return np.zeros((1, 3), dtype=np.int64)
    span = np.arange(-radius, radius + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[np.sum(grid.astype(np.float64) ** 2, axis=1) <= float(radius) ** 2]


def build_occupancy_grid(
    cloud: GaussianCloud,
    voxel_size: float = 0.1,
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD,
    bounds: np.ndarray | None = None,
    bounds_percentile: float = DEFAULT_BOUNDS_PERCENTILE,
    min_density: float = DEFAULT_OPACITY_THRESHOLD,
    dilation_voxels: int = 0,
    extent_sigma: float = 1.0,
    max_radius_voxels: int = 3,
) -> OccupancyGrid:
    """Voxelize a Gaussian cloud into an occupancy grid.

    Args:
        cloud: Parsed Gaussian cloud.
        voxel_size: Edge length of one voxel in scene units.
        opacity_threshold: Minimum activated opacity treated as solid.
        bounds: Optional explicit ``(2, 3)`` bounds; derived when omitted.
        bounds_percentile: Percentile clip used when deriving bounds.
        min_density: Accumulated opacity required to mark a voxel occupied.
            The default admits a voxel holding a single solid Gaussian.
        dilation_voxels: Optional dilation passes, used to close thin surfaces.
        extent_sigma: Gaussian extent, in standard deviations, splatted into
            the grid. Larger values close gaps between sparse Gaussians at
            the cost of thickening surfaces.
        max_radius_voxels: Upper bound on the splat radius, which keeps the
            cost bounded when a scene contains very large Gaussians.

    Returns:
        Occupancy grid covering the scene bounds.

    Raises:
        ValueError: If ``voxel_size`` is not positive.
    """
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}.")

    solid_mask = cloud.opacities >= np.float32(opacity_threshold)
    positions = cloud.positions[solid_mask]
    opacities = cloud.opacities[solid_mask]

    extent = (
        derive_scene_bounds(cloud, bounds_percentile, opacity_threshold)
        if bounds is None
        else np.asarray(bounds, dtype=np.float32).reshape(2, 3)
    )
    origin = extent[0].astype(np.float32)
    span = np.maximum(extent[1] - extent[0], np.float32(voxel_size))
    # One extra voxel so Gaussians lying exactly on the upper bound are inside.
    dims = np.maximum(np.ceil(span / np.float32(voxel_size)).astype(np.int64) + 1, 1)

    density = np.zeros(tuple(int(d) for d in dims), dtype=np.float32)
    if positions.size:
        index = np.floor((positions - origin) / np.float32(voxel_size)).astype(np.int64)
        inside = np.all((index >= 0) & (index < dims), axis=1)
        index = index[inside]
        weights = opacities[inside]
        scales = cloud.scales[solid_mask][inside]

        # A Gaussian covers the voxels within its own extent, not just the one
        # holding its centre. Without this, a scene whose Gaussians are spaced
        # more widely than the voxel size yields dotted, permeable surfaces.
        radii = np.ceil(
            (np.float32(extent_sigma) * np.max(scales, axis=1)) / np.float32(voxel_size)
        ).astype(np.int64)
        radii = np.clip(radii, 0, int(max_radius_voxels))

        for radius in np.unique(radii):
            selected = index[radii == radius]
            selected_weights = weights[radii == radius]
            for offset in _sphere_offsets(int(radius)):
                shifted = selected + offset
                valid = np.all((shifted >= 0) & (shifted < dims), axis=1)
                if not np.any(valid):
                    continue
                target = shifted[valid]
                np.add.at(
                    density,
                    (target[:, 0], target[:, 1], target[:, 2]),
                    selected_weights[valid],
                )

    occupied = density >= np.float32(min_density)
    if dilation_voxels > 0:
        occupied = _dilate(occupied, dilation_voxels)

    LOGGER.info(
        "Built occupancy grid %s at %.3f m/voxel; %.2f%% occupied from %d solid Gaussians.",
        tuple(int(d) for d in dims),
        voxel_size,
        100.0 * float(np.count_nonzero(occupied)) / max(occupied.size, 1),
        int(positions.shape[0]),
    )
    return OccupancyGrid(origin=origin, voxel_size=float(voxel_size), occupied=occupied)
