"""Plot the scene-derived collision geometry used in the README.

Builds the synthetic Gaussian room, loads it into the environment so that the
occupancy grid is derived exactly as it is for any Gaussian scene, and flies the
drone straight at the central pillar until that derived geometry reports a
collision. Runs on CPU.

Run with:
    python tools/make_scene_geometry_figure.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import gs_dronegym
from gs_dronegym.scene.occupancy import GaussianCloud, OccupancyGrid, load_gaussian_cloud
from gs_dronegym.scene.synthetic import SyntheticRoomConfig, write_synthetic_room_ply

OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "scene_geometry.png"
ROOM = SyntheticRoomConfig(size=(10.0, 8.0, 3.0), spacing=0.1)
FLIGHT_HEIGHT_M = 1.5


def fly_into_pillar(ply_path: Path) -> tuple[np.ndarray, bool, OccupancyGrid]:
    """Fly forward from the room edge towards the pillar until a collision.

    Args:
        ply_path: Path to the synthetic Gaussian room.

    Returns:
        Flown positions, whether a collision was reported, and the occupancy
        grid the environment derived from the scene.

    Raises:
        RuntimeError: If the environment did not derive occupancy.
    """
    env = gs_dronegym.make(
        "PointNav-v0",
        scene=str(ply_path),
        renderer_device="cpu",
        observation_mode="state",
        action_mode="waypoint",
    )
    inner = env.unwrapped
    inner.reset(seed=0)
    if inner.occupancy is None:
        raise RuntimeError("environment did not derive occupancy from the scene")
    inner.dynamics.state[:] = 0.0
    inner.dynamics.state[:3] = (-3.5, 0.0, FLIGHT_HEIGHT_M)

    forward = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    positions = [inner.dynamics.get_state()[:3].copy()]
    collided = False
    for _ in range(120):
        _, _, terminated, truncated, info = inner.step(forward)
        positions.append(inner.dynamics.get_state()[:3].copy())
        if info["collision"]:
            collided = True
            break
        if terminated or truncated:
            break
    grid = inner.occupancy
    env.close()
    return np.stack(positions), collided, grid


def plot(cloud: GaussianCloud, grid: OccupancyGrid, path: np.ndarray, collided: bool) -> None:
    """Draw the Gaussian top view beside the derived occupancy slice.

    Args:
        cloud: Parsed Gaussian cloud.
        grid: Occupancy grid derived by the environment.
        path: Flown drone positions.
        collided: Whether the flight ended in a reported collision.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solid = cloud.positions[cloud.opacities >= 0.5]
    walls = solid[solid[:, 2] > 0.05]
    sample = walls[:: max(1, len(walls) // 8000)]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    left.scatter(sample[:, 0], sample[:, 1], s=1.0, c=sample[:, 2], cmap="viridis")
    left.set_title(f"Scene: {cloud.n_gaussians:,} Gaussians (top view, floor hidden)")

    layer = int((FLIGHT_HEIGHT_M - float(grid.origin[2])) / grid.voxel_size)
    bounds = grid.bounds
    right.imshow(
        grid.occupied[:, :, layer].T,
        origin="lower",
        extent=(bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1]),
        cmap="Greys",
        interpolation="nearest",
    )
    right.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=2.0, label="drone path")
    if collided:
        right.scatter(
            path[-1, 0], path[-1, 1], marker="x", s=140, c="tab:red", zorder=3, label="collision"
        )
    right.set_title(f"Derived occupancy at z = {FLIGHT_HEIGHT_M} m, {grid.voxel_size} m voxels")
    right.legend(loc="upper right")

    for axis in (left, right):
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    figure.suptitle("Collision geometry is derived from the scene's own Gaussians")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Build the scene, fly the drone and write the figure."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ply_path = write_synthetic_room_ply(Path(tmp) / "room.ply", ROOM)
        cloud = load_gaussian_cloud(ply_path)
        path, collided, grid = fly_into_pillar(ply_path)
    plot(cloud, grid, path, collided)
    print(
        f"flew {len(path) - 1} steps from x={path[0, 0]:.2f} to x={path[-1, 0]:.2f} m; "
        f"collision={collided}; wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e3:.0f} KB)"
    )


if __name__ == "__main__":
    main()
