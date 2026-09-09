"""Scene geometry must be derived from the Gaussians that produce the image.

Historically, navigation bounds came from raw min/max over Gaussian positions
and obstacles were hand-authored primitives unrelated to the rendered scene, so
a policy could fly through a visible wall. These tests pin the derived geometry
against a synthetic room whose true geometry is known exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gs_dronegym
from gs_dronegym.scene.occupancy import (
    build_occupancy_grid,
    derive_scene_bounds,
    load_gaussian_cloud,
)
from gs_dronegym.scene.synthetic import SyntheticRoomConfig, write_synthetic_room_ply

ROOM = SyntheticRoomConfig(size=(10.0, 8.0, 3.0), spacing=0.15)


@pytest.fixture(scope="module")
def room_ply(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a synthetic Gaussian room once for the module.

    Args:
        tmp_path_factory: Pytest temporary directory factory.

    Returns:
        Path to the written scene.
    """
    path = tmp_path_factory.mktemp("scene") / "room.ply"
    return write_synthetic_room_ply(path, ROOM)


def test_synthetic_scene_passes_loader_validation(room_ply: Path) -> None:
    """The generated fixture must be a valid Gaussian scene."""
    from gs_dronegym.scene.scene_loader import SceneLoader

    SceneLoader().validate_ply(room_ply)


def test_derived_bounds_match_the_room(room_ply: Path) -> None:
    """Bounds must recover the room extent rather than raw outlier extremes."""
    bounds = derive_scene_bounds(load_gaussian_cloud(room_ply))
    assert bounds[0] == pytest.approx([-5.0, -4.0, 0.0], abs=0.2)
    assert bounds[1] == pytest.approx([5.0, 4.0, 3.0], abs=0.2)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.0, 3.0, 1.5), False),
        ((3.0, -3.0, 2.0), False),
        ((0.0, 0.0, 1.5), False),
        ((0.5, 0.0, 1.5), True),
        ((5.0, 0.0, 1.5), True),
        ((0.0, -4.0, 1.5), True),
        ((2.0, 2.0, 0.0), True),
    ],
)
def test_occupancy_matches_known_room_geometry(
    room_ply: Path, point: tuple[float, float, float], expected: bool
) -> None:
    """Occupancy must agree with the room's true surfaces."""
    grid = build_occupancy_grid(load_gaussian_cloud(room_ply), voxel_size=0.1)
    query = np.array([point], dtype=np.float32)
    assert bool(grid.is_occupied(query)[0]) is expected


def test_occupancy_is_sparse_but_non_empty(room_ply: Path) -> None:
    """A room of surfaces should occupy a small minority of voxels."""
    grid = build_occupancy_grid(load_gaussian_cloud(room_ply), voxel_size=0.1)
    assert 0.001 < grid.occupancy_fraction() < 0.5


def test_environment_derives_geometry_from_the_scene(room_ply: Path) -> None:
    """Loading a Gaussian scene must populate bounds and collision geometry."""
    env = gs_dronegym.make(
        "PointNav-v0", scene=str(room_ply), renderer_device="cpu", observation_mode="state"
    )
    inner = env.unwrapped
    inner.reset(seed=0)
    assert inner.occupancy is not None
    assert inner.dynamics.occupancy is not None
    assert inner.scene_bbox[0] == pytest.approx([-5.0, -4.0, 0.0], abs=0.2)
    env.close()


def test_scene_collision_can_be_disabled(room_ply: Path) -> None:
    """Opting out must restore the previous hand-authored behaviour."""
    env = gs_dronegym.make(
        "PointNav-v0",
        scene=str(room_ply),
        renderer_device="cpu",
        observation_mode="state",
        scene_collision=False,
    )
    inner = env.unwrapped
    inner.reset(seed=0)
    assert inner.occupancy is None
    assert inner.dynamics.occupancy is None
    env.close()


def test_positions_inside_a_wall_register_a_collision(room_ply: Path) -> None:
    """A point on a rendered surface must be a collision; open air must not."""
    env = gs_dronegym.make(
        "PointNav-v0", scene=str(room_ply), renderer_device="cpu", observation_mode="state"
    )
    inner = env.unwrapped
    inner.reset(seed=0)
    dynamics = inner.dynamics

    dynamics.state[:3] = np.array([4.99, 0.0, 1.5], dtype=np.float32)
    assert dynamics._check_collision(dynamics.state) is True

    dynamics.state[:3] = np.array([0.0, 3.0, 1.5], dtype=np.float32)
    assert dynamics._check_collision(dynamics.state) is False
    env.close()


def test_mock_scene_keeps_hand_authored_geometry() -> None:
    """With no Gaussian scene there is nothing to derive geometry from."""
    env = gs_dronegym.make("PointNav-v0", scene=None, observation_mode="state")
    inner = env.unwrapped
    inner.reset(seed=0)
    assert inner.occupancy is None
    env.close()
