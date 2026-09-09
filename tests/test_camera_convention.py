"""The camera must follow the vision convention the rasterizer assumes.

The drone body frame is x forward, y left, z up. Pinhole intrinsics and the
Gaussian rasterizer use x right, y down, z along the optical axis. If the two
are not related by an axis permutation, the rasterizer treats a sideways axis as
depth and renders a view unrelated to where the drone is looking. The mock
renderer never exercised this, so it went undetected.
"""

from __future__ import annotations

import numpy as np
import pytest

from gs_dronegym.renderer.camera_model import CameraModel


def _camera_coords(state: np.ndarray, world_point: list[float]) -> np.ndarray:
    """Transform a world point into the camera frame.

    Args:
        state: 12D drone state.
        world_point: World-space point.

    Returns:
        Point in camera coordinates.
    """
    w2c = CameraModel(image_width=96, image_height=96).get_extrinsics(state)
    homogeneous = np.array([*world_point, 1.0], dtype=np.float32)
    return np.asarray(w2c @ homogeneous, dtype=np.float32)[:3]


def _level_state(position: list[float], yaw: float = 0.0) -> np.ndarray:
    """Build a level drone state at a position and yaw.

    Args:
        position: World position.
        yaw: Yaw angle in radians.

    Returns:
        12D state vector.
    """
    state = np.zeros(12, dtype=np.float32)
    state[:3] = position
    state[8] = yaw
    return state


def test_extrinsics_place_the_camera_at_the_drone() -> None:
    """Inverting the transform must recover the camera's world position."""
    state = _level_state([-3.0, 0.0, 1.5])
    w2c = CameraModel().get_extrinsics(state)
    camera_world = np.linalg.inv(w2c)[:3, 3]
    # The default mount sits 0.1 m forward and 0.02 m above the body origin.
    assert camera_world == pytest.approx([-2.9, 0.0, 1.52], abs=0.05)


def test_points_ahead_have_positive_depth() -> None:
    """A point in front of the drone must lie at positive optical depth."""
    coords = _camera_coords(_level_state([-3.0, 0.0, 1.5]), [5.0, 0.0, 1.5])
    assert coords[2] > 7.0


def test_points_behind_have_negative_depth() -> None:
    """A point behind the drone must lie at negative optical depth."""
    coords = _camera_coords(_level_state([-3.0, 0.0, 1.5]), [-5.0, 0.0, 1.5])
    assert coords[2] < 0.0


def test_points_below_map_to_positive_image_y() -> None:
    """Image y increases downwards in the vision convention."""
    coords = _camera_coords(_level_state([0.0, 0.0, 2.0]), [4.0, 0.0, 0.0])
    assert coords[1] > 0.0


def test_points_to_the_left_map_to_negative_image_x() -> None:
    """Image x increases to the right in the vision convention."""
    coords = _camera_coords(_level_state([0.0, 0.0, 1.5]), [4.0, 3.0, 1.5])
    assert coords[0] < 0.0


def test_yaw_rotates_the_optical_axis() -> None:
    """After yawing 90 degrees, world +y must be ahead of the camera."""
    coords = _camera_coords(_level_state([0.0, 0.0, 1.5], yaw=float(np.pi / 2.0)), [0.0, 5.0, 1.5])
    assert coords[2] > 4.0
    assert abs(float(coords[0])) < 0.5


def test_optical_axis_is_tilted_downwards() -> None:
    """The default mount looks slightly below the horizon."""
    w2c = CameraModel().get_extrinsics(_level_state([0.0, 0.0, 1.5]))
    optical_axis_world = np.linalg.inv(w2c)[:3, 2]
    assert optical_axis_world[0] > 0.9
    assert optical_axis_world[2] < 0.0


def test_rotation_matrix_is_orthonormal() -> None:
    """The extrinsic rotation must be a proper rigid transform."""
    w2c = CameraModel().get_extrinsics(_level_state([1.0, -2.0, 3.0], yaw=0.7))
    rotation = w2c[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-5)
