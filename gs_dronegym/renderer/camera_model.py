"""Camera intrinsics and extrinsics utilities for GS-DroneGym.

The renderer consumes camera matrices derived from the drone state. This module
encapsulates the drone-mounted pinhole camera geometry used by both the
Gaussian splatting renderer and the CPU fallback renderer.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class CameraModel:
    """Pinhole camera model attached to the drone body frame."""

    def __init__(
        self,
        image_width: int = 224,
        image_height: int = 224,
        fov_deg: float = 90.0,
        body_to_cam: np.ndarray | None = None,
    ) -> None:
        """Initialize the camera model.

        Args:
            image_width: Output image width in pixels.
            image_height: Output image height in pixels.
            fov_deg: Horizontal field of view in degrees.
            body_to_cam: Optional ``(4, 4)`` body-to-camera transform. If not
                provided, a forward-facing camera with 15 degree downward tilt
                is used.
        """
        self._width = int(image_width)
        self._height = int(image_height)
        self.fov_deg = float(fov_deg)
        self.body_to_cam = (
            self._default_body_to_cam()
            if body_to_cam is None
            else np.asarray(body_to_cam, dtype=np.float32)
        )

    @property
    def width(self) -> int:
        """Return the image width."""
        return self._width

    @property
    def height(self) -> int:
        """Return the image height."""
        return self._height

    def get_intrinsics(self) -> np.ndarray:
        """Compute the pinhole camera intrinsic matrix.

        Returns:
            Camera intrinsics matrix of shape ``(3, 3)``.
        """
        focal = 0.5 * self._width / np.tan(np.deg2rad(self.fov_deg) / 2.0)
        cx = self._width / 2.0
        cy = self._height / 2.0
        return np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def get_extrinsics(self, drone_state: np.ndarray) -> np.ndarray:
        """Compute the world-to-camera transform from the drone state.

        Args:
            drone_state: 12D drone state vector.

        Returns:
            World-to-camera SE(3) matrix of shape ``(4, 4)``.
        """
        state = np.asarray(drone_state, dtype=np.float32)
        roll, pitch, yaw = [float(v) for v in state[6:9]]
        position = state[:3].astype(np.float32)

        world_from_body = np.eye(4, dtype=np.float32)
        world_from_body[:3, :3] = self._rotation_matrix(roll, pitch, yaw)
        world_from_body[:3, 3] = position

        world_from_cam = world_from_body @ self.body_to_cam
        cam_from_world = np.linalg.inv(world_from_cam).astype(np.float32)
        LOGGER.debug("Computed camera extrinsics for state=%s", state)
        return cam_from_world

    def _default_body_to_cam(self) -> np.ndarray:
        """Create the default forward-facing camera transform.

        Returns:
            Body-to-camera transform.
        """
        tilt = np.deg2rad(-15.0)
        rot_y = np.array(
            [
                [np.cos(tilt), 0.0, np.sin(tilt)],
                [0.0, 1.0, 0.0],
                [-np.sin(tilt), 0.0, np.cos(tilt)],
            ],
            dtype=np.float32,
        )
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot_y
        transform[:3, 3] = np.array([0.1, 0.0, 0.02], dtype=np.float32)
        return transform

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute a ZYX rotation matrix.

        Args:
            roll: Roll angle in radians.
            pitch: Pitch angle in radians.
            yaw: Yaw angle in radians.

        Returns:
            Rotation matrix of shape ``(3, 3)``.
        """
        cr = float(np.cos(roll))
        sr = float(np.sin(roll))
        cp = float(np.cos(pitch))
        sp = float(np.sin(pitch))
        cy = float(np.cos(yaw))
        sy = float(np.sin(yaw))
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float32,
        )
