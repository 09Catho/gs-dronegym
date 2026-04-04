"""Waypoint tracking controller for GS-DroneGym quadrotor dynamics.

The controller maps VLA-style waypoint outputs into low-level thrust and body
rate commands. It uses a cascaded architecture with position, velocity,
attitude, and yaw loops tuned for the lightweight quadrotor model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ControllerGains:
    """Controller gains for the cascaded waypoint controller."""

    kp_pos: float = 1.25
    ki_pos: float = 0.05
    kd_pos: float = 0.35
    kp_att: float = 4.0
    kp_yaw: float = 2.5


class WaypointController:
    """Convert waypoint targets into thrust and body-rate commands."""

    def __init__(self, gains: dict[str, float] | None = None) -> None:
        """Initialize the waypoint controller.

        Args:
            gains: Optional gain override dictionary with keys ``kp_pos``,
                ``ki_pos``, ``kd_pos``, ``kp_att``, and ``kp_yaw``.
        """
        base = ControllerGains()
        if gains:
            for key, value in gains.items():
                if hasattr(base, key):
                    setattr(base, key, float(value))
        self.gains = {
            "kp_pos": base.kp_pos,
            "ki_pos": base.ki_pos,
            "kd_pos": base.kd_pos,
            "kp_att": base.kp_att,
            "kp_yaw": base.kp_yaw,
        }
        self.mass = 0.5
        self.gravity = 9.81
        self.max_rate = float(np.pi)
        self.max_integral = 2.0
        self.integral_error = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        """Reset the controller integrator state."""
        self.integral_error = np.zeros(3, dtype=np.float32)

    def compute(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute a low-level action from the current state and waypoint.

        Args:
            state: Current 12D quadrotor state.
            target: Target waypoint ``[x, y, z, yaw]``.

        Returns:
            Action vector ``[thrust, roll_rate, pitch_rate, yaw_rate]`` where
            thrust is centered around hover.
        """
        state_vec = np.asarray(state, dtype=np.float32)
        target_vec = np.asarray(target, dtype=np.float32)
        position = state_vec[:3]
        velocity = state_vec[3:6]
        roll, pitch, yaw = [float(v) for v in state_vec[6:9]]

        position_error = target_vec[:3] - position
        desired_velocity = self.gains["kp_pos"] * position_error
        velocity_error = desired_velocity - velocity
        self.integral_error += velocity_error * np.float32(0.1)
        self.integral_error = np.clip(
            self.integral_error, -self.max_integral, self.max_integral
        ).astype(np.float32)

        desired_accel = (
            self.gains["kp_pos"] * position_error
            + self.gains["kd_pos"] * velocity_error
            + self.gains["ki_pos"] * self.integral_error
        )
        desired_accel[2] += self.gravity

        desired_roll = np.clip(desired_accel[1] / self.gravity, -0.35, 0.35)
        desired_pitch = np.clip(-desired_accel[0] / self.gravity, -0.35, 0.35)
        thrust_total = float(self.mass * desired_accel[2])
        thrust_cmd = thrust_total - self.mass * self.gravity

        roll_rate = self.gains["kp_att"] * (desired_roll - roll)
        pitch_rate = self.gains["kp_att"] * (desired_pitch - pitch)
        yaw_error = self._wrap_angle(float(target_vec[3]) - yaw)
        yaw_rate = self.gains["kp_yaw"] * yaw_error

        action = np.array(
            [
                thrust_cmd,
                np.clip(roll_rate, -self.max_rate, self.max_rate),
                np.clip(pitch_rate, -self.max_rate, self.max_rate),
                np.clip(yaw_rate, -self.max_rate, self.max_rate),
            ],
            dtype=np.float32,
        )
        LOGGER.debug("Controller action=%s for target=%s", action, target_vec)
        return action

    def _wrap_angle(self, angle: float) -> float:
        """Wrap an angle to the ``[-pi, pi]`` range.

        Args:
            angle: Input angle in radians.

        Returns:
            Wrapped angle.
        """
        wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
        return float(wrapped)
