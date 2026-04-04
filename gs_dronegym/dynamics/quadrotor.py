"""Six-degree-of-freedom quadrotor dynamics for GS-DroneGym.

This module implements a lightweight SE(3)-style quadrotor model with RK4
integration, a first-order thrust actuator, and simple collision checks against
scene bounds and task obstacles. It is designed to be dependency-light and easy
to inspect for research prototyping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from gs_dronegym.tasks.base_task import BoxObstacle, CylinderObstacle, Obstacle

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DynamicsConfig:
    """Physical parameters for the quadrotor model."""

    mass: float = 0.5
    gravity: float = 9.81
    sim_dt: float = 0.005
    obs_dt: float = 0.1
    thrust_time_constant: float = 0.05
    rate_time_constant: float = 0.08
    max_rate: float = float(np.pi)
    drag_coefficient: float = 0.08


class QuadrotorDynamics:
    """Simplified 6-DOF quadrotor dynamics model."""

    def __init__(self, mass: float = 0.5, sim_dt: float = 0.005, obs_dt: float = 0.1) -> None:
        """Initialize the quadrotor dynamics.

        Args:
            mass: Vehicle mass in kilograms.
            sim_dt: Internal integration step in seconds.
            obs_dt: Observation/control step in seconds.
        """
        self.config = DynamicsConfig(mass=mass, sim_dt=sim_dt, obs_dt=obs_dt)
        self.state = np.zeros(12, dtype=np.float32)
        self._actual_thrust = np.float32(self.hover_thrust)
        self.scene_bbox = np.array(
            [[-50.0, -50.0, 0.0], [50.0, 50.0, 50.0]],
            dtype=np.float32,
        )
        self.obstacles: list[Obstacle] = []
        self._substeps = max(1, int(round(self.config.obs_dt / self.config.sim_dt)))

    @property
    def hover_thrust(self) -> float:
        """Return the nominal hover thrust.

        Returns:
            Mass times gravity in Newtons.
        """
        return self.config.mass * self.config.gravity

    def reset(self, init_state: np.ndarray | None = None) -> np.ndarray:
        """Reset the quadrotor state.

        Args:
            init_state: Optional initial state vector.

        Returns:
            The reset state.
        """
        self.state = (
            np.zeros(12, dtype=np.float32)
            if init_state is None
            else np.asarray(init_state, dtype=np.float32).copy()
        )
        if init_state is None:
            self.state[2] = np.float32(1.0)
        self._actual_thrust = np.float32(self.hover_thrust)
        return self.get_state()

    def set_collision_geometry(
        self,
        scene_bbox: np.ndarray,
        obstacles: list[Obstacle] | None = None,
    ) -> None:
        """Set the scene collision geometry.

        Args:
            scene_bbox: Scene bounding box as a ``(2, 3)`` array.
            obstacles: Optional list of obstacles.
        """
        self.scene_bbox = np.asarray(scene_bbox, dtype=np.float32).reshape(2, 3)
        self.obstacles = list(obstacles or [])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Advance the dynamics by one observation interval.

        Args:
            action: Physical action vector
                ``[collective_thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]``.
                The thrust command is interpreted around hover thrust, so zero
                corresponds to hovering.

        Returns:
            The new state and whether a collision occurred.
        """
        clipped = self._clip_action(np.asarray(action, dtype=np.float32))
        augmented_state = np.concatenate(
            [self.state.astype(np.float32), np.array([self._actual_thrust], dtype=np.float32)]
        )
        for _ in range(self._substeps):
            augmented_state = self._integrate_rk4(augmented_state, clipped, self.config.sim_dt)
        self.state = augmented_state[:12].astype(np.float32)
        self._actual_thrust = np.float32(augmented_state[12])
        collision = self._check_collision(self.state)
        return self.get_state(), collision

    def get_state(self) -> np.ndarray:
        """Return a copy of the current state.

        Returns:
            State vector of shape ``(12,)``.
        """
        return self.state.astype(np.float32, copy=True)

    def rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute the body-to-world rotation matrix.

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

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip the action to the physical command limits.

        Args:
            action: Unclipped action vector.

        Returns:
            Clipped action vector.
        """
        clipped = action.astype(np.float32, copy=True)
        thrust_min = -self.hover_thrust
        thrust_max = 3.0 * self.hover_thrust
        clipped[0] = np.clip(clipped[0], thrust_min, thrust_max)
        clipped[1:] = np.clip(clipped[1:], -self.config.max_rate, self.config.max_rate)
        return clipped

    def _augment_command(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        """Convert an action into total thrust and desired body rates.

        Args:
            action: Clipped action vector.

        Returns:
            Total thrust command and desired angular rates.
        """
        total_thrust = float(np.clip(action[0] + self.hover_thrust, 0.0, 4.0 * self.hover_thrust))
        desired_rates = action[1:].astype(np.float32)
        return total_thrust, desired_rates

    def _integrate_rk4(
        self,
        augmented_state: np.ndarray,
        action: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Perform RK4 integration on the augmented state.

        Args:
            augmented_state: 13D state including actual thrust.
            action: Clipped action vector.
            dt: Integration time step.

        Returns:
            Updated augmented state.
        """
        k1 = self._derivative(augmented_state, action)
        k2 = self._derivative(augmented_state + 0.5 * dt * k1, action)
        k3 = self._derivative(augmented_state + 0.5 * dt * k2, action)
        k4 = self._derivative(augmented_state + dt * k3, action)
        return (augmented_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(
            np.float32
        )

    def _integrate_euler(
        self,
        augmented_state: np.ndarray,
        action: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Perform explicit Euler integration on the augmented state.

        Args:
            augmented_state: 13D state including actual thrust.
            action: Clipped action vector.
            dt: Integration time step.

        Returns:
            Updated augmented state.
        """
        derivative = self._derivative(augmented_state, action)
        return (augmented_state + dt * derivative).astype(np.float32)

    def _derivative(self, augmented_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute continuous-time state derivatives.

        Args:
            augmented_state: 13D state including actual thrust.
            action: Clipped action vector.

        Returns:
            Derivative vector of shape ``(13,)``.
        """
        state = augmented_state[:12].astype(np.float32)
        thrust = float(augmented_state[12])
        total_thrust_cmd, desired_rates = self._augment_command(action)

        position = state[:3]
        velocity = state[3:6]
        roll, pitch, yaw = [float(v) for v in state[6:9]]
        rates = state[9:12]

        del position
        rotation = self.rotation_matrix(roll, pitch, yaw)
        thrust_world = rotation @ np.array([0.0, 0.0, thrust], dtype=np.float32)
        gravity = np.array([0.0, 0.0, -self.config.gravity], dtype=np.float32)
        drag = -self.config.drag_coefficient * velocity
        acceleration = gravity + thrust_world / self.config.mass + drag

        rate_error = desired_rates - rates
        rate_dot = rate_error / self.config.rate_time_constant
        euler_dot = self._body_rates_to_euler_dot(roll, pitch, rates)
        thrust_dot = (total_thrust_cmd - thrust) / self.config.thrust_time_constant

        derivative = np.zeros(13, dtype=np.float32)
        derivative[:3] = velocity
        derivative[3:6] = acceleration.astype(np.float32)
        derivative[6:9] = euler_dot.astype(np.float32)
        derivative[9:12] = rate_dot.astype(np.float32)
        derivative[12] = np.float32(thrust_dot)
        return derivative

    def _body_rates_to_euler_dot(self, roll: float, pitch: float, rates: np.ndarray) -> np.ndarray:
        """Convert body angular rates to ZYX Euler angle derivatives.

        Args:
            roll: Current roll angle.
            pitch: Current pitch angle.
            rates: Body rates ``[p, q, r]``.

        Returns:
            Euler angle derivatives.
        """
        p, q, r = [float(v) for v in rates]
        tan_pitch = float(np.tan(np.clip(pitch, -1.55, 1.55)))
        sec_pitch = 1.0 / float(np.cos(np.clip(pitch, -1.55, 1.55)))
        roll_dot = p + q * np.sin(roll) * tan_pitch + r * np.cos(roll) * tan_pitch
        pitch_dot = q * np.cos(roll) - r * np.sin(roll)
        yaw_dot = q * np.sin(roll) * sec_pitch + r * np.cos(roll) * sec_pitch
        return np.array([roll_dot, pitch_dot, yaw_dot], dtype=np.float32)

    def _check_collision(self, state: np.ndarray) -> bool:
        """Check collisions against the scene bounds and task obstacles.

        Args:
            state: Current drone state.

        Returns:
            ``True`` if a collision is detected.
        """
        position = state[:3]
        if position[2] < 0.0:
            return True
        if np.any(position < self.scene_bbox[0]) or np.any(position > self.scene_bbox[1]):
            return True
        for obstacle in self.obstacles:
            if isinstance(obstacle, CylinderObstacle) and self._collides_cylinder(
                position, obstacle
            ):
                return True
            if isinstance(obstacle, BoxObstacle) and self._collides_box(position, obstacle):
                return True
        return False

    def _collides_cylinder(self, position: np.ndarray, obstacle: CylinderObstacle) -> bool:
        """Check collision against a cylinder.

        Args:
            position: Drone position.
            obstacle: Cylindrical obstacle.

        Returns:
            ``True`` if the position intersects the cylinder volume.
        """
        horizontal = position[:2] - obstacle.center[:2]
        within_radius = float(np.linalg.norm(horizontal)) <= obstacle.radius
        z_min = obstacle.center[2] - obstacle.height / 2.0
        z_max = obstacle.center[2] + obstacle.height / 2.0
        within_height = z_min <= float(position[2]) <= z_max
        return within_radius and within_height

    def _collides_box(self, position: np.ndarray, obstacle: BoxObstacle) -> bool:
        """Check collision against an axis-aligned box.

        Args:
            position: Drone position.
            obstacle: Box obstacle.

        Returns:
            ``True`` if the position lies inside the box.
        """
        return bool(
            np.all(position >= obstacle.min_corner.astype(np.float32))
            and np.all(position <= obstacle.max_corner.astype(np.float32))
        )
