"""Tests for quadrotor dynamics and low-level integration behavior."""

from __future__ import annotations

import numpy as np

from gs_dronegym.dynamics import QuadrotorDynamics


def test_hover_action_keeps_altitude_stable() -> None:
    """Zero hover-centered action should keep altitude nearly constant."""
    dynamics = QuadrotorDynamics()
    state = dynamics.reset()
    initial_z = float(state[2])
    action = np.zeros(4, dtype=np.float32)
    for _ in range(10):
        state, collision = dynamics.step(action)
        assert not collision
    assert abs(float(state[2]) - initial_z) < 0.1


def test_step_output_shape() -> None:
    """Dynamics step should return a 12D state vector."""
    dynamics = QuadrotorDynamics()
    dynamics.reset()
    state, _ = dynamics.step(np.zeros(4, dtype=np.float32))
    assert state.shape == (12,)


def test_rk4_is_more_energy_conserving_than_euler() -> None:
    """RK4 integration should drift less than Euler in ballistic motion."""
    dynamics = QuadrotorDynamics()
    dynamics.config.drag_coefficient = 0.0
    base_state = np.zeros(12, dtype=np.float32)
    base_state[2] = 10.0
    augmented = np.concatenate([base_state, np.array([0.0], dtype=np.float32)])
    action = np.array([-dynamics.hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)

    rk_state = augmented.copy()
    euler_state = augmented.copy()
    for _ in range(20):
        rk_state = dynamics._integrate_rk4(rk_state, action, 0.05)
        euler_state = dynamics._integrate_euler(euler_state, action, 0.05)

    def energy(aug_state: np.ndarray) -> float:
        velocity = aug_state[3:6]
        z = float(aug_state[2])
        kinetic = 0.5 * dynamics.config.mass * float(np.dot(velocity, velocity))
        potential = dynamics.config.mass * dynamics.config.gravity * z
        return kinetic + potential

    initial_energy = energy(augmented)
    rk_drift = abs(energy(rk_state) - initial_energy)
    euler_drift = abs(energy(euler_state) - initial_energy)
    assert rk_drift < euler_drift


def test_collision_detection_triggers_below_ground() -> None:
    """Collision detection should trigger when altitude goes below zero."""
    dynamics = QuadrotorDynamics()
    init_state = np.zeros(12, dtype=np.float32)
    init_state[2] = 0.05
    dynamics.reset(init_state)
    collision = False
    for _ in range(20):
        action = np.array([-dynamics.hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)
        _, collision = dynamics.step(action)
        if collision:
            break
    assert collision
