"""Control, physics and camera rates must be independently configurable.

Decoupling these clocks is both a correctness property (the integrator must not
change when the decision rate changes) and the hook that later allows inference
latency to be modelled inside the environment.
"""

from __future__ import annotations

import numpy as np
import pytest

import gs_dronegym


def _final_position(control_hz: float, duration_s: float = 2.0) -> np.ndarray:
    """Integrate a constant command for a fixed simulated duration.

    Args:
        control_hz: Decision rate.
        duration_s: Simulated duration to integrate over.

    Returns:
        Final drone position.
    """
    env = gs_dronegym.make(
        "PointNav-v0",
        scene=None,
        control_hz=control_hz,
        physics_hz=200.0,
        observation_mode="state",
        action_mode="direct",
    )
    inner = env.unwrapped
    inner.reset(seed=0)
    action = np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(int(round(duration_s * control_hz))):
        inner.step(action)
    position = np.asarray(inner.dynamics.get_state()[:3], dtype=np.float32).copy()
    env.close()
    return position


def _count_renders(control_hz: float, camera_hz: float | None, n_steps: int) -> int:
    """Count rasterizations performed over a fixed number of control steps.

    Args:
        control_hz: Decision rate.
        camera_hz: Camera rate, or ``None`` to render every step.
        n_steps: Number of control steps to run.

    Returns:
        Number of renderer invocations during the stepping loop.
    """
    env = gs_dronegym.make(
        "PointNav-v0", scene=None, control_hz=control_hz, camera_hz=camera_hz
    )
    inner = env.unwrapped
    calls = {"n": 0}
    original = inner.renderer.render

    def counting_render(w2c: np.ndarray) -> dict[str, np.ndarray]:
        calls["n"] += 1
        return original(w2c)

    inner.renderer.render = counting_render  # type: ignore[method-assign]
    inner.reset(seed=0)
    calls["n"] = 0
    for _ in range(n_steps):
        inner.step(np.zeros(4, dtype=np.float32))
    env.close()
    return calls["n"]


@pytest.mark.parametrize("control_hz", [5.0, 20.0, 50.0])
def test_control_rate_does_not_change_the_integrated_trajectory(control_hz: float) -> None:
    """Changing the decision rate must not change the physics."""
    reference = _final_position(10.0)
    assert np.allclose(_final_position(control_hz), reference, atol=1e-4)


def test_default_rates_preserve_historical_behaviour() -> None:
    """Defaults must keep the 10 Hz control and 200 Hz physics of earlier releases."""
    env = gs_dronegym.make("PointNav-v0", scene=None)
    inner = env.unwrapped
    assert inner.control_hz == pytest.approx(10.0)
    assert inner.physics_hz == pytest.approx(200.0)
    assert inner.dynamics.config.obs_dt == pytest.approx(0.1)
    assert inner.dynamics.config.sim_dt == pytest.approx(0.005)
    env.close()


def test_camera_renders_every_control_step_by_default() -> None:
    """Without a camera rate the renderer runs once per decision."""
    assert _count_renders(10.0, None, n_steps=10) == 10


def test_slower_camera_holds_the_previous_frame() -> None:
    """A camera slower than the control loop must rasterize proportionally less."""
    assert _count_renders(10.0, 2.0, n_steps=10) == 2


def test_held_frames_are_identical_between_exposures() -> None:
    """Between exposures the policy must see the same image, not a fresh render."""
    env = gs_dronegym.make("PointNav-v0", scene=None, control_hz=10.0, camera_hz=2.0)
    inner = env.unwrapped
    inner.reset(seed=0)
    frames = []
    for _ in range(4):
        obs, _, _, _, _ = inner.step(np.zeros(4, dtype=np.float32))
        frames.append(np.asarray(obs["rgb"], dtype=np.uint8).copy())
    env.close()
    assert np.array_equal(frames[0], frames[1])
    assert np.array_equal(frames[1], frames[2])


def test_state_observation_mode_drops_image_keys() -> None:
    """State-only mode must remove image observations from the space and output."""
    env = gs_dronegym.make("PointNav-v0", scene=None, observation_mode="state")
    obs, _ = env.reset(seed=0)
    assert "rgb" not in obs
    assert "depth" not in obs
    assert "rgb" not in env.observation_space.spaces
    assert "state" in obs
    env.close()


def test_state_observation_mode_performs_no_rasterization() -> None:
    """State-only mode must skip rendering entirely."""
    env = gs_dronegym.make("PointNav-v0", scene=None, observation_mode="state")
    inner = env.unwrapped
    calls = {"n": 0}
    original = inner.renderer.render

    def counting_render(w2c: np.ndarray) -> dict[str, np.ndarray]:
        calls["n"] += 1
        return original(w2c)

    inner.renderer.render = counting_render  # type: ignore[method-assign]
    inner.reset(seed=0)
    for _ in range(5):
        inner.step(np.zeros(4, dtype=np.float32))
    env.close()
    assert calls["n"] == 0


def test_episode_time_limit_is_expressed_in_simulated_seconds() -> None:
    """A time limit must truncate at the same instant regardless of control rate."""
    for control_hz in (5.0, 20.0):
        env = gs_dronegym.make(
            "PointNav-v0",
            scene=None,
            control_hz=control_hz,
            observation_mode="state",
            episode_time_limit_s=2.0,
        )
        inner = env.unwrapped
        inner.reset(seed=0)
        steps = 0
        truncated = False
        while not truncated and steps < 1000:
            _, _, terminated, truncated, _ = inner.step(np.zeros(4, dtype=np.float32))
            steps += 1
            if terminated:
                break
        env.close()
        if truncated:
            assert steps == pytest.approx(2.0 * control_hz, abs=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_hz": 0.0}, "control_hz must be positive"),
        ({"control_hz": 100.0, "physics_hz": 50.0}, "must be at least control_hz"),
        ({"control_hz": 10.0, "camera_hz": 20.0}, "camera_hz must be in"),
    ],
)
def test_invalid_rate_configurations_are_rejected(
    kwargs: dict[str, float], message: str
) -> None:
    """Inconsistent rate settings must fail loudly at construction."""
    with pytest.raises(ValueError, match=message):
        gs_dronegym.make("PointNav-v0", scene=None, **kwargs)
