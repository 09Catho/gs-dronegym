"""Tests for the GS-DroneGym environment integration path."""

from __future__ import annotations

import numpy as np

from gs_dronegym.data.planner import ExpertPlanner
from gs_dronegym.env import GSDroneEnv
from gs_dronegym.tasks import PointNavTask


def test_env_reset_matches_observation_space() -> None:
    """Environment reset should emit an observation matching the declared space."""
    env = GSDroneEnv(task=PointNavTask(), scene_path=None)
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)


def test_step_returns_gymnasium_five_tuple() -> None:
    """Environment step should follow the Gymnasium API contract."""
    env = GSDroneEnv(task=PointNavTask(), scene_path=None)
    env.reset(seed=1)
    result = env.step(env.action_space.sample())
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_runs_random_steps_without_error() -> None:
    """The mock-renderer environment should survive several random steps."""
    env = GSDroneEnv(task=PointNavTask(), scene_path=None)
    env.reset(seed=3)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        if terminated or truncated:
            break


def test_episode_terminates_on_collision() -> None:
    """A collision should terminate the episode."""
    env = GSDroneEnv(task=PointNavTask(), scene_path=None, action_mode="direct")
    env.reset(seed=5)
    env.dynamics.state[2] = np.float32(0.05)
    terminated = False
    for _ in range(20):
        action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert terminated
    assert info["collision"] is True


def test_env_is_reproducible_given_same_seed() -> None:
    """Seeded mock-renderer resets should be deterministic."""
    env_a = GSDroneEnv(task=PointNavTask(), scene_path=None)
    env_b = GSDroneEnv(task=PointNavTask(), scene_path=None)
    obs_a, info_a = env_a.reset(seed=11)
    obs_b, info_b = env_b.reset(seed=11)

    assert np.array_equal(obs_a["rgb"], obs_b["rgb"])
    assert np.array_equal(obs_a["depth"], obs_b["depth"])
    assert np.allclose(obs_a["state"], obs_b["state"])
    assert obs_a["instruction"] == obs_b["instruction"]
    assert np.allclose(info_a["drone_state"], info_b["drone_state"])


def test_expert_planner_solves_live_pointnav() -> None:
    """The geometric expert should solve seeded live PointNav rollouts."""
    planner = ExpertPlanner()
    for seed in range(3):
        env = GSDroneEnv(task=PointNavTask(), scene_path=None, image_size=(32, 32))
        obs, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        info: dict[str, object] = {}
        while not (terminated or truncated):
            state = np.asarray(obs["state"], dtype=np.float32)
            waypoint, _ = planner.plan_waypoint(
                state=state,
                goal_position=env.goal_position,
                task=env.task,
                scene_bbox=env.scene_bbox,
                obs_dt=env.dynamics.config.obs_dt,
            )
            action = planner.normalized_waypoint_action(state, waypoint)
            obs, _, terminated, truncated, info = env.step(action)
        assert info["success"] is True
        assert info["collision"] is False
