"""Tests for task reset logic, success conditions, and obstacle sequencing."""

from __future__ import annotations

import numpy as np

from gs_dronegym.tasks import (
    DynamicFollowTask,
    NarrowCorridorTask,
    ObjectNavTask,
    ObstacleSlalomTask,
    PointNavTask,
)

SCENE_BBOX = np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 5.0]], dtype=np.float32)


def test_each_task_reset_returns_expected_shapes() -> None:
    """Every task reset should return a valid init state, goal, and instruction."""
    tasks = [
        PointNavTask(),
        ObjectNavTask(),
        ObstacleSlalomTask(),
        DynamicFollowTask(),
        NarrowCorridorTask(),
    ]
    for task in tasks:
        init_state, goal, instruction = task.reset(SCENE_BBOX)
        assert init_state.shape == (12,)
        assert goal.shape == (3,)
        assert isinstance(instruction, str)
        assert instruction


def test_each_task_success_or_goal_logic() -> None:
    """Tasks should recognize goal completion under their success conditions."""
    point = PointNavTask()
    point.reset(SCENE_BBOX)
    state = np.zeros(12, dtype=np.float32)
    state[:3] = point.get_goal_position()
    assert point.is_success(state)

    object_nav = ObjectNavTask()
    object_nav.reset(SCENE_BBOX)
    state[:3] = object_nav.get_goal_position()
    assert object_nav.is_success(state)

    slalom = ObstacleSlalomTask()
    slalom.reset(SCENE_BBOX)
    slalom.current_gate_index = len(slalom.gate_centers)
    state[:3] = slalom.final_goal
    assert slalom.is_success(state)

    follow = DynamicFollowTask()
    follow.reset(SCENE_BBOX)
    for step in range(follow.required_hold_steps):
        state[:3] = follow.get_goal_position()
        follow.update(state, step=step, obs_dt=0.1)
    assert follow.is_success(state)

    corridor = NarrowCorridorTask()
    corridor.reset(SCENE_BBOX)
    state[:3] = corridor.get_goal_position()
    assert corridor.is_success(state)


def test_each_task_failure_on_collision() -> None:
    """All tasks should treat collision as failure by default."""
    tasks = [
        PointNavTask(),
        ObjectNavTask(),
        ObstacleSlalomTask(),
        DynamicFollowTask(),
        NarrowCorridorTask(),
    ]
    state = np.zeros(12, dtype=np.float32)
    for task in tasks:
        task.reset(SCENE_BBOX)
        assert task.is_failure(state, collision=True)


def test_obstacle_slalom_gates_are_in_sequence() -> None:
    """Slalom gates should progress monotonically along the course."""
    task = ObstacleSlalomTask()
    task.reset(SCENE_BBOX)
    gate_x = task.gate_centers[:, 0]
    assert np.all(np.diff(gate_x) > 0.0)
