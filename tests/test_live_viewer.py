"""Tests for the interactive live viewer helpers."""

from __future__ import annotations

import numpy as np

from gs_dronegym.cli.live_viewer import (
    KeyboardState,
    _depth_to_rgb,
    _make_keyboard_action,
    _make_scripted_demo_action,
    _normalize_scene,
    run_live_viewer,
)


def test_normalize_scene_handles_none_variants() -> None:
    """Scene normalization should collapse string none variants."""
    assert _normalize_scene(None) is None
    assert _normalize_scene("None") is None
    assert _normalize_scene("none") is None
    assert _normalize_scene("garden") == "garden"


def test_keyboard_action_waypoint_mapping() -> None:
    """Keyboard waypoint control should map keys into normalized deltas."""
    keyboard_state = KeyboardState(pressed={"i", "j", "u", "n"})
    action = _make_keyboard_action(keyboard_state, action_mode="waypoint")
    assert np.allclose(action, np.asarray([0.9, 0.9, 0.8, 0.7], dtype=np.float32))


def test_depth_to_rgb_returns_uint8_image() -> None:
    """Depth visualization should emit a displayable RGB image."""
    depth = np.linspace(0.1, 2.0, num=16, dtype=np.float32).reshape(4, 4)
    rgb = _depth_to_rgb(depth)
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8


def test_scripted_demo_action_is_deterministic() -> None:
    """The scripted demo policy should emit deterministic phase actions."""
    assert np.allclose(
        _make_scripted_demo_action(step_index=0, action_mode="waypoint"),
        np.asarray([0.9, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.allclose(
        _make_scripted_demo_action(step_index=20, action_mode="waypoint"),
        np.asarray([0.0, 0.0, 0.8, 0.0], dtype=np.float32),
    )


def test_live_viewer_can_save_gif_without_show(tmp_path: object) -> None:
    """The live viewer should save a short GIF in headless mode."""
    gif_path = tmp_path / "viewer.gif"
    saved_path = run_live_viewer(
        env_id="PointNav-v0",
        scene=None,
        steps=2,
        seed=0,
        policy="zero",
        fps=4.0,
        save_gif=gif_path,
        show=False,
    )
    assert saved_path == gif_path
    assert gif_path.exists()
