"""CLI entrypoint for interactive GS-DroneGym visualization.

This viewer is meant to make the environment understandable immediately. It can
run a rollout with a built-in policy or let a user fly manually with the
keyboard, display RGB and depth views, overlay state and distance-to-goal
information, and show a top-down trajectory plot while the environment steps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import gs_dronegym


@dataclass(slots=True)
class KeyboardState:
    """Track currently pressed keys for manual control."""

    pressed: set[str] = field(default_factory=set)
    paused: bool = False
    quit_requested: bool = False
    reset_requested: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Create the live-viewer CLI parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Show a live GS-DroneGym rollout.")
    parser.add_argument("--env-id", default="PointNav-v0", help="Registered environment ID.")
    parser.add_argument("--scene", default=None, help="Scene name or .ply path. Use None for mock.")
    parser.add_argument("--steps", type=int, default=100, help="Maximum number of steps.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument(
        "--policy",
        choices=["random", "zero", "keyboard", "scripted"],
        default="random",
        help="Built-in policy used to drive the environment.",
    )
    parser.add_argument(
        "--renderer-device",
        default="cuda",
        help="Renderer device, typically cuda or cpu.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["waypoint", "direct"],
        default="waypoint",
        help="Environment action mode for the viewer.",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="Viewer update rate.")
    parser.add_argument(
        "--save-gif",
        default=None,
        help="Optional output GIF path for saving the rendered run.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window; useful with --save-gif.",
    )
    return parser


def _normalize_scene(scene: str | None) -> str | None:
    """Normalize user-provided scene arguments.

    Args:
        scene: Raw scene CLI value.

    Returns:
        Normalized scene value or ``None``.
    """
    if scene in {None, "None", "none"}:
        return None
    return scene


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Convert a depth map into a viewable RGB heatmap.

    Args:
        depth: Depth map in meters.

    Returns:
        RGB heatmap image.
    """
    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    finite_depth = depth[finite_mask]
    depth_min = float(np.min(finite_depth))
    depth_max = float(np.max(finite_depth))
    if depth_max - depth_min < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = (depth - depth_min) / (depth_max - depth_min)
    colormap = plt.get_cmap("viridis")
    rgb = colormap(np.clip(normalized, 0.0, 1.0))[..., :3]
    return (rgb * 255.0).astype(np.uint8)


def _make_keyboard_action(
    keyboard_state: KeyboardState,
    action_mode: str,
) -> np.ndarray:
    """Convert pressed keys into an action vector.

    Args:
        keyboard_state: Current keyboard state.
        action_mode: Environment action mode.

    Returns:
        Normalized 4D action vector.
    """
    action = np.zeros(4, dtype=np.float32)
    keys = keyboard_state.pressed

    if action_mode == "waypoint":
        if "i" in keys or "up" in keys:
            action[0] += 0.9
        if "k" in keys or "down" in keys:
            action[0] -= 0.9
        if "j" in keys or "left" in keys:
            action[1] += 0.9
        if "l" in keys or "right" in keys:
            action[1] -= 0.9
        if "u" in keys:
            action[2] += 0.8
        if "o" in keys:
            action[2] -= 0.8
        if "n" in keys:
            action[3] += 0.7
        if "m" in keys:
            action[3] -= 0.7
    else:
        if "i" in keys or "up" in keys:
            action[0] += 0.7
        if "k" in keys or "down" in keys:
            action[0] -= 0.7
        if "j" in keys:
            action[1] += 0.7
        if "l" in keys:
            action[1] -= 0.7
        if "u" in keys:
            action[2] += 0.7
        if "o" in keys:
            action[2] -= 0.7
        if "n" in keys:
            action[3] += 0.7
        if "m" in keys:
            action[3] -= 0.7

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _select_action(
    env: object,
    policy: str,
    action_mode: str,
    keyboard_state: KeyboardState,
    step_index: int,
) -> np.ndarray:
    """Select an action from a built-in policy.

    Args:
        env: Gymnasium environment.
        policy: Policy name.
        action_mode: Environment action mode.
        keyboard_state: Current keyboard state.
        step_index: Current rollout step index.

    Returns:
        Action vector.
    """
    if policy == "zero":
        shape = tuple(int(dim) for dim in env.action_space.shape)
        return np.zeros(shape, dtype=np.float32)
    if policy == "keyboard":
        return _make_keyboard_action(keyboard_state, action_mode=action_mode)
    if policy == "scripted":
        return _make_scripted_demo_action(step_index=step_index, action_mode=action_mode)
    return np.asarray(env.action_space.sample(), dtype=np.float32)


def _make_scripted_demo_action(
    step_index: int,
    action_mode: str,
) -> np.ndarray:
    """Generate a deterministic action sequence for GIF demos.

    Args:
        step_index: Current rollout step index.
        action_mode: Environment action mode.

    Returns:
        Normalized 4D action vector.
    """
    phase = (step_index // 10) % 6
    action = np.zeros(4, dtype=np.float32)
    if phase == 0:
        action[0] = 0.9
    elif phase == 1:
        action[1] = 0.9
    elif phase == 2:
        action[2] = 0.8
    elif phase == 3:
        action[3] = 0.7
    elif phase == 4:
        action[1] = -0.9
    else:
        action[2] = -0.8
    if action_mode == "direct":
        action[0] = np.clip(action[0], -0.5, 0.7)
    return action


def _overlay_text(
    env_id: str,
    step_index: int,
    reward: float,
    info: dict[str, object],
    instruction: str,
    renderer_name: str,
    scene: str | None,
    action_mode: str,
    policy: str,
) -> str:
    """Build the text shown over the RGB panel.

    Args:
        env_id: Environment ID.
        step_index: Current step number.
        reward: Latest reward.
        info: Environment info dictionary.
        instruction: Task instruction.
        renderer_name: Active renderer class name.
        scene: Scene label.
        action_mode: Active action mode.
        policy: Active policy name.

    Returns:
        Multiline overlay text.
    """
    state = np.asarray(info["drone_state"], dtype=np.float32)
    position = state[:3]
    yaw_deg = float(np.degrees(state[8]))
    distance = float(info["distance_to_goal"])
    return (
        f"{env_id} | renderer={renderer_name} | scene={scene or 'mock'}\n"
        f"step={step_index} reward={reward:.3f} dist={distance:.2f}m "
        f"success={bool(info['success'])} collision={bool(info['collision'])}\n"
        f"pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) yaw={yaw_deg:.1f}deg "
        f"mode={action_mode} policy={policy}\n"
        f"instruction: {instruction}"
    )


def _controls_text(action_mode: str, keyboard_enabled: bool) -> str:
    """Build the viewer controls legend.

    Args:
        action_mode: Active action mode.
        keyboard_enabled: Whether keyboard control is enabled.

    Returns:
        Controls legend.
    """
    if not keyboard_enabled:
        return "policy-driven rollout"
    if action_mode == "waypoint":
        return (
            "keyboard: I/K forward-back | J/L lateral | U/O up-down | "
            "N/M yaw | P pause | R reset | Esc quit"
        )
    return (
        "keyboard: I/K thrust | J/L roll-rate | U/O pitch-rate | "
        "N/M yaw-rate | P pause | R reset | Esc quit"
    )


def _update_topdown_axis(
    axis: plt.Axes,
    positions: list[np.ndarray],
    goal_position: np.ndarray,
) -> None:
    """Redraw the top-down trajectory axis.

    Args:
        axis: Matplotlib axis.
        positions: History of XYZ positions.
        goal_position: Current goal position.
    """
    axis.clear()
    axis.set_title("Top-Down Trajectory")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.grid(True, alpha=0.25)
    axis.set_aspect("equal", adjustable="box")
    if not positions:
        return
    path = np.asarray(positions, dtype=np.float32)
    axis.plot(path[:, 0], path[:, 1], color="black", linewidth=1.5, alpha=0.8)
    axis.scatter(path[0, 0], path[0, 1], color="green", s=60, label="start")
    axis.scatter(path[-1, 0], path[-1, 1], color="royalblue", s=50, label="current")
    axis.scatter(goal_position[0], goal_position[1], color="red", marker="*", s=120, label="goal")
    padding = 1.0
    x_min = float(min(np.min(path[:, 0]), goal_position[0]) - padding)
    x_max = float(max(np.max(path[:, 0]), goal_position[0]) + padding)
    y_min = float(min(np.min(path[:, 1]), goal_position[1]) - padding)
    y_max = float(max(np.max(path[:, 1]), goal_position[1]) + padding)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.legend(loc="upper right")


def _connect_keyboard(
    figure: plt.Figure,
    keyboard_state: KeyboardState,
) -> None:
    """Attach keyboard event handlers to a matplotlib figure.

    Args:
        figure: Target figure.
        keyboard_state: Shared keyboard state.
    """

    def on_press(event: object) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        key_lower = str(key).lower()
        if key_lower == "p":
            keyboard_state.paused = not keyboard_state.paused
        elif key_lower == "escape":
            keyboard_state.quit_requested = True
        elif key_lower == "r":
            keyboard_state.reset_requested = True
        else:
            keyboard_state.pressed.add(key_lower)

    def on_release(event: object) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        keyboard_state.pressed.discard(str(key).lower())

    figure.canvas.mpl_connect("key_press_event", on_press)
    figure.canvas.mpl_connect("key_release_event", on_release)


def run_live_viewer(
    env_id: str = "PointNav-v0",
    scene: str | None = None,
    steps: int = 100,
    seed: int = 0,
    policy: str = "random",
    fps: float = 8.0,
    save_gif: str | Path | None = None,
    show: bool = True,
    renderer_device: str = "cuda",
    action_mode: str = "waypoint",
) -> Path | None:
    """Run the live viewer.

    Args:
        env_id: Registered environment ID.
        scene: Scene name or path.
        steps: Maximum number of environment steps.
        seed: Environment seed.
        policy: Built-in policy name.
        fps: Viewer update rate.
        save_gif: Optional GIF output path.
        show: Whether to open an interactive matplotlib window.
        renderer_device: Renderer device string.
        action_mode: Environment action mode.

    Returns:
        Saved GIF path if one was written, otherwise ``None``.
    """
    env = gs_dronegym.make(
        env_id,
        scene=scene,
        renderer_device=renderer_device,
        action_mode=action_mode,
    )
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    keyboard_state = KeyboardState()
    observation, info = env.reset(seed=seed)
    instruction = str(observation["instruction"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rgb_axis, depth_axis, topdown_axis = axes
    rgb_artist = rgb_axis.imshow(np.asarray(observation["rgb"], dtype=np.uint8))
    depth_artist = depth_axis.imshow(
        _depth_to_rgb(np.asarray(observation["depth"], dtype=np.float32))
    )
    rgb_axis.set_title("RGB")
    depth_axis.set_title("Depth")
    for axis in (rgb_axis, depth_axis):
        axis.axis("off")

    positions = [np.asarray(info["drone_state"], dtype=np.float32)[:3]]
    _update_topdown_axis(
        topdown_axis,
        positions=positions,
        goal_position=np.asarray(base_env.goal_position, dtype=np.float32),
    )

    renderer_name = base_env.renderer.__class__.__name__
    overlay = rgb_axis.text(
        0.02,
        0.02,
        _overlay_text(
            env_id=env_id,
            step_index=0,
            reward=0.0,
            info=info,
            instruction=instruction,
            renderer_name=renderer_name,
            scene=scene,
            action_mode=action_mode,
            policy=policy,
        ),
        transform=rgb_axis.transAxes,
        fontsize=8,
        color="white",
        verticalalignment="bottom",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )
    controls = fig.suptitle(_controls_text(action_mode, keyboard_enabled=policy == "keyboard"))
    fig.tight_layout()

    captured_frames: list[np.ndarray] = []

    def capture_frame() -> None:
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        captured_frames.append(frame)

    if show:
        plt.ion()
        _connect_keyboard(fig, keyboard_state)
        plt.show(block=False)

    if save_gif is not None:
        capture_frame()

    step_index = 0
    last_reward = 0.0
    while step_index < steps and not keyboard_state.quit_requested:
        if keyboard_state.reset_requested:
            observation, info = env.reset(seed=seed)
            instruction = str(observation["instruction"])
            positions = [np.asarray(info["drone_state"], dtype=np.float32)[:3]]
            keyboard_state.reset_requested = False
            step_index = 0
            last_reward = 0.0
        elif not keyboard_state.paused:
            action = _select_action(
                env,
                policy=policy,
                action_mode=action_mode,
                keyboard_state=keyboard_state,
                step_index=step_index,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            positions.append(np.asarray(info["drone_state"], dtype=np.float32)[:3])
            step_index += 1
            last_reward = float(reward)
            if terminated or truncated:
                controls.set_text(
                    _controls_text(action_mode, keyboard_enabled=policy == "keyboard")
                    + " | episode ended"
                )
                if policy != "keyboard":
                    break
        rgb_artist.set_data(np.asarray(observation["rgb"], dtype=np.uint8))
        depth_artist.set_data(_depth_to_rgb(np.asarray(observation["depth"], dtype=np.float32)))
        _update_topdown_axis(
            topdown_axis,
            positions=positions,
            goal_position=np.asarray(base_env.goal_position, dtype=np.float32),
        )
        overlay.set_text(
            _overlay_text(
                env_id=env_id,
                step_index=step_index,
                reward=last_reward,
                info=info,
                instruction=instruction,
                renderer_name=renderer_name,
                scene=scene,
                action_mode=action_mode,
                policy=policy,
            )
        )
        fig.canvas.draw_idle()
        if save_gif is not None:
            capture_frame()
        if show:
            plt.pause(1.0 / max(fps, 1e-3))
        else:
            if policy == "keyboard":
                break
        if keyboard_state.paused and not show:
            break

    env.close()

    saved_path: Path | None = None
    if save_gif is not None and captured_frames:
        saved_path = Path(save_gif)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        gif_figure = plt.figure(figsize=(15, 5))
        gif_axis = gif_figure.add_subplot(111)
        gif_axis.axis("off")
        gif_artists = [[gif_axis.imshow(frame, animated=True)] for frame in captured_frames]
        gif_animation = animation.ArtistAnimation(
            gif_figure,
            gif_artists,
            interval=int(1000.0 / max(fps, 1e-3)),
            blit=True,
        )
        gif_animation.save(saved_path, writer="pillow", fps=max(fps, 1e-3))
        plt.close(gif_figure)

    if show:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)
    return saved_path


def main() -> None:
    """Run the live-viewer CLI."""
    parser = build_parser()
    args = parser.parse_args()
    saved_path = run_live_viewer(
        env_id=args.env_id,
        scene=_normalize_scene(args.scene),
        steps=args.steps,
        seed=args.seed,
        policy=args.policy,
        fps=args.fps,
        save_gif=args.save_gif,
        show=not args.no_show,
        renderer_device=args.renderer_device,
        action_mode=args.action_mode,
    )
    if saved_path is not None:
        print(saved_path)


if __name__ == "__main__":  # pragma: no cover
    main()
