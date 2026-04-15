"""CLI for evaluating the geometric expert in live GS-DroneGym environments.

This command is primarily a control-stack diagnostic. If the expert cannot
solve PointNav, learned policies should not be expected to work either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

import gs_dronegym
from gs_dronegym.data.planner import ExpertPlanner, PlannerConfig
from gs_dronegym.utils.metrics import Episode, avg_speed, collision_rate, path_length, spl


def _scene_arg(value: str | None) -> str | Path | None:
    """Convert CLI scene text into a scene handle."""
    if value is None or value.lower() in {"none", "null"}:
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    """Create the expert evaluation parser."""
    parser = argparse.ArgumentParser(description="Evaluate the geometric expert live.")
    parser.add_argument("--env-id", default="PointNav-v0", help="Environment ID.")
    parser.add_argument("--scene", default=None, help="Scene name/path or None.")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--renderer-device", default="cpu", help="Renderer device.")
    parser.add_argument("--width", type=int, default=64, help="Observation width.")
    parser.add_argument("--height", type=int, default=64, help="Observation height.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def evaluate_expert(args: argparse.Namespace) -> dict[str, object]:
    """Run expert rollouts and return a JSON-safe report."""
    planner = ExpertPlanner(PlannerConfig())
    episodes: list[Episode] = []
    episode_rows: list[dict[str, object]] = []
    scene = _scene_arg(cast(str | None, args.scene))

    for episode_index in range(args.n_episodes):
        env = gs_dronegym.make(
            args.env_id,
            scene=scene,
            renderer_device=args.renderer_device,
            image_size=(args.width, args.height),
        )
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        obs, info = env.reset(seed=args.seed + episode_index)
        positions = [np.asarray(obs["state"], dtype=np.float32)[:3].copy()]
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        terminated = False
        truncated = False
        collision = False
        final_info = dict(info)

        while not (terminated or truncated):
            state = np.asarray(obs["state"], dtype=np.float32)
            waypoint, _ = planner.plan_waypoint(
                state=state,
                goal_position=np.asarray(base_env.goal_position, dtype=np.float32),
                task=base_env.task,
                scene_bbox=np.asarray(base_env.scene_bbox, dtype=np.float32),
                obs_dt=float(base_env.dynamics.config.obs_dt),
            )
            action = planner.normalized_waypoint_action(state, waypoint)
            obs, reward, terminated, truncated, final_info = env.step(action)
            positions.append(np.asarray(obs["state"], dtype=np.float32)[:3].copy())
            actions.append(action.astype(np.float32))
            rewards.append(float(reward))
            collision = collision or bool(final_info.get("collision", False))

        goal_position = np.asarray(base_env.goal_position, dtype=np.float32)
        success = bool(final_info.get("success", False))
        episode = Episode(
            positions=positions,
            actions=actions,
            rewards=rewards,
            success=success,
            collision=collision,
            goal_position=goal_position,
            n_steps=len(actions),
        )
        episodes.append(episode)
        episode_rows.append(
            {
                "episode": episode_index,
                "success": success,
                "collision": collision,
                "n_steps": len(actions),
                "return": float(np.sum(np.asarray(rewards, dtype=np.float32))),
                "path_length": path_length(episode),
                "final_distance_to_goal": float(final_info.get("distance_to_goal", 0.0)),
            }
        )
        env.close()

    report = {
        "env_id": args.env_id,
        "scene": "" if scene is None else str(scene),
        "n_episodes": int(args.n_episodes),
        "success_rate": float(sum(item.success for item in episodes) / max(len(episodes), 1)),
        "collision_rate": collision_rate(episodes),
        "spl": spl(episodes),
        "avg_speed": avg_speed(episodes),
        "mean_path_length": float(
            np.mean(np.asarray([path_length(item) for item in episodes], dtype=np.float32))
        )
        if episodes
        else 0.0,
        "episodes": episode_rows,
    }
    return report


def main() -> None:
    """Run expert evaluation from the command line."""
    parser = build_parser()
    args = parser.parse_args()
    report = evaluate_expert(args)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
