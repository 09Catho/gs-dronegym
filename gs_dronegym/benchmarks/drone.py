"""Live GS-DroneGym benchmark adapter and rollout exporter.

This adapter keeps the drone simulator as a first-class live benchmark while
normalizing rollouts into the shared trajectory schema used across other
benchmarks and datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np

from gs_dronegym.benchmarks.base import (
    BenchmarkAdapter,
    build_task_breakdown,
    call_policy,
    compute_core_metrics,
)
from gs_dronegym.data.dataset import load_common_dataset
from gs_dronegym.data.schema import (
    ActionSpec,
    BenchmarkReport,
    JsonValue,
    TaskSpec,
    TrajectoryEpisode,
    TrajectoryStep,
    infer_observation_spec,
)
from gs_dronegym.utils.metrics import Episode, avg_speed, collision_rate, path_length, spl

LOGGER = logging.getLogger(__name__)


def _normalize_observation(observation: dict[str, object]) -> dict[str, object]:
    """Copy an environment observation into a schema-safe dict.

    Args:
        observation: Environment observation.

    Returns:
        Copied observation.
    """
    normalized: dict[str, object] = {}
    for key, value in observation.items():
        normalized[key] = value.copy() if isinstance(value, np.ndarray) else value
    return normalized


def trajectory_to_nav_episode(episode: TrajectoryEpisode) -> Episode:
    """Convert a normalized trajectory episode into the legacy nav metric type.

    Args:
        episode: Common trajectory episode.

    Returns:
        Navigation metric episode.
    """
    positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    collision = False
    for step in episode.steps:
        state = step.observation.get("state")
        if isinstance(state, np.ndarray) and state.shape[0] >= 3:
            positions.append(np.asarray(state[:3], dtype=np.float32))
        actions.append(np.asarray(step.action, dtype=np.float32))
        rewards.append(float(step.reward))
        collision = collision or bool(step.info.get("collision", False))
    goal_state = episode.metadata.get("goal_position", [0.0, 0.0, 0.0])
    goal_position = np.asarray(goal_state, dtype=np.float32)
    return Episode(
        positions=positions,
        actions=actions,
        rewards=rewards,
        success=episode.success,
        collision=collision,
        goal_position=goal_position,
        n_steps=episode.n_steps,
    )


class DroneBenchmark(BenchmarkAdapter):
    """Benchmark adapter for live GS-DroneGym environments."""

    name = "gs_dronegym"
    embodiment = "drone"

    def __init__(
        self,
        env_id: str = "PointNav-v0",
        scene: str | Path | None = None,
        **env_kwargs: object,
    ) -> None:
        """Initialize the live drone benchmark adapter.

        Args:
            env_id: Registered environment ID.
            scene: Scene handle forwarded to :func:`gs_dronegym.make`.
            **env_kwargs: Additional environment kwargs.
        """
        self.env_id = env_id
        self.scene = scene
        self.env_kwargs = dict(env_kwargs)

    def make_env(self) -> object:
        """Create a new live GS-DroneGym environment.

        Returns:
            Gymnasium environment instance.
        """
        import gs_dronegym

        return gs_dronegym.make(self.env_id, scene=self.scene, **self.env_kwargs)

    def collect_episode(
        self,
        policy: object | None = None,
        seed: int | None = None,
    ) -> TrajectoryEpisode:
        """Run one live episode and export it into the common schema.

        Args:
            policy: Policy callable or object with ``predict``.
            seed: Optional environment seed.

        Returns:
            Exported trajectory episode.
        """
        env = self.make_env()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        observation, info = env.reset(seed=seed)
        initial_observation = _normalize_observation(cast(dict[str, object], observation))
        observation_spec = infer_observation_spec(initial_observation)
        action_spec = ActionSpec(
            shape=tuple(int(dim) for dim in env.action_space.shape),
            dtype=str(env.action_space.dtype),
            normalized=True,
            semantics="normalized waypoint delta or direct control action",
            metadata={
                "action_mode": cast(str, getattr(base_env, "action_mode", "waypoint"))
            },
        )
        task_spec = TaskSpec(
            task_id=str(base_env.task.task_id),
            benchmark_name=self.name,
            embodiment=self.embodiment,
            instruction=str(initial_observation.get("instruction", "")),
            description=f"Live rollout from {self.env_id}",
            metadata={"env_id": self.env_id},
        )
        steps: list[TrajectoryStep] = []
        current_observation = initial_observation
        terminated = False
        truncated = False
        step_index = 0
        while not terminated and not truncated:
            action = call_policy(policy, current_observation, action_dim=action_spec.shape[0])
            next_observation, reward, terminated, truncated, step_info = env.step(action)
            steps.append(
                TrajectoryStep(
                    observation=_normalize_observation(current_observation),
                    action=np.asarray(action, dtype=np.float32),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=cast(dict[str, object], step_info),
                    step_index=step_index,
                    benchmark_metrics={
                        "distance_to_goal": float(step_info.get("distance_to_goal", 0.0))
                    },
                )
            )
            current_observation = _normalize_observation(cast(dict[str, object], next_observation))
            step_index += 1
        metadata: dict[str, JsonValue] = {
            "env_id": self.env_id,
            "scene": "" if self.scene is None else str(self.scene),
            "goal_position": cast(
                JsonValue,
                np.asarray(
                    getattr(base_env, "goal_position", np.zeros(3)),
                    dtype=np.float32,
                ).tolist(),
            ),
        }
        success = bool(steps[-1].info.get("success", False)) if steps else False
        env.close()
        return TrajectoryEpisode(
            episode_id=f"{self.env_id}-{seed if seed is not None else 'rollout'}",
            benchmark_name=self.name,
            embodiment=self.embodiment,
            task=task_spec,
            action_spec=action_spec,
            observation_spec=observation_spec,
            steps=steps,
            success=success,
            split="eval",
            source=self.env_id,
            metadata=metadata,
            benchmark_metrics={
                "initial_distance_to_goal": float(info.get("distance_to_goal", 0.0))
            },
        )

    def load_dataset(self, source: str) -> list[TrajectoryEpisode]:
        """Load a saved normalized drone dataset.

        Args:
            source: Dataset path.

        Returns:
            Parsed episodes.
        """
        return load_common_dataset(source)

    def evaluate_policy(
        self,
        policy: object | None = None,
        n_episodes: int = 10,
        seed: int = 0,
    ) -> BenchmarkReport:
        """Evaluate a policy in live drone simulation.

        Args:
            policy: Policy callable or predictor object.
            n_episodes: Number of episodes to run.
            seed: Base RNG seed.

        Returns:
            Standardized benchmark report.
        """
        episodes = [
            self.collect_episode(policy=policy, seed=seed + episode_idx)
            for episode_idx in range(n_episodes)
        ]
        core_metrics = compute_core_metrics(episodes)
        nav_episodes = [trajectory_to_nav_episode(episode) for episode in episodes]
        benchmark_metrics = {
            "success_rate": core_metrics["success_rate"],
            "collision_rate": collision_rate(nav_episodes),
            "spl": spl(nav_episodes),
            "avg_speed": avg_speed(nav_episodes),
            "mean_path_length": float(
                np.mean(np.asarray([path_length(item) for item in nav_episodes], dtype=np.float32))
            )
            if nav_episodes
            else 0.0,
        }
        return BenchmarkReport(
            benchmark_name=self.name,
            embodiment=self.embodiment,
            n_episodes=len(episodes),
            core_metrics=core_metrics,
            benchmark_metrics=benchmark_metrics,
            task_breakdown=build_task_breakdown(episodes),
            metadata={
                "env_id": self.env_id,
                "scene": "" if self.scene is None else str(self.scene),
            },
            raw_results=[episode.to_dict() for episode in episodes],
        )
