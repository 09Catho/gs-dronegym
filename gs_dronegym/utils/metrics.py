"""Evaluation metrics for GS-DroneGym benchmark episodes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Episode:
    """Trajectory and outcome record for a benchmark episode."""

    positions: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[float]
    success: bool
    collision: bool
    goal_position: np.ndarray
    n_steps: int


def success_rate(episodes: list[Episode]) -> float:
    """Compute the fraction of successful episodes.

    Args:
        episodes: Benchmark episodes.

    Returns:
        Success rate in ``[0, 1]``.
    """
    if not episodes:
        return 0.0
    return float(sum(1 for episode in episodes if episode.success) / len(episodes))


def path_length(episode: Episode) -> float:
    """Compute the cumulative trajectory length for one episode.

    Args:
        episode: Episode record.

    Returns:
        Path length in meters.
    """
    if len(episode.positions) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(episode.positions)):
        total += float(
            np.linalg.norm(
                np.asarray(episode.positions[idx], dtype=np.float32)
                - np.asarray(episode.positions[idx - 1], dtype=np.float32)
            )
        )
    return total


def spl(episodes: list[Episode]) -> float:
    """Compute Success weighted by Path Length (SPL).

    Args:
        episodes: Benchmark episodes.

    Returns:
        SPL score.
    """
    if not episodes:
        return 0.0

    scores: list[float] = []
    for episode in episodes:
        if not episode.success or not episode.positions:
            scores.append(0.0)
            continue
        start = np.asarray(episode.positions[0], dtype=np.float32)
        goal = np.asarray(episode.goal_position, dtype=np.float32)
        optimal = float(np.linalg.norm(goal - start))
        actual = path_length(episode)
        if actual <= 0.0 and optimal <= 0.0:
            scores.append(1.0)
        else:
            scores.append(optimal / max(actual, optimal, 1e-8))
    return float(sum(scores) / len(scores))


def collision_rate(episodes: list[Episode]) -> float:
    """Compute the fraction of episodes that collided.

    Args:
        episodes: Benchmark episodes.

    Returns:
        Collision rate in ``[0, 1]``.
    """
    if not episodes:
        return 0.0
    return float(sum(1 for episode in episodes if episode.collision) / len(episodes))


def avg_speed(episodes: list[Episode], obs_dt: float = 0.1) -> float:
    """Compute average speed across successful episodes.

    Args:
        episodes: Benchmark episodes.
        obs_dt: Observation interval in seconds.

    Returns:
        Mean speed in meters per second.
    """
    successful = [episode for episode in episodes if episode.success]
    if not successful:
        return 0.0
    speeds: list[float] = []
    for episode in successful:
        if episode.n_steps <= 0:
            continue
        speeds.append(path_length(episode) / max(episode.n_steps * obs_dt, 1e-8))
    if not speeds:
        return 0.0
    return float(np.mean(np.asarray(speeds, dtype=np.float32)))


@dataclass(slots=True)
class BenchmarkResult:
    """Aggregate benchmark metrics for one task."""

    task_id: str
    n_episodes: int
    success_rate: float
    spl: float
    collision_rate: float
    avg_speed: float
    raw_episodes: list[Episode]

    def to_dict(self) -> dict[str, object]:
        """Serialize the benchmark result as a JSON-safe dictionary.

        Returns:
            Dictionary representation of the result.
        """
        payload: dict[str, object] = {
            "task_id": self.task_id,
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "spl": self.spl,
            "collision_rate": self.collision_rate,
            "avg_speed": self.avg_speed,
            "raw_episodes": [],
        }
        raw: list[dict[str, object]] = []
        for episode in self.raw_episodes:
            item = asdict(episode)
            item["positions"] = [
                np.asarray(pos, dtype=np.float32).tolist() for pos in episode.positions
            ]
            item["actions"] = [
                np.asarray(action, dtype=np.float32).tolist()
                for action in episode.actions
            ]
            item["goal_position"] = np.asarray(episode.goal_position, dtype=np.float32).tolist()
            raw.append(item)
        payload["raw_episodes"] = raw
        return payload

    def to_json(self, path: str | Path) -> None:
        """Write the benchmark result to disk as JSON.

        Args:
            path: Output JSON path.
        """
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
