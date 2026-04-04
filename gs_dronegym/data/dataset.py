"""Dataset IO and replay helpers for the shared trajectory schema.

This module keeps cross-benchmark trajectory storage lightweight and explicit.
It provides a canonical on-disk JSON format for GS-DroneGym trajectories while
also dispatching to optional benchmark-specific dataset loaders for LIBERO and
LeRobot.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from gs_dronegym.data.schema import JsonValue, TrajectoryEpisode, TrajectoryStep

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TransitionRecord:
    """Replay-friendly view of one transition from a trajectory dataset."""

    episode_id: str
    step_index: int
    observation: dict[str, object]
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


def iter_transitions(episodes: Iterable[TrajectoryEpisode]) -> Iterator[TransitionRecord]:
    """Iterate over step-level transitions from a list of episodes.

    Args:
        episodes: Trajectory episodes to flatten.

    Yields:
        Per-step transition records.
    """
    for episode in episodes:
        for step in episode.steps:
            yield TransitionRecord(
                episode_id=episode.episode_id,
                step_index=step.step_index,
                observation=step.observation,
                action=np.asarray(step.action, dtype=np.float32),
                reward=float(step.reward),
                terminated=bool(step.terminated),
                truncated=bool(step.truncated),
                info=dict(step.info),
            )


def save_dataset(episodes: list[TrajectoryEpisode], path: str | Path) -> Path:
    """Write a normalized dataset to disk.

    Args:
        episodes: Episodes to serialize.
        path: Output JSON path or output directory.

    Returns:
        The written JSON file path.
    """
    destination = Path(path)
    output_path = destination / "dataset.json" if destination.suffix == "" else destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, JsonValue] = {
        "format": "gs_dronegym.common.v1",
        "n_episodes": len(episodes),
        "episodes": [episode.to_dict() for episode in episodes],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d normalized episodes to %s", len(episodes), output_path)
    return output_path


def load_common_dataset(path: str | Path) -> list[TrajectoryEpisode]:
    """Load a dataset saved in the canonical GS-DroneGym JSON format.

    Args:
        path: JSON path or directory containing ``dataset.json``.

    Returns:
        Parsed trajectory episodes.
    """
    source_path = Path(path)
    input_path = source_path / "dataset.json" if source_path.is_dir() else source_path
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    episodes_payload = cast(list[JsonValue], payload.get("episodes", []))
    return [
        TrajectoryEpisode.from_dict(cast(dict[str, JsonValue], episode_payload))
        for episode_payload in episodes_payload
    ]


def summarize_dataset(episodes: list[TrajectoryEpisode]) -> dict[str, JsonValue]:
    """Compute a JSON-safe summary for a normalized dataset.

    Args:
        episodes: Dataset episodes.

    Returns:
        Summary dictionary suitable for CLI printing.
    """
    if not episodes:
        return {
            "n_episodes": 0,
            "benchmarks": [],
            "embodiments": [],
            "avg_steps": 0.0,
            "task_ids": [],
            "splits": {},
        }
    benchmarks = sorted({episode.benchmark_name for episode in episodes})
    embodiments = sorted({episode.embodiment for episode in episodes})
    task_ids = sorted({episode.task.task_id for episode in episodes})
    avg_steps = float(
        np.mean(np.asarray([episode.n_steps for episode in episodes], dtype=np.float32))
    )
    splits: dict[str, int] = {}
    for episode in episodes:
        splits[episode.split] = splits.get(episode.split, 0) + 1
    return {
        "n_episodes": len(episodes),
        "benchmarks": benchmarks,
        "embodiments": embodiments,
        "avg_steps": avg_steps,
        "task_ids": task_ids,
        "splits": splits,
    }


def load_dataset(
    source: str | Path,
    format: Literal["gs_dronegym", "libero", "lerobot"],
) -> list[TrajectoryEpisode]:
    """Load a dataset from one of the supported benchmark formats.

    Args:
        source: Dataset source path or benchmark-specific handle.
        format: Dataset format name.

    Returns:
        Parsed trajectory episodes.

    Raises:
        ValueError: If the dataset format is unsupported.
    """
    if format == "gs_dronegym":
        return load_common_dataset(source)
    if format == "libero":
        from gs_dronegym.benchmarks.libero import load_libero_dataset

        return load_libero_dataset(source)
    if format == "lerobot":
        from gs_dronegym.benchmarks.lerobot import load_lerobot_dataset

        return load_lerobot_dataset(source)
    raise ValueError(f"Unsupported dataset format: {format}")


def replay_episode(episode: TrajectoryEpisode) -> list[TrajectoryStep]:
    """Return a shallow copy of the steps in an episode for replay code.

    Args:
        episode: Episode to replay.

    Returns:
        List of trajectory steps.
    """
    return list(episode.steps)
