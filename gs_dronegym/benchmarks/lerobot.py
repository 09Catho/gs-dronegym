"""LeRobot-format dataset loader and dataset-level evaluator.

This adapter focuses on LeRobot as a dataset format rather than a live
simulation backend. It can ingest metadata-only local datasets directly and can
optionally use the official LeRobot package when installed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np

from gs_dronegym.benchmarks.base import (
    BenchmarkAdapter,
    build_task_breakdown,
    call_policy,
    compute_core_metrics,
)
from gs_dronegym.data.schema import (
    ActionSpec,
    BenchmarkReport,
    JsonValue,
    TaskSpec,
    TrajectoryEpisode,
    TrajectoryStep,
    infer_observation_spec,
)

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None


def _normalize_column_name(name: str) -> str:
    """Normalize dataset column names to dot-separated form.

    Args:
        name: Raw column name.

    Returns:
        Normalized column name.
    """
    return name.replace("/", ".")


def _coerce_sequence(value: object) -> np.ndarray:
    """Convert a column value into a float32 vector.

    Args:
        value: Input scalar or sequence.

    Returns:
        Float32 NumPy array.
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    return np.asarray([value], dtype=np.float32)


def _load_tasks_manifest(path: Path) -> dict[int, str]:
    """Load optional LeRobot task labels.

    Args:
        path: Dataset root path.

    Returns:
        Mapping from task index to language string.
    """
    manifest_path = path / "meta" / "tasks.jsonl"
    if not manifest_path.exists():
        return {}
    mapping: dict[int, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        mapping[int(payload["task_index"])] = str(payload["task"])
    return mapping


def _load_episode_manifest(path: Path) -> dict[int, dict[str, JsonValue]]:
    """Load optional per-episode metadata for LeRobot datasets.

    Args:
        path: Dataset root path.

    Returns:
        Mapping from episode index to metadata.
    """
    manifest_path = path / "meta" / "episodes.jsonl"
    if not manifest_path.exists():
        return {}
    mapping: dict[int, dict[str, JsonValue]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = cast(dict[str, JsonValue], json.loads(line))
        mapping[int(payload["episode_index"])] = payload
    return mapping


def _extract_episode_index(path: Path) -> int:
    """Extract the numeric episode index from a LeRobot parquet filename.

    Args:
        path: Parquet path.

    Returns:
        Episode index.
    """
    stem = path.stem
    return int(stem.split("_")[-1])


def _build_steps_from_rows(
    rows: list[dict[str, object]],
    instruction: str,
) -> tuple[list[TrajectoryStep], ActionSpec, dict[str, object]]:
    """Convert tabular LeRobot rows into normalized trajectory steps.

    Args:
        rows: Row dictionaries from one episode parquet file.
        instruction: Episode instruction string.

    Returns:
        Tuple of steps, action spec, and example observation.
    """
    steps: list[TrajectoryStep] = []
    action_shape: tuple[int, ...] | None = None
    example_observation: dict[str, object] = {}
    for step_index, row in enumerate(rows):
        observation: dict[str, object] = {"instruction": instruction}
        action_parts: list[np.ndarray] = []
        state_parts: list[np.ndarray] = []
        info: dict[str, object] = {}
        reward = float(row.get("reward", 0.0))
        terminated = bool(row.get("done", False) or row.get("terminated", False))
        truncated = bool(row.get("truncated", False))
        for raw_key, value in row.items():
            key = _normalize_column_name(raw_key)
            if key in {"task", "instruction"}:
                observation["instruction"] = str(value)
            elif key.startswith("observation.images.") or key.startswith("observation.image."):
                image_key = key.split(".", 2)[-1]
                observation[image_key] = value
            elif key == "observation.state" or key.startswith("observation.state."):
                state_parts.append(_coerce_sequence(value))
            elif key == "action" or key.startswith("action."):
                action_parts.append(_coerce_sequence(value))
            elif key not in {"reward", "done", "terminated", "truncated"}:
                info[key] = value
        if state_parts:
            observation["state"] = np.concatenate(state_parts, dtype=np.float32)
        action = (
            np.concatenate(action_parts, dtype=np.float32)
            if action_parts
            else np.zeros(1, dtype=np.float32)
        )
        action_shape = tuple(int(dim) for dim in action.shape)
        if not example_observation:
            example_observation = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
        steps.append(
            TrajectoryStep(
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                step_index=step_index,
            )
        )
    spec = ActionSpec(
        shape=action_shape or (1,),
        normalized=False,
        semantics="LeRobot dataset action",
    )
    return steps, spec, example_observation


def load_lerobot_dataset(source: str | Path) -> list[TrajectoryEpisode]:
    """Load a LeRobot-format dataset into the common schema.

    Args:
        source: Local dataset root.

    Returns:
        Parsed trajectory episodes.

    Raises:
        RuntimeError: If ``pyarrow`` is unavailable.
        FileNotFoundError: If the dataset structure cannot be found.
    """
    root = Path(source)
    if pq is None:  # pragma: no cover - import-gated
        raise RuntimeError(
            "LeRobot local dataset loading requires pyarrow. Install the [lerobot] extra first."
        )
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"LeRobot dataset root is missing a data directory: {root}")
    info_path = root / "meta" / "info.json"
    dataset_info = (
        cast(dict[str, JsonValue], json.loads(info_path.read_text(encoding="utf-8")))
        if info_path.exists()
        else {}
    )
    task_manifest = _load_tasks_manifest(root)
    episode_manifest = _load_episode_manifest(root)
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet episodes found under {data_dir}")

    episodes: list[TrajectoryEpisode] = []
    for parquet_path in parquet_files:
        episode_index = _extract_episode_index(parquet_path)
        metadata = episode_manifest.get(episode_index, {})
        task_text = task_manifest.get(
            int(metadata.get("task_index", -1)) if "task_index" in metadata else -1,
            str(metadata.get("task", dataset_info.get("robot_type", "lerobot task"))),
        )
        table = pq.read_table(parquet_path)
        rows = cast(list[dict[str, object]], table.to_pylist())
        steps, action_spec, example_observation = _build_steps_from_rows(
            rows,
            instruction=task_text,
        )
        if example_observation:
            observation_spec = infer_observation_spec(example_observation)
        else:
            observation_spec = infer_observation_spec({"instruction": task_text})
        task_spec = TaskSpec(
            task_id=str(metadata.get("task", f"episode_{episode_index}")),
            benchmark_name="lerobot",
            embodiment=str(dataset_info.get("robot_type", "robot")),
            instruction=task_text,
            description="LeRobot dataset episode",
            metadata={"episode_index": episode_index},
        )
        episodes.append(
            TrajectoryEpisode(
                episode_id=f"lerobot:{episode_index}",
                benchmark_name="lerobot",
                embodiment=str(dataset_info.get("robot_type", "robot")),
                task=task_spec,
                action_spec=action_spec,
                observation_spec=observation_spec,
                steps=steps,
                success=bool(metadata.get("success", True)),
                split=str(metadata.get("split", "train")),
                source=str(root),
                metadata={"dataset_info": dataset_info, "episode_metadata": metadata},
            )
        )
    return episodes


def _flatten_predicted_actions(
    policy: object | None,
    episodes: Iterable[TrajectoryEpisode],
) -> tuple[np.ndarray, np.ndarray]:
    """Run a policy over a dataset and collect predictions and targets.

    Args:
        policy: Policy callable or predictor object.
        episodes: Episodes to score.

    Returns:
        Predicted and target action arrays.
    """
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    action_dim = 1
    for episode in episodes:
        action_dim = episode.action_spec.shape[0]
        for step in episode.steps:
            prediction = call_policy(policy, step.observation, action_dim=action_dim)
            predictions.append(np.asarray(prediction, dtype=np.float32).reshape(-1))
            targets.append(np.asarray(step.action, dtype=np.float32).reshape(-1))
    if not predictions:
        empty = np.zeros((0, action_dim), dtype=np.float32)
        return empty, empty.copy()
    return np.stack(predictions).astype(np.float32), np.stack(targets).astype(np.float32)


class LeRobotBenchmark(BenchmarkAdapter):
    """Dataset-first benchmark adapter for LeRobot-format datasets."""

    name = "lerobot"
    embodiment = "robot"

    def load_dataset(self, source: str) -> list[TrajectoryEpisode]:
        """Load LeRobot episodes from a local dataset root.

        Args:
            source: Local dataset root.

        Returns:
            Parsed episodes.
        """
        return load_lerobot_dataset(source)

    def evaluate_policy(
        self,
        policy: object | None = None,
        source: str | None = None,
        episodes: list[TrajectoryEpisode] | None = None,
    ) -> BenchmarkReport:
        """Evaluate a policy against a LeRobot-format dataset.

        Args:
            policy: Policy callable or predictor object.
            source: Dataset source path.
            episodes: Optional preloaded episodes.

        Returns:
            Standardized benchmark report.

        Raises:
            ValueError: If neither ``source`` nor ``episodes`` is provided.
        """
        if episodes is None:
            if source is None:
                raise ValueError(
                    "Either source or episodes must be provided for LeRobot evaluation."
                )
            episodes = self.load_dataset(source)
        core_metrics = compute_core_metrics(episodes)
        predictions, targets = _flatten_predicted_actions(policy, episodes)
        action_mse = float(np.mean((predictions - targets) ** 2)) if targets.size > 0 else 0.0
        action_mae = float(np.mean(np.abs(predictions - targets))) if targets.size > 0 else 0.0
        benchmark_metrics = {
            "action_mse": action_mse,
            "action_mae": action_mae,
        }
        return BenchmarkReport(
            benchmark_name=self.name,
            embodiment=episodes[0].embodiment if episodes else self.embodiment,
            n_episodes=len(episodes),
            core_metrics=core_metrics,
            benchmark_metrics=benchmark_metrics,
            task_breakdown=build_task_breakdown(episodes),
            metadata={"source": source or "in_memory"},
            raw_results=[episode.to_dict() for episode in episodes],
        )
