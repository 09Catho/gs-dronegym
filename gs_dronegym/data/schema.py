"""Canonical trajectory and report schema for cross-benchmark experiments.

This module defines the shared data contracts used to normalize trajectories,
task metadata, and evaluation reports across GS-DroneGym, LIBERO, and
LeRobot-format datasets. Keeping a single schema allows the project to stay
drone-first while still supporting common training and reporting pipelines
across different robot embodiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _serialize_value(value: object) -> JsonValue:
    """Convert Python and NumPy values into a JSON-safe representation.

    Args:
        value: Value to serialize.

    Returns:
        JSON-safe value.

    Raises:
        TypeError: If the value type is not supported.
    """
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "ndarray",
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
            "data": cast(list[JsonValue], value.tolist()),
        }
    if isinstance(value, np.generic):
        return cast(JsonScalar, value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return cast(JsonScalar, value)
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    raise TypeError(f"Unsupported value type for serialization: {type(value)!r}")


def _deserialize_value(value: JsonValue) -> object:
    """Convert a JSON-safe value back into Python/NumPy values.

    Args:
        value: Serialized JSON value.

    Returns:
        Deserialized Python value.
    """
    if isinstance(value, dict):
        if value.get("__kind__") == "ndarray":
            dtype = np.dtype(cast(str, value["dtype"]))
            shape = tuple(int(dim) for dim in cast(list[JsonValue], value["shape"]))
            data = cast(list[JsonValue], value["data"])
            return np.asarray(data, dtype=dtype).reshape(shape)
        return {key: _deserialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    return value


def infer_observation_spec(observation: dict[str, object]) -> ObservationSpec:
    """Infer an observation spec from a concrete observation dictionary.

    Args:
        observation: Example observation dictionary.

    Returns:
        Inferred observation spec.
    """
    rgb_shape: tuple[int, int, int] | None = None
    depth_shape: tuple[int, int] | None = None
    state_shape: tuple[int, ...] | None = None
    modalities: list[str] = []
    metadata: dict[str, JsonValue] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            if key == "rgb" and value.ndim == 3:
                rgb_shape = tuple(int(dim) for dim in value.shape)
                modalities.append("rgb")
            elif key == "depth" and value.ndim == 2:
                depth_shape = tuple(int(dim) for dim in value.shape)
                modalities.append("depth")
            elif key == "state":
                state_shape = tuple(int(dim) for dim in value.shape)
                modalities.append("state")
            else:
                metadata[key] = _serialize_value(value)
        elif key == "instruction":
            modalities.append("instruction")
        else:
            metadata[key] = _serialize_value(value)
    return ObservationSpec(
        modalities=tuple(dict.fromkeys(modalities)),
        rgb_shape=rgb_shape,
        depth_shape=depth_shape,
        state_shape=state_shape,
        metadata=metadata,
    )


@dataclass(slots=True)
class TaskSpec:
    """Normalized task description shared across benchmarks."""

    task_id: str
    benchmark_name: str
    embodiment: str
    instruction: str = ""
    description: str = ""
    suite_name: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the task spec.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "task_id": self.task_id,
            "benchmark_name": self.benchmark_name,
            "embodiment": self.embodiment,
            "instruction": self.instruction,
            "description": self.description,
            "suite_name": self.suite_name,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, JsonValue]) -> TaskSpec:
        """Deserialize a task spec from a dictionary.

        Args:
            payload: Serialized task spec.

        Returns:
            Parsed task spec.
        """
        metadata = cast(dict[str, JsonValue], payload.get("metadata", {}))
        return cls(
            task_id=str(payload["task_id"]),
            benchmark_name=str(payload["benchmark_name"]),
            embodiment=str(payload["embodiment"]),
            instruction=str(payload.get("instruction", "")),
            description=str(payload.get("description", "")),
            suite_name=cast(str | None, payload.get("suite_name")),
            metadata=cast(dict[str, JsonValue], _deserialize_value(metadata)),
        )


@dataclass(slots=True)
class ActionSpec:
    """Normalized action-space description for a benchmark."""

    shape: tuple[int, ...]
    dtype: str = "float32"
    normalized: bool = True
    semantics: str = "continuous control action"
    name: str = "action"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the action spec.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "shape": [int(dim) for dim in self.shape],
            "dtype": self.dtype,
            "normalized": self.normalized,
            "semantics": self.semantics,
            "name": self.name,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, JsonValue]) -> ActionSpec:
        """Deserialize an action spec.

        Args:
            payload: Serialized action spec.

        Returns:
            Parsed action spec.
        """
        return cls(
            shape=tuple(int(dim) for dim in cast(list[JsonValue], payload["shape"])),
            dtype=str(payload.get("dtype", "float32")),
            normalized=bool(payload.get("normalized", True)),
            semantics=str(payload.get("semantics", "continuous control action")),
            name=str(payload.get("name", "action")),
            metadata=cast(
                dict[str, JsonValue],
                _deserialize_value(cast(dict[str, JsonValue], payload.get("metadata", {}))),
            ),
        )


@dataclass(slots=True)
class ObservationSpec:
    """Normalized observation-space description for a benchmark."""

    modalities: tuple[str, ...]
    rgb_shape: tuple[int, int, int] | None = None
    depth_shape: tuple[int, int] | None = None
    state_shape: tuple[int, ...] | None = None
    instruction_key: str = "instruction"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the observation spec.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "modalities": [str(name) for name in self.modalities],
            "rgb_shape": None if self.rgb_shape is None else [int(dim) for dim in self.rgb_shape],
            "depth_shape": None
            if self.depth_shape is None
            else [int(dim) for dim in self.depth_shape],
            "state_shape": None
            if self.state_shape is None
            else [int(dim) for dim in self.state_shape],
            "instruction_key": self.instruction_key,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, JsonValue]) -> ObservationSpec:
        """Deserialize an observation spec.

        Args:
            payload: Serialized observation spec.

        Returns:
            Parsed observation spec.
        """
        rgb_shape_raw = payload.get("rgb_shape")
        depth_shape_raw = payload.get("depth_shape")
        state_shape_raw = payload.get("state_shape")
        return cls(
            modalities=tuple(str(name) for name in cast(list[JsonValue], payload["modalities"])),
            rgb_shape=None
            if rgb_shape_raw is None
            else tuple(int(dim) for dim in cast(list[JsonValue], rgb_shape_raw)),
            depth_shape=None
            if depth_shape_raw is None
            else tuple(int(dim) for dim in cast(list[JsonValue], depth_shape_raw)),
            state_shape=None
            if state_shape_raw is None
            else tuple(int(dim) for dim in cast(list[JsonValue], state_shape_raw)),
            instruction_key=str(payload.get("instruction_key", "instruction")),
            metadata=cast(
                dict[str, JsonValue],
                _deserialize_value(cast(dict[str, JsonValue], payload.get("metadata", {}))),
            ),
        )


@dataclass(slots=True)
class TrajectoryStep:
    """One transition in a normalized trajectory."""

    observation: dict[str, object]
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)
    step_index: int = 0
    timestamp_s: float | None = None
    benchmark_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the trajectory step.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "observation": cast(
                dict[str, JsonValue],
                _serialize_value(self.observation),
            ),
            "action": cast(dict[str, JsonValue], _serialize_value(self.action)),
            "reward": float(self.reward),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "info": cast(dict[str, JsonValue], _serialize_value(self.info)),
            "step_index": int(self.step_index),
            "timestamp_s": self.timestamp_s,
            "benchmark_metrics": {
                key: float(value) for key, value in self.benchmark_metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, JsonValue]) -> TrajectoryStep:
        """Deserialize a trajectory step.

        Args:
            payload: Serialized step.

        Returns:
            Parsed trajectory step.
        """
        return cls(
            observation=cast(
                dict[str, object],
                _deserialize_value(cast(dict[str, JsonValue], payload["observation"])),
            ),
            action=np.asarray(
                _deserialize_value(cast(dict[str, JsonValue], payload["action"])),
                dtype=np.float32,
            ),
            reward=float(payload.get("reward", 0.0)),
            terminated=bool(payload.get("terminated", False)),
            truncated=bool(payload.get("truncated", False)),
            info=cast(
                dict[str, object],
                _deserialize_value(cast(dict[str, JsonValue], payload.get("info", {}))),
            ),
            step_index=int(payload.get("step_index", 0)),
            timestamp_s=cast(float | None, payload.get("timestamp_s")),
            benchmark_metrics={
                str(key): float(value)
                for key, value in cast(
                    dict[str, JsonValue], payload.get("benchmark_metrics", {})
                ).items()
            },
        )


@dataclass(slots=True)
class TrajectoryEpisode:
    """Canonical representation of one benchmark episode."""

    episode_id: str
    benchmark_name: str
    embodiment: str
    task: TaskSpec
    action_spec: ActionSpec
    observation_spec: ObservationSpec
    steps: list[TrajectoryStep]
    success: bool
    split: str = "train"
    source: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    benchmark_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        """Return the total return of the episode.

        Returns:
            Sum of per-step rewards.
        """
        return float(sum(step.reward for step in self.steps))

    @property
    def n_steps(self) -> int:
        """Return the number of transitions in the episode.

        Returns:
            Number of steps.
        """
        return len(self.steps)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the episode.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "episode_id": self.episode_id,
            "benchmark_name": self.benchmark_name,
            "embodiment": self.embodiment,
            "task": self.task.to_dict(),
            "action_spec": self.action_spec.to_dict(),
            "observation_spec": self.observation_spec.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "success": self.success,
            "split": self.split,
            "source": self.source,
            "metadata": _serialize_value(self.metadata),
            "benchmark_metrics": {
                key: float(value) for key, value in self.benchmark_metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, JsonValue]) -> TrajectoryEpisode:
        """Deserialize an episode.

        Args:
            payload: Serialized episode.

        Returns:
            Parsed episode.
        """
        return cls(
            episode_id=str(payload["episode_id"]),
            benchmark_name=str(payload["benchmark_name"]),
            embodiment=str(payload["embodiment"]),
            task=TaskSpec.from_dict(cast(dict[str, JsonValue], payload["task"])),
            action_spec=ActionSpec.from_dict(
                cast(dict[str, JsonValue], payload["action_spec"])
            ),
            observation_spec=ObservationSpec.from_dict(
                cast(dict[str, JsonValue], payload["observation_spec"])
            ),
            steps=[
                TrajectoryStep.from_dict(cast(dict[str, JsonValue], step_payload))
                for step_payload in cast(list[JsonValue], payload["steps"])
            ],
            success=bool(payload.get("success", False)),
            split=str(payload.get("split", "train")),
            source=str(payload.get("source", "")),
            metadata=cast(
                dict[str, JsonValue],
                _deserialize_value(cast(dict[str, JsonValue], payload.get("metadata", {}))),
            ),
            benchmark_metrics={
                str(key): float(value)
                for key, value in cast(
                    dict[str, JsonValue], payload.get("benchmark_metrics", {})
                ).items()
            },
        )

    def to_json(self, path: str | Path) -> None:
        """Write the episode to disk as JSON.

        Args:
            path: Output path.
        """
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> TrajectoryEpisode:
        """Load an episode from disk.

        Args:
            path: Input JSON path.

        Returns:
            Parsed episode.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(cast(dict[str, JsonValue], payload))


@dataclass(slots=True)
class BenchmarkReport:
    """Standard benchmark report spanning different robot embodiments."""

    benchmark_name: str
    embodiment: str
    n_episodes: int
    core_metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    task_breakdown: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    raw_results: list[dict[str, JsonValue]] = field(default_factory=list)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the report.

        Returns:
            JSON-safe dictionary.
        """
        return {
            "benchmark_name": self.benchmark_name,
            "embodiment": self.embodiment,
            "n_episodes": self.n_episodes,
            "core_metrics": {key: float(value) for key, value in self.core_metrics.items()},
            "benchmark_metrics": {
                key: float(value) for key, value in self.benchmark_metrics.items()
            },
            "task_breakdown": self.task_breakdown,
            "metadata": self.metadata,
            "raw_results": self.raw_results,
        }

    def to_json(self, path: str | Path) -> None:
        """Write the report to disk as JSON.

        Args:
            path: Output report path.
        """
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
