"""LIBERO dataset and evaluation bridge.

This adapter intentionally does not reimplement the LIBERO manipulation
environment. Instead it normalizes LIBERO demonstrations into the shared
trajectory schema and exposes a best-effort evaluation bridge that can use the
official LIBERO installation when available.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
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
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None


def _ensure_h5py() -> None:
    """Require the optional ``h5py`` dependency.

    Raises:
        RuntimeError: If ``h5py`` is not installed.
    """
    if h5py is None:  # pragma: no cover - import-gated
        raise RuntimeError(
            "LIBERO dataset support requires h5py. Install the [libero] extra first."
        )


def _load_suite_manifest(path: Path) -> dict[str, JsonValue]:
    """Load an optional suite manifest file alongside LIBERO demos.

    Args:
        path: Dataset root path.

    Returns:
        Parsed manifest dictionary, or an empty dictionary if missing.
    """
    manifest_path = path / "suite_manifest.json"
    if not manifest_path.exists():
        return {}
    return cast(dict[str, JsonValue], json.loads(manifest_path.read_text(encoding="utf-8")))


def _extract_instruction(
    demo_group: object,
    root_attrs: dict[str, object],
    default_task: str,
) -> str:
    """Extract a language instruction from common LIBERO metadata fields.

    Args:
        demo_group: Demonstration group.
        root_attrs: Root HDF5 attributes.
        default_task: Fallback task label.

    Returns:
        Best-effort instruction string.
    """
    demo_attrs = getattr(demo_group, "attrs", {})
    for key in ("language_instruction", "language", "instruction", "lang"):
        if key in demo_attrs:
            return str(demo_attrs[key])
        if key in root_attrs:
            return str(root_attrs[key])
    return f"complete {default_task}"


def _extract_observation_step(obs_group: object, index: int) -> dict[str, object]:
    """Extract one observation dictionary from a LIBERO demo group.

    Args:
        obs_group: HDF5 observation group.
        index: Step index.

    Returns:
        Observation dictionary.
    """
    observation: dict[str, object] = {}
    for key in obs_group.keys():
        data = np.asarray(obs_group[key][index])
        if key.endswith("rgb") or key.endswith("image"):
            observation["rgb"] = data.astype(np.uint8)
        elif key.endswith("depth"):
            observation["depth"] = data.astype(np.float32)
        else:
            observation[key] = data.astype(np.float32)
    if "state" not in observation:
        state_parts = [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in observation.values()
            if isinstance(value, np.ndarray) and value.ndim <= 2 and value.dtype != np.uint8
        ]
        if state_parts:
            observation["state"] = np.concatenate(state_parts, dtype=np.float32)
    return observation


def load_libero_dataset(source: str | Path) -> list[TrajectoryEpisode]:
    """Load LIBERO demonstrations into the common trajectory schema.

    Args:
        source: LIBERO HDF5 file or directory containing demonstration files.

    Returns:
        Parsed trajectory episodes.

    Raises:
        RuntimeError: If ``h5py`` is unavailable.
        FileNotFoundError: If no HDF5 files are discovered.
    """
    _ensure_h5py()
    root_path = Path(source)
    files = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.hdf5"))
    if not files:
        files = sorted(root_path.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No LIBERO HDF5 files found under {root_path}")
    suite_manifest = _load_suite_manifest(root_path if root_path.is_dir() else root_path.parent)

    episodes: list[TrajectoryEpisode] = []
    for file_path in files:
        with h5py.File(file_path, "r") as handle:  # type: ignore[union-attr]
            data_group = handle["data"] if "data" in handle else handle
            root_attrs = {str(key): handle.attrs[key] for key in handle.attrs.keys()}
            for demo_key in sorted(data_group.keys()):
                demo_group = data_group[demo_key]
                actions = np.asarray(demo_group["actions"], dtype=np.float32)
                rewards = (
                    np.asarray(demo_group["rewards"], dtype=np.float32)
                    if "rewards" in demo_group
                    else np.zeros(actions.shape[0], dtype=np.float32)
                )
                dones = (
                    np.asarray(demo_group["dones"], dtype=np.bool_)
                    if "dones" in demo_group
                    else np.zeros(actions.shape[0], dtype=np.bool_)
                )
                if actions.shape[0] > 0:
                    dones[-1] = True
                obs_group = demo_group["obs"] if "obs" in demo_group else None
                task_name = str(
                    demo_group.attrs.get("task_name", root_attrs.get("task_name", file_path.stem))
                )
                instruction = _extract_instruction(demo_group, root_attrs, task_name)
                example_observation = (
                    _extract_observation_step(obs_group, 0) if obs_group is not None else {}
                )
                example_observation["instruction"] = instruction
                observation_spec = infer_observation_spec(example_observation)
                action_spec = ActionSpec(
                    shape=tuple(int(dim) for dim in actions.shape[1:]),
                    dtype=str(actions.dtype),
                    normalized=False,
                    semantics="LIBERO control action",
                )
                steps: list[TrajectoryStep] = []
                for index in range(actions.shape[0]):
                    observation = (
                        _extract_observation_step(obs_group, index) if obs_group is not None else {}
                    )
                    observation["instruction"] = instruction
                    step_info = {
                        "dataset_file": str(file_path),
                        "demo_key": str(demo_key),
                    }
                    steps.append(
                        TrajectoryStep(
                            observation=observation,
                            action=np.asarray(actions[index], dtype=np.float32),
                            reward=float(rewards[index]),
                            terminated=bool(dones[index]),
                            truncated=False,
                            info=step_info,
                            step_index=index,
                        )
                    )
                task_spec = TaskSpec(
                    task_id=task_name,
                    benchmark_name="libero",
                    embodiment="manipulator",
                    instruction=instruction,
                    suite_name=cast(str | None, suite_manifest.get("suite_name")),
                    metadata={"dataset_file": str(file_path)},
                )
                success_value = demo_group.attrs.get("success", bool(np.any(rewards > 0.0)))
                episodes.append(
                    TrajectoryEpisode(
                        episode_id=f"{file_path.stem}:{demo_key}",
                        benchmark_name="libero",
                        embodiment="manipulator",
                        task=task_spec,
                        action_spec=action_spec,
                        observation_spec=observation_spec,
                        steps=steps,
                        success=bool(success_value),
                        split=str(demo_group.attrs.get("split", "train")),
                        source=str(file_path),
                        metadata={
                            "dataset_file": str(file_path),
                            "demo_key": str(demo_key),
                            "suite_manifest": suite_manifest,
                        },
                    )
                )
    return episodes


class LiberoBenchmark(BenchmarkAdapter):
    """Optional bridge for LIBERO datasets and live evaluation."""

    name = "libero"
    embodiment = "manipulator"

    def __init__(
        self,
        suite_name: str = "libero_spatial",
        env_factory: Callable[[str], object] | None = None,
    ) -> None:
        """Initialize the LIBERO adapter.

        Args:
            suite_name: LIBERO suite identifier.
            env_factory: Optional custom environment factory for evaluation.
        """
        self.suite_name = suite_name
        self.env_factory = env_factory

    def load_dataset(self, source: str) -> list[TrajectoryEpisode]:
        """Load LIBERO trajectories.

        Args:
            source: HDF5 file or dataset directory.

        Returns:
            Parsed episodes.
        """
        return load_libero_dataset(source)

    def list_suite_tasks(self) -> list[TaskSpec]:
        """Query suite metadata from an installed LIBERO package when available.

        Returns:
            Available task specs for the suite.
        """
        try:  # pragma: no cover - optional dependency
            from libero.libero import benchmark as libero_benchmark
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Official LIBERO package not installed. Install the [libero] extra "
                "or provide a custom env_factory."
            ) from exc

        benchmark_dict = libero_benchmark.get_benchmark_dict()
        if self.suite_name not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO suite: {self.suite_name}")
        suite = benchmark_dict[self.suite_name]()
        task_specs: list[TaskSpec] = []
        for task_id in range(int(suite.n_tasks)):
            task = suite.get_task(task_id)
            task_specs.append(
                TaskSpec(
                    task_id=str(getattr(task, "name", f"task_{task_id}")),
                    benchmark_name=self.name,
                    embodiment=self.embodiment,
                    instruction=str(getattr(task, "language", getattr(task, "problem_name", ""))),
                    suite_name=self.suite_name,
                )
            )
        return task_specs

    def evaluate_policy(
        self,
        policy: object | None = None,
        source: str | None = None,
        max_episodes: int = 10,
    ) -> BenchmarkReport:
        """Evaluate a policy against LIBERO data or environments.

        Args:
            policy: Policy callable or predictor object.
            source: Dataset source path. When provided, a dataset-based report is
                returned. When omitted, a custom or official LIBERO env bridge is
                required.
            max_episodes: Maximum number of episodes to evaluate.

        Returns:
            Normalized benchmark report.
        """
        if source is not None:
            episodes = self.load_dataset(source)[:max_episodes]
            core_metrics = compute_core_metrics(episodes)
            benchmark_metrics = {
                "dataset_success_rate": core_metrics["success_rate"],
                "mean_reward": core_metrics["mean_return"],
            }
            return BenchmarkReport(
                benchmark_name=self.name,
                embodiment=self.embodiment,
                n_episodes=len(episodes),
                core_metrics=core_metrics,
                benchmark_metrics=benchmark_metrics,
                task_breakdown=build_task_breakdown(episodes),
                metadata={"suite_name": self.suite_name, "source": source},
                raw_results=[episode.to_dict() for episode in episodes],
            )

        if self.env_factory is None:
            raise RuntimeError(
                "Live LIBERO evaluation requires a custom env_factory or an explicit "
                "dataset source. The adapter intentionally does not reimplement LIBERO."
            )

        task_specs = self.list_suite_tasks()
        collected: list[TrajectoryEpisode] = []
        for task_spec in task_specs[:max_episodes]:
            env = self.env_factory(task_spec.task_id)
            reset_output = env.reset()
            observation = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            observation_dict = cast(dict[str, object], observation)
            observation_dict["instruction"] = task_spec.instruction
            action = call_policy(policy, observation_dict, action_dim=7)
            step_output = env.step(action)
            next_observation = cast(dict[str, object], step_output[0])
            reward = float(step_output[1])
            terminated = bool(step_output[2])
            truncated = bool(step_output[3]) if len(step_output) > 4 else False
            info = cast(dict[str, object], step_output[-1])
            collected.append(
                TrajectoryEpisode(
                    episode_id=f"{task_spec.task_id}:live",
                    benchmark_name=self.name,
                    embodiment=self.embodiment,
                    task=task_spec,
                    action_spec=ActionSpec(shape=(7,), normalized=False, semantics="LIBERO action"),
                    observation_spec=infer_observation_spec(observation_dict),
                    steps=[
                        TrajectoryStep(
                            observation=observation_dict,
                            action=np.asarray(action, dtype=np.float32),
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            info=info,
                            step_index=0,
                        )
                    ],
                    success=bool(info.get("success", terminated)),
                    split="eval",
                    source=self.suite_name,
                    metadata={"next_observation": next_observation},
                )
            )
        core_metrics = compute_core_metrics(collected)
        return BenchmarkReport(
            benchmark_name=self.name,
            embodiment=self.embodiment,
            n_episodes=len(collected),
            core_metrics=core_metrics,
            benchmark_metrics={"live_eval_success_rate": core_metrics["success_rate"]},
            task_breakdown=build_task_breakdown(collected),
            metadata={"suite_name": self.suite_name},
            raw_results=[episode.to_dict() for episode in collected],
        )
