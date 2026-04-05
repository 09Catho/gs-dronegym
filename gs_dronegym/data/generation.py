"""Synthetic VLA-AN-like dataset generation and validation for GS-DroneGym.

This module turns the live GS-DroneGym simulator into a dataset factory for
waypoint-supervised aerial navigation research. It generates staged curricula,
writes sharded Parquet datasets with external RGB/depth media, preserves debug
episodes in the common trajectory format, and validates generated datasets for
training and reporting workflows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

try:
    import pyarrow as pa
    from pyarrow import parquet as pq
except ImportError:  # pragma: no cover - optional import path
    pa = None
    pq = None

from gs_dronegym.data.planner import ExpertPlanner, PlannerConfig
from gs_dronegym.data.schema import (
    ActionSpec,
    JsonValue,
    ObservationSpec,
    TaskSpec,
    TrajectoryEpisode,
    TrajectoryStep,
    infer_observation_spec,
)
from gs_dronegym.scene import BUILTIN_SCENES, SceneLoader
from gs_dronegym.tasks import (
    BaseTask,
    DynamicFollowTask,
    NarrowCorridorTask,
    ObjectNavTask,
    ObstacleSlalomTask,
    PointNavTask,
    TaskConfig,
)

LOGGER = logging.getLogger(__name__)

FORMAT_VERSION = "gs_dronegym.synthetic.v1"


def _require_pyarrow() -> None:
    """Raise a helpful error if pyarrow is not available."""
    if pa is None or pq is None:
        raise ImportError(
            "pyarrow is required for synthetic dataset generation. "
            "Install it with `pip install pyarrow` or `pip install gs-dronegym[benchmarks]`."
        )


def _json_default(value: object) -> JsonValue:
    """Serialize numpy values for JSON dumps."""
    if isinstance(value, np.ndarray):
        return cast(JsonValue, value.tolist())
    if isinstance(value, np.generic):
        return cast(JsonValue, value.item())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON file with safe serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


@dataclass(slots=True)
class SceneSelectionConfig:
    """Scene source and split configuration for dataset generation."""

    sources: tuple[str, ...]
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the scene selection config."""
        return {
            "sources": list(self.sources),
            "train_fraction": float(self.train_fraction),
            "val_fraction": float(self.val_fraction),
            "test_fraction": float(self.test_fraction),
        }


@dataclass(slots=True)
class CurriculumStageConfig:
    """One curriculum stage for VLA-AN-like synthetic generation."""

    name: str
    weight: float
    task_ids: tuple[str, ...]
    max_steps: int
    description: str

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the stage configuration."""
        return {
            "name": self.name,
            "weight": float(self.weight),
            "task_ids": list(self.task_ids),
            "max_steps": int(self.max_steps),
            "description": self.description,
        }


@dataclass(slots=True)
class DatasetGenerationConfig:
    """Configuration for synthetic dataset generation."""

    output_root: Path
    scene_selection: SceneSelectionConfig
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    stages: tuple[CurriculumStageConfig, ...] = field(default_factory=tuple)
    episodes_per_scene: int = 12
    shard_size_episodes: int = 8
    image_size: tuple[int, int] = (224, 224)
    renderer_device: str = "cuda"
    use_depth: bool = True
    action_mode: str = "waypoint"
    seed: int = 0
    debug_export_episodes_per_split: int = 2
    allow_mock_rendering: bool = False
    dataset_id: str = "synthetic_vla_an"

    def __post_init__(self) -> None:
        """Populate default stages after initialization."""
        self.output_root = Path(self.output_root)
        if not self.stages:
            self.stages = default_curriculum_stages()

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the generation configuration."""
        return {
            "output_root": str(self.output_root),
            "scene_selection": self.scene_selection.to_dict(),
            "planner": self.planner.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "episodes_per_scene": int(self.episodes_per_scene),
            "shard_size_episodes": int(self.shard_size_episodes),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "renderer_device": self.renderer_device,
            "use_depth": bool(self.use_depth),
            "action_mode": self.action_mode,
            "seed": int(self.seed),
            "debug_export_episodes_per_split": int(self.debug_export_episodes_per_split),
            "allow_mock_rendering": bool(self.allow_mock_rendering),
            "dataset_id": self.dataset_id,
        }


@dataclass(slots=True)
class ShardManifest:
    """One Parquet shard entry in the dataset manifest."""

    shard_id: str
    split: str
    parquet_path: str
    episode_ids: list[str]
    row_count: int
    scene_ids: list[str]
    stage_names: list[str]
    status: str = "completed"

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the shard manifest."""
        return {
            "shard_id": self.shard_id,
            "split": self.split,
            "parquet_path": self.parquet_path,
            "episode_ids": list(self.episode_ids),
            "row_count": int(self.row_count),
            "scene_ids": list(self.scene_ids),
            "stage_names": list(self.stage_names),
            "status": self.status,
        }


@dataclass(slots=True)
class DatasetManifest:
    """Top-level manifest for a generated synthetic dataset."""

    dataset_id: str
    format_version: str
    created_at_utc: str
    config: dict[str, JsonValue]
    scene_splits: dict[str, str]
    shard_manifests: list[ShardManifest]
    counts: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the dataset manifest."""
        return {
            "dataset_id": self.dataset_id,
            "format_version": self.format_version,
            "created_at_utc": self.created_at_utc,
            "config": self.config,
            "scene_splits": self.scene_splits,
            "shard_manifests": [item.to_dict() for item in self.shard_manifests],
            "counts": self.counts,
        }

    def to_json(self, path: str | Path) -> None:
        """Write the dataset manifest to disk."""
        _write_json(Path(path), cast(dict[str, object], self.to_dict()))

    @classmethod
    def from_json(cls, path: str | Path) -> DatasetManifest:
        """Read one dataset manifest from disk."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            dataset_id=str(payload["dataset_id"]),
            format_version=str(payload["format_version"]),
            created_at_utc=str(payload["created_at_utc"]),
            config=cast(dict[str, JsonValue], payload["config"]),
            scene_splits={str(key): str(value) for key, value in payload["scene_splits"].items()},
            shard_manifests=[
                ShardManifest(
                    shard_id=str(item["shard_id"]),
                    split=str(item["split"]),
                    parquet_path=str(item["parquet_path"]),
                    episode_ids=[str(value) for value in item["episode_ids"]],
                    row_count=int(item["row_count"]),
                    scene_ids=[str(value) for value in item["scene_ids"]],
                    stage_names=[str(value) for value in item["stage_names"]],
                    status=str(item["status"]),
                )
                for item in cast(list[dict[str, object]], payload["shard_manifests"])
            ],
            counts=cast(dict[str, JsonValue], payload["counts"]),
        )


@dataclass(slots=True)
class ValidationReport:
    """Dataset validation summary."""

    dataset_id: str
    n_episodes: int
    n_steps: int
    split_counts: dict[str, int]
    stage_counts: dict[str, int]
    task_counts: dict[str, int]
    scene_counts: dict[str, int]
    instruction_coverage: dict[str, int]
    success_rate: float
    collision_rate: float
    waypoint_magnitude_mean: float
    valid: bool

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the validation report."""
        return {
            "dataset_id": self.dataset_id,
            "n_episodes": int(self.n_episodes),
            "n_steps": int(self.n_steps),
            "split_counts": {key: int(value) for key, value in self.split_counts.items()},
            "stage_counts": {key: int(value) for key, value in self.stage_counts.items()},
            "task_counts": {key: int(value) for key, value in self.task_counts.items()},
            "scene_counts": {key: int(value) for key, value in self.scene_counts.items()},
            "instruction_coverage": {
                key: int(value) for key, value in self.instruction_coverage.items()
            },
            "success_rate": float(self.success_rate),
            "collision_rate": float(self.collision_rate),
            "waypoint_magnitude_mean": float(self.waypoint_magnitude_mean),
            "valid": bool(self.valid),
        }


@dataclass(slots=True)
class PreviewSummary:
    """Summary returned by preview generation."""

    task_id: str
    stage_name: str
    scene_id: str
    instruction: str
    n_steps: int
    terminated: bool
    truncated: bool
    save_path: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the preview summary."""
        return cast(dict[str, JsonValue], json.loads(json.dumps(asdict(self))))


@dataclass(slots=True)
class _SceneSpec:
    """Resolved scene metadata used internally during generation."""

    scene_id: str
    scene_ref: str | Path | None
    bbox: np.ndarray
    split: str


def default_curriculum_stages() -> tuple[CurriculumStageConfig, ...]:
    """Return the default three-stage VLA-AN-like curriculum."""
    return (
        CurriculumStageConfig(
            name="stage1_scene_comprehension",
            weight=0.2,
            task_ids=("object_nav", "point_nav"),
            max_steps=80,
            description="Short scene grounding and pointing hops.",
        ),
        CurriculumStageConfig(
            name="stage2_flight_skills",
            weight=0.3,
            task_ids=("point_nav", "obstacle_slalom", "narrow_corridor"),
            max_steps=120,
            description="Core flight skills and obstacle-aware short-horizon clips.",
        ),
        CurriculumStageConfig(
            name="stage3_long_horizon_navigation",
            weight=0.5,
            task_ids=(
                "point_nav",
                "object_nav",
                "dynamic_follow",
                "obstacle_slalom",
                "narrow_corridor",
            ),
            max_steps=200,
            description="Long-horizon navigation and dynamic target tracking.",
        ),
    )


class _ShardWriter:
    """Incremental Parquet/media writer for one generated dataset."""

    def __init__(self, root: Path, shard_size_episodes: int) -> None:
        """Initialize shard buffers."""
        _require_pyarrow()
        self.root = root
        self.shard_size_episodes = int(shard_size_episodes)
        self.step_buffers: dict[str, list[dict[str, object]]] = {}
        self.shard_episode_ids: dict[str, list[str]] = {}
        self.shard_scene_ids: dict[str, list[str]] = {}
        self.shard_stage_names: dict[str, list[str]] = {}
        self.shard_indices: dict[str, int] = {}
        self.episodes_per_split: dict[str, list[dict[str, object]]] = {}
        self.shards: list[ShardManifest] = []
        self.debug_counts: dict[str, int] = {}

    def current_shard_name(self, split: str) -> str:
        """Return the current shard name for one split."""
        index = self.shard_indices.get(split, 0)
        return f"{split}-{index:05d}"

    def media_directory(self, split: str) -> Path:
        """Return the current media directory for one split."""
        shard_name = self.current_shard_name(split)
        return self.root / "media" / split / shard_name

    def add_episode(
        self,
        split: str,
        episode: TrajectoryEpisode,
        scene_id: str,
        stage_name: str,
        step_rows: list[dict[str, object]],
        episode_row: dict[str, object],
    ) -> None:
        """Add one episode and flush shards when needed."""
        self.step_buffers.setdefault(split, []).extend(step_rows)
        self.shard_episode_ids.setdefault(split, []).append(episode.episode_id)
        self.shard_scene_ids.setdefault(split, []).append(scene_id)
        self.shard_stage_names.setdefault(split, []).append(stage_name)
        self.episodes_per_split.setdefault(split, []).append(episode_row)
        if len(self.shard_episode_ids[split]) >= self.shard_size_episodes:
            self.flush(split)

    def flush(self, split: str) -> None:
        """Write one split buffer into a Parquet shard."""
        rows = self.step_buffers.get(split, [])
        if not rows:
            return
        shard_name = self.current_shard_name(split)
        parquet_dir = self.root / "parquet" / split
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / f"steps-{shard_name}.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, parquet_path)
        manifest = ShardManifest(
            shard_id=shard_name,
            split=split,
            parquet_path=str(parquet_path.relative_to(self.root)),
            episode_ids=list(self.shard_episode_ids.get(split, [])),
            row_count=len(rows),
            scene_ids=sorted(set(self.shard_scene_ids.get(split, []))),
            stage_names=sorted(set(self.shard_stage_names.get(split, []))),
        )
        self.shards.append(manifest)
        self.step_buffers[split] = []
        self.shard_episode_ids[split] = []
        self.shard_scene_ids[split] = []
        self.shard_stage_names[split] = []
        self.shard_indices[split] = self.shard_indices.get(split, 0) + 1

    def finalize(self) -> list[ShardManifest]:
        """Flush all split buffers and write episode metadata tables."""
        for split in list(self.step_buffers.keys()):
            self.flush(split)
        for split, rows in self.episodes_per_split.items():
            if not rows:
                continue
            parquet_dir = self.root / "parquet" / split
            parquet_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), parquet_dir / "episodes.parquet")
        return list(self.shards)


def _resolve_scenes(config: DatasetGenerationConfig) -> list[_SceneSpec]:
    """Resolve scene handles into local generation specs."""
    loader = SceneLoader()
    rng = np.random.default_rng(config.seed)
    sources = list(config.scene_selection.sources)
    if not sources:
        raise ValueError("At least one scene source is required for dataset generation.")
    shuffled = list(sources)
    rng.shuffle(shuffled)
    split_map = _assign_scene_splits(shuffled, config.scene_selection)

    scenes: list[_SceneSpec] = []
    for source in sources:
        if source.startswith("mock://"):
            if not config.allow_mock_rendering:
                raise ValueError(
                    "Mock scenes are only supported when allow_mock_rendering=True."
                )
            scene_id = source.removeprefix("mock://") or "mock_scene"
            bbox = np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 5.0]], dtype=np.float32)
            scenes.append(
                _SceneSpec(
                    scene_id=scene_id,
                    scene_ref=None,
                    bbox=bbox,
                    split=split_map[source],
                )
            )
            continue

        if source in BUILTIN_SCENES:
            scene_info = BUILTIN_SCENES[source]
            try:
                from gs_dronegym.scene import get_scene

                resolved = get_scene(source)
            except Exception as exc:  # pragma: no cover - network path
                raise RuntimeError(f"Failed to resolve built-in scene '{source}': {exc}") from exc
            scenes.append(
                _SceneSpec(
                    scene_id=source,
                    scene_ref=resolved,
                    bbox=scene_info.bbox.astype(np.float32),
                    split=split_map[source],
                )
            )
            continue

        resolved_path = loader.load(source)
        scenes.append(
            _SceneSpec(
                scene_id=resolved_path.stem,
                scene_ref=resolved_path,
                bbox=loader.infer_bbox(resolved_path),
                split=split_map[source],
            )
        )
    return scenes


def _assign_scene_splits(
    shuffled_sources: list[str],
    config: SceneSelectionConfig,
) -> dict[str, str]:
    """Assign deterministic scene-level train/val/test splits."""
    if len(shuffled_sources) == 1:
        return {shuffled_sources[0]: "train"}
    if len(shuffled_sources) == 2:
        return {
            shuffled_sources[0]: "train",
            shuffled_sources[1]: "val",
        }
    n_scenes = len(shuffled_sources)
    n_train = max(1, int(round(n_scenes * config.train_fraction)))
    n_val = max(1, int(round(n_scenes * config.val_fraction)))
    remaining = n_scenes - n_train - n_val
    n_test = max(1, remaining)
    while n_train + n_val + n_test > n_scenes:
        n_train = max(1, n_train - 1)
    if n_train + n_val + n_test < n_scenes:
        n_train += n_scenes - (n_train + n_val + n_test)
    split_map: dict[str, str] = {}
    for index, source in enumerate(shuffled_sources):
        if index < n_train:
            split_map[source] = "train"
        elif index < n_train + n_val:
            split_map[source] = "val"
        else:
            split_map[source] = "test"
    return split_map


def _episodes_per_stage(
    episodes_per_scene: int,
    stages: tuple[CurriculumStageConfig, ...],
) -> dict[str, int]:
    """Allocate per-scene episode counts to curriculum stages."""
    weights = np.asarray([stage.weight for stage in stages], dtype=np.float32)
    weights /= np.sum(weights)
    raw = weights * np.float32(episodes_per_scene)
    counts = np.floor(raw).astype(int)
    while int(np.sum(counts)) < episodes_per_scene:
        counts[int(np.argmax(raw - counts))] += 1
    return {stage.name: int(count) for stage, count in zip(stages, counts, strict=True)}


def _synthetic_regions(scene_bbox: np.ndarray) -> dict[str, np.ndarray]:
    """Construct simple language-grounded regions from one scene bounding box."""
    low = scene_bbox[0]
    high = scene_bbox[1]
    center = (low + high) / 2.0
    half = (high - low) / 4.0
    return {
        "north bay": np.stack(
            [
                np.array([center[0] - half[0], center[1], low[2] + 0.6], dtype=np.float32),
                np.array([center[0] + half[0], high[1] - 0.6, center[2] + 0.4], dtype=np.float32),
            ]
        ).astype(np.float32),
        "south bay": np.stack(
            [
                np.array([center[0] - half[0], low[1] + 0.6, low[2] + 0.6], dtype=np.float32),
                np.array([center[0] + half[0], center[1], center[2] + 0.4], dtype=np.float32),
            ]
        ).astype(np.float32),
        "center zone": np.stack(
            [
                np.array(
                    [center[0] - half[0] / 2.0, center[1] - half[1] / 2.0, low[2] + 0.6],
                    dtype=np.float32,
                ),
                np.array(
                    [center[0] + half[0] / 2.0, center[1] + half[1] / 2.0, center[2] + 0.6],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32),
    }


def _make_task(
    stage: CurriculumStageConfig,
    task_id: str,
    scene_bbox: np.ndarray,
) -> BaseTask:
    """Construct one task instance for a dataset episode."""
    task_config = TaskConfig(max_steps=stage.max_steps)
    if task_id == "point_nav":
        return PointNavTask(config=task_config)
    if task_id == "object_nav":
        return ObjectNavTask(regions=_synthetic_regions(scene_bbox), config=task_config)
    if task_id == "obstacle_slalom":
        return ObstacleSlalomTask(config=task_config)
    if task_id == "dynamic_follow":
        return DynamicFollowTask(config=task_config)
    if task_id == "narrow_corridor":
        return NarrowCorridorTask(config=task_config)
    raise ValueError(f"Unsupported dataset task id: {task_id}")


def _save_episode_debug(path: Path, episode: TrajectoryEpisode) -> None:
    """Write one debug episode in JSON form."""
    _write_json(path, cast(dict[str, object], episode.to_dict()))


def _write_preview_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 120) -> None:
    """Write one preview GIF from RGB frames."""
    if not frames:
        raise ValueError("Cannot write a preview GIF with no frames.")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def generate_dataset(config: DatasetGenerationConfig) -> DatasetManifest:
    """Generate a synthetic VLA-AN-like dataset on disk."""
    _require_pyarrow()
    planner = ExpertPlanner(config.planner)
    scenes = _resolve_scenes(config)
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(output_root, shard_size_episodes=config.shard_size_episodes)
    episode_counter = 0
    counts_per_stage: dict[str, int] = {}
    counts_per_split: dict[str, int] = {}
    counts_per_task: dict[str, int] = {}

    for scene_index, scene in enumerate(scenes):
        stage_episode_counts = _episodes_per_stage(config.episodes_per_scene, config.stages)
        for stage in config.stages:
            counts = stage_episode_counts[stage.name]
            for local_episode_idx in range(counts):
                task_rng = np.random.default_rng(
                    config.seed + scene_index * 1000 + local_episode_idx + episode_counter
                )
                task_id = str(task_rng.choice(np.asarray(stage.task_ids, dtype=object)))
                task = _make_task(stage, task_id, scene.bbox)
                from gs_dronegym import make

                env = make(
                    _env_id_for_task(task_id),
                    scene=scene.scene_ref,
                    task=task,
                    renderer_device=config.renderer_device,
                    image_size=config.image_size,
                    use_depth=config.use_depth,
                    action_mode=config.action_mode,
                )
                base_env = env.unwrapped if hasattr(env, "unwrapped") else env
                seed_value = config.seed + episode_counter
                observation, info = env.reset(seed=seed_value)
                current_observation = _copy_observation(cast(dict[str, object], observation))
                terminated = False
                truncated = False
                steps: list[TrajectoryStep] = []
                step_rows: list[dict[str, object]] = []

                while not terminated and not truncated:
                    drone_state = np.asarray(current_observation["state"], dtype=np.float32)
                    expert_waypoint, labels = planner.plan_waypoint(
                        state=drone_state,
                        goal_position=base_env.goal_position,
                        task=base_env.task,
                        scene_bbox=base_env.scene_bbox,
                        obs_dt=base_env.dynamics.config.obs_dt,
                    )
                    action = planner.normalized_waypoint_action(drone_state, expert_waypoint)
                    next_observation, reward, terminated, truncated, step_info = env.step(action)
                    step_index = len(steps)
                    step_info_dict = cast(dict[str, object], dict(step_info))
                    step_info_dict.update(
                        {
                            "expert_waypoint": expert_waypoint.astype(np.float32),
                            "collision_imminent": bool(labels["collision_imminent"]),
                            "min_clearance_m": float(labels["min_clearance_m"]),
                            "recovery_required": bool(labels["recovery_required"]),
                            "path_progress_m": float(labels["path_progress_m"]),
                            "goal_distance_m": float(labels["goal_distance_m"]),
                            "future_target_position": np.asarray(
                                labels["future_target_position"],
                                dtype=np.float32,
                            ),
                            "planner_path_xyz": np.asarray(
                                labels["planner_path_xyz"],
                                dtype=np.float32,
                            ),
                            "stage_name": stage.name,
                            "scene_id": scene.scene_id,
                            "task_id": task_id,
                        }
                    )
                    step = TrajectoryStep(
                        observation=_copy_observation(current_observation),
                        action=action.astype(np.float32),
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=step_info_dict,
                        step_index=step_index,
                        timestamp_s=step_index * float(base_env.dynamics.config.obs_dt),
                        benchmark_metrics={
                            "distance_to_goal": float(step_info_dict.get("distance_to_goal", 0.0)),
                            "min_clearance_m": float(labels["min_clearance_m"]),
                        },
                    )
                    steps.append(step)
                    current_media_dir = writer.media_directory(scene.split)
                    rgb_rel_path = _save_rgb(
                        current_media_dir / "rgb",
                        scene.scene_id,
                        episode_counter,
                        step_index,
                        np.asarray(current_observation["rgb"], dtype=np.uint8),
                        output_root,
                    )
                    depth_rel_path = None
                    if config.use_depth and "depth" in current_observation:
                        depth_rel_path = _save_depth(
                            current_media_dir / "depth",
                            scene.scene_id,
                            episode_counter,
                            step_index,
                            np.asarray(current_observation["depth"], dtype=np.float32),
                            output_root,
                        )
                    step_rows.append(
                        _step_to_row(
                            dataset_id=config.dataset_id,
                            scene_id=scene.scene_id,
                            split=scene.split,
                            stage_name=stage.name,
                            task_id=task_id,
                            episode_id=f"{config.dataset_id}-{episode_counter:06d}",
                            step=step,
                            rgb_rel_path=rgb_rel_path,
                            depth_rel_path=depth_rel_path,
                            seed_value=seed_value,
                        )
                    )
                    current_observation = _copy_observation(
                        cast(dict[str, object], next_observation)
                    )

                action_spec = ActionSpec(
                    shape=(4,),
                    normalized=True,
                    semantics="normalized waypoint delta action",
                    name="waypoint_action",
                    metadata={"label_semantics": "expert_waypoint_global_xyz_yaw"},
                )
                task_spec = TaskSpec(
                    task_id=task_id,
                    benchmark_name="gs_dronegym",
                    embodiment="drone",
                    instruction=str(current_observation.get("instruction", "")),
                    description=f"Synthetic {stage.name} rollout",
                    metadata={"stage_name": stage.name, "scene_id": scene.scene_id},
                )
                episode = TrajectoryEpisode(
                    episode_id=f"{config.dataset_id}-{episode_counter:06d}",
                    benchmark_name="gs_dronegym",
                    embodiment="drone",
                    task=task_spec,
                    action_spec=action_spec,
                    observation_spec=infer_observation_spec(current_observation),
                    steps=steps,
                    success=bool(steps[-1].info.get("success", False)) if steps else False,
                    split=scene.split,
                    source=scene.scene_id,
                    metadata={
                        "scene_id": scene.scene_id,
                        "scene_path": "" if scene.scene_ref is None else str(scene.scene_ref),
                        "stage_name": stage.name,
                        "seed": seed_value,
                        "goal_position": np.asarray(base_env.goal_position, dtype=np.float32),
                    },
                    benchmark_metrics={
                        "initial_distance_to_goal": float(info.get("distance_to_goal", 0.0))
                    },
                )
                writer.add_episode(
                    split=scene.split,
                    episode=episode,
                    scene_id=scene.scene_id,
                    stage_name=stage.name,
                    step_rows=step_rows,
                    episode_row=_episode_to_row(episode),
                )
                if writer.debug_counts.get(scene.split, 0) < config.debug_export_episodes_per_split:
                    debug_path = (
                        output_root / "episodes_debug" / scene.split / f"{episode.episode_id}.json"
                    )
                    _save_episode_debug(debug_path, episode)
                    writer.debug_counts[scene.split] = writer.debug_counts.get(scene.split, 0) + 1
                env.close()
                episode_counter += 1
                counts_per_stage[stage.name] = counts_per_stage.get(stage.name, 0) + 1
                counts_per_split[scene.split] = counts_per_split.get(scene.split, 0) + 1
                counts_per_task[task_id] = counts_per_task.get(task_id, 0) + 1

    shard_manifests = writer.finalize()
    _write_json(
        output_root / "splits.json",
        {"scene_splits": {scene.scene_id: scene.split for scene in scenes}},
    )
    manifest = DatasetManifest(
        dataset_id=config.dataset_id,
        format_version=FORMAT_VERSION,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config.to_dict(),
        scene_splits={scene.scene_id: scene.split for scene in scenes},
        shard_manifests=shard_manifests,
        counts={
            "n_episodes": episode_counter,
            "n_shards": len(shard_manifests),
            "stage_counts": counts_per_stage,
            "split_counts": counts_per_split,
            "task_counts": counts_per_task,
        },
    )
    manifest.to_json(output_root / "manifest.json")
    return manifest


def preview_dataset_task(
    scene: str | Path | None,
    stage_name: str = "stage1_scene_comprehension",
    task_id: str | None = None,
    steps: int = 60,
    save_gif: str | Path | None = None,
    renderer_device: str = "cpu",
    use_depth: bool = True,
    seed: int = 0,
    allow_mock_rendering: bool = False,
) -> PreviewSummary:
    """Preview one synthetic dataset episode without writing a dataset root."""
    config = DatasetGenerationConfig(
        output_root=Path("."),
        scene_selection=SceneSelectionConfig(
            sources=("mock://preview",) if scene is None else (str(scene),)
        ),
        renderer_device=renderer_device,
        use_depth=use_depth,
        allow_mock_rendering=allow_mock_rendering or scene is None,
        episodes_per_scene=1,
        seed=seed,
    )
    planner = ExpertPlanner(config.planner)
    scenes = _resolve_scenes(config)
    scene_spec = scenes[0]
    stage = next(item for item in config.stages if item.name == stage_name)
    chosen_task_id = task_id or stage.task_ids[0]
    task = _make_task(stage, chosen_task_id, scene_spec.bbox)
    from gs_dronegym import make

    env = make(
        _env_id_for_task(chosen_task_id),
        scene=scene_spec.scene_ref,
        task=task,
        renderer_device=renderer_device,
        use_depth=use_depth,
    )
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    observation, _ = env.reset(seed=seed)
    current_observation = _copy_observation(cast(dict[str, object], observation))
    terminated = False
    truncated = False
    frames: list[np.ndarray] = []
    step_count = 0
    while not terminated and not truncated and step_count < steps:
        state = np.asarray(current_observation["state"], dtype=np.float32)
        waypoint, _ = planner.plan_waypoint(
            state=state,
            goal_position=base_env.goal_position,
            task=base_env.task,
            scene_bbox=base_env.scene_bbox,
            obs_dt=base_env.dynamics.config.obs_dt,
        )
        action = planner.normalized_waypoint_action(state, waypoint)
        next_observation, _, terminated, truncated, _ = env.step(action)
        frames.append(np.asarray(current_observation["rgb"], dtype=np.uint8))
        current_observation = _copy_observation(cast(dict[str, object], next_observation))
        step_count += 1
    env.close()
    if save_gif is not None:
        _write_preview_gif(frames, Path(save_gif))
    return PreviewSummary(
        task_id=chosen_task_id,
        stage_name=stage.name,
        scene_id=scene_spec.scene_id,
        instruction=str(current_observation.get("instruction", "")),
        n_steps=step_count,
        terminated=bool(terminated),
        truncated=bool(truncated),
        save_path=None if save_gif is None else str(Path(save_gif)),
    )


def validate_generated_dataset(path: str | Path) -> ValidationReport:
    """Validate a generated synthetic dataset and summarize its contents."""
    dataset_root = Path(path)
    manifest = DatasetManifest.from_json(dataset_root / "manifest.json")
    episodes = load_generated_dataset(dataset_root)
    task_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    scene_counts: dict[str, int] = {}
    instruction_coverage: dict[str, int] = {}
    collisions = 0
    successes = 0
    waypoint_magnitudes: list[float] = []
    n_steps = 0

    for episode in episodes:
        split_counts[episode.split] = split_counts.get(episode.split, 0) + 1
        task_counts[episode.task.task_id] = task_counts.get(episode.task.task_id, 0) + 1
        stage_name = str(episode.metadata.get("stage_name", "unknown"))
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        scene_id = str(episode.metadata.get("scene_id", "unknown"))
        scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        instruction_coverage[episode.task.instruction] = (
            instruction_coverage.get(episode.task.instruction, 0) + 1
        )
        successes += int(episode.success)
        for step in episode.steps:
            rgb_path = dataset_root / cast(str, step.info["rgb_path"])
            if not rgb_path.exists():
                raise FileNotFoundError(f"Missing RGB file referenced by dataset: {rgb_path}")
            depth_path_raw = cast(str | None, step.info.get("depth_path"))
            if depth_path_raw:
                depth_path = dataset_root / depth_path_raw
                if not depth_path.exists():
                    raise FileNotFoundError(
                        f"Missing depth file referenced by dataset: {depth_path}"
                    )
            collisions += int(bool(step.info.get("collision_occurred", False)))
            waypoint = np.asarray(step.info.get("expert_waypoint", np.zeros(4)), dtype=np.float32)
            waypoint_magnitudes.append(float(np.linalg.norm(waypoint[:3])))
            n_steps += 1

    return ValidationReport(
        dataset_id=manifest.dataset_id,
        n_episodes=len(episodes),
        n_steps=n_steps,
        split_counts=split_counts,
        stage_counts=stage_counts,
        task_counts=task_counts,
        scene_counts=scene_counts,
        instruction_coverage=instruction_coverage,
        success_rate=0.0 if not episodes else successes / len(episodes),
        collision_rate=0.0 if n_steps == 0 else collisions / n_steps,
        waypoint_magnitude_mean=0.0
        if not waypoint_magnitudes
        else float(np.mean(np.asarray(waypoint_magnitudes, dtype=np.float32))),
        valid=True,
    )


def load_generated_dataset(
    source: str | Path,
    split: str | None = None,
    max_episodes: int | None = None,
) -> list[TrajectoryEpisode]:
    """Load generated Parquet shards back into common trajectory episodes."""
    _require_pyarrow()
    dataset_root = Path(source)
    manifest = DatasetManifest.from_json(dataset_root / "manifest.json")
    split_names = (
        [split]
        if split is not None
        else sorted({item.split for item in manifest.shard_manifests})
    )
    episodes: list[TrajectoryEpisode] = []
    steps_by_episode: dict[str, list[dict[str, object]]] = {}
    for split_name in split_names:
        parquet_dir = dataset_root / "parquet" / split_name
        if not parquet_dir.exists():
            continue
        for parquet_path in sorted(parquet_dir.glob("steps-*.parquet")):
            rows = cast(list[dict[str, object]], pq.read_table(parquet_path).to_pylist())
            for row in rows:
                episode_id = str(row["episode_id"])
                steps_by_episode.setdefault(episode_id, []).append(row)
        episode_meta_path = parquet_dir / "episodes.parquet"
        if not episode_meta_path.exists():
            continue
        episode_rows = cast(list[dict[str, object]], pq.read_table(episode_meta_path).to_pylist())
        for row in episode_rows:
            episode_id = str(row["episode_id"])
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
            step_rows = sorted(
                steps_by_episode.get(episode_id, []),
                key=lambda item: int(item["step_index"]),
            )
            steps = [_row_to_step(dataset_root, item) for item in step_rows]
            episode = TrajectoryEpisode(
                episode_id=episode_id,
                benchmark_name=str(row["benchmark_name"]),
                embodiment=str(row["embodiment"]),
                task=TaskSpec.from_dict(
                    cast(dict[str, JsonValue], json.loads(str(row["task_json"])))
                ),
                action_spec=ActionSpec.from_dict(
                    cast(dict[str, JsonValue], json.loads(str(row["action_spec_json"])))
                ),
                observation_spec=ObservationSpec.from_dict(
                    cast(dict[str, JsonValue], json.loads(str(row["observation_spec_json"])))
                ),
                steps=steps,
                success=bool(row["success"]),
                split=str(row["split"]),
                source=str(row["source"]),
                metadata=cast(dict[str, JsonValue], json.loads(str(row["metadata_json"]))),
                benchmark_metrics=cast(
                    dict[str, float], json.loads(str(row["benchmark_metrics_json"]))
                ),
            )
            episodes.append(episode)
    return episodes


def _save_rgb(
    directory: Path,
    scene_id: str,
    episode_index: int,
    step_index: int,
    rgb: np.ndarray,
    dataset_root: Path,
) -> str:
    """Save one RGB frame and return its relative path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scene_id}-{episode_index:06d}-{step_index:04d}.png"
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return str(path.relative_to(dataset_root))


def _save_depth(
    directory: Path,
    scene_id: str,
    episode_index: int,
    step_index: int,
    depth: np.ndarray,
    dataset_root: Path,
) -> str:
    """Save one depth map and return its relative path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scene_id}-{episode_index:06d}-{step_index:04d}.npy"
    np.save(path, depth.astype(np.float32))
    return str(path.relative_to(dataset_root))


def _copy_observation(observation: dict[str, object]) -> dict[str, object]:
    """Copy an observation dict with arrays duplicated."""
    copied: dict[str, object] = {}
    for key, value in observation.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _step_to_row(
    dataset_id: str,
    scene_id: str,
    split: str,
    stage_name: str,
    task_id: str,
    episode_id: str,
    step: TrajectoryStep,
    rgb_rel_path: str,
    depth_rel_path: str | None,
    seed_value: int,
) -> dict[str, object]:
    """Convert one trajectory step into a Parquet row."""
    state = np.asarray(step.observation.get("state", np.zeros(0)), dtype=np.float32)
    expert_waypoint = np.asarray(step.info["expert_waypoint"], dtype=np.float32)
    future_target = np.asarray(step.info["future_target_position"], dtype=np.float32)
    return {
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "step_index": int(step.step_index),
        "stage": stage_name,
        "split": split,
        "task_id": task_id,
        "instruction": str(step.observation.get("instruction", "")),
        "state": state.astype(np.float32).tolist(),
        "action": np.asarray(step.action, dtype=np.float32).tolist(),
        "expert_waypoint": expert_waypoint.astype(np.float32).tolist(),
        "future_target_position": future_target.astype(np.float32).tolist(),
        "collision_imminent": bool(step.info["collision_imminent"]),
        "min_clearance_m": float(step.info["min_clearance_m"]),
        "recovery_required": bool(step.info["recovery_required"]),
        "collision_occurred": bool(step.info.get("collision", False)),
        "success": bool(step.info.get("success", False)),
        "distance_to_goal": float(step.info.get("distance_to_goal", 0.0)),
        "reward": float(step.reward),
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
        "timestamp_s": 0.0 if step.timestamp_s is None else float(step.timestamp_s),
        "rgb_path": rgb_rel_path,
        "depth_path": depth_rel_path,
        "seed": int(seed_value),
    }


def _episode_to_row(episode: TrajectoryEpisode) -> dict[str, object]:
    """Convert one episode into an episode metadata row."""
    return {
        "episode_id": episode.episode_id,
        "benchmark_name": episode.benchmark_name,
        "embodiment": episode.embodiment,
        "task_json": json.dumps(episode.task.to_dict()),
        "action_spec_json": json.dumps(episode.action_spec.to_dict()),
        "observation_spec_json": json.dumps(episode.observation_spec.to_dict()),
        "success": bool(episode.success),
        "split": episode.split,
        "source": episode.source,
        "metadata_json": json.dumps(episode.metadata, default=_json_default),
        "benchmark_metrics_json": json.dumps(episode.benchmark_metrics),
    }


def _row_to_step(dataset_root: Path, row: dict[str, object]) -> TrajectoryStep:
    """Reconstruct one trajectory step from a Parquet row."""
    rgb = np.asarray(Image.open(dataset_root / str(row["rgb_path"])).convert("RGB"), dtype=np.uint8)
    observation: dict[str, object] = {
        "rgb": rgb,
        "state": np.asarray(row["state"], dtype=np.float32),
        "instruction": str(row["instruction"]),
    }
    depth_path_raw = cast(str | None, row.get("depth_path"))
    if depth_path_raw:
        observation["depth"] = np.load(dataset_root / depth_path_raw).astype(np.float32)
    info = {
        "collision": bool(row["collision_occurred"]),
        "collision_occurred": bool(row["collision_occurred"]),
        "success": bool(row["success"]),
        "distance_to_goal": float(row["distance_to_goal"]),
        "expert_waypoint": np.asarray(row["expert_waypoint"], dtype=np.float32),
        "collision_imminent": bool(row["collision_imminent"]),
        "min_clearance_m": float(row["min_clearance_m"]),
        "recovery_required": bool(row["recovery_required"]),
        "future_target_position": np.asarray(row["future_target_position"], dtype=np.float32),
        "rgb_path": str(row["rgb_path"]),
        "depth_path": depth_path_raw,
    }
    return TrajectoryStep(
        observation=observation,
        action=np.asarray(row["action"], dtype=np.float32),
        reward=float(row["reward"]),
        terminated=bool(row["terminated"]),
        truncated=bool(row["truncated"]),
        info=info,
        step_index=int(row["step_index"]),
        timestamp_s=float(row["timestamp_s"]),
        benchmark_metrics={
            "distance_to_goal": float(row["distance_to_goal"]),
            "min_clearance_m": float(row["min_clearance_m"]),
        },
    )


def _env_id_for_task(task_id: str) -> str:
    """Map a task id to a registered environment id."""
    mapping = {
        "point_nav": "PointNav-v0",
        "object_nav": "ObjectNav-v0",
        "obstacle_slalom": "ObstacleSlalom-v0",
        "dynamic_follow": "DynamicFollow-v0",
        "narrow_corridor": "NarrowCorridor-v0",
    }
    return mapping[task_id]
