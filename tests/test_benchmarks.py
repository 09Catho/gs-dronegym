"""Tests for benchmark adapters and dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from gs_dronegym import list_benchmarks, load_dataset, make_benchmark
from gs_dronegym.benchmarks.drone import DroneBenchmark
from gs_dronegym.cli import evaluate as evaluate_cli
from gs_dronegym.cli._scene import normalize_scene_arg


def test_list_benchmarks_exposes_supported_adapters() -> None:
    """The benchmark registry should list all public adapter names."""
    assert set(list_benchmarks()) == {"drone", "gs_dronegym", "libero", "lerobot"}


def test_make_benchmark_returns_drone_adapter() -> None:
    """Registry construction should return the live drone adapter."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    assert isinstance(benchmark, DroneBenchmark)


def test_drone_benchmark_collects_common_schema_episode() -> None:
    """A live drone rollout should export into the shared trajectory schema."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    episode = benchmark.collect_episode(seed=3)
    assert episode.benchmark_name == "gs_dronegym"
    assert episode.task.task_id == "point_nav"
    assert episode.action_spec.shape == (4,)
    assert episode.n_steps >= 1


def test_drone_benchmark_report_contains_navigation_metrics() -> None:
    """Drone benchmark reports should include navigation-specific metrics."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    report = benchmark.evaluate_policy(policy=None, n_episodes=2, seed=9)
    assert report.n_episodes == 2
    assert "spl" in report.benchmark_metrics
    assert "collision_rate" in report.benchmark_metrics


def test_load_libero_dataset_from_synthetic_hdf5(tmp_path: Path) -> None:
    """The LIBERO adapter should parse a minimal local HDF5 demo file."""
    h5py = pytest.importorskip("h5py")
    dataset_path = tmp_path / "demo.hdf5"
    with h5py.File(dataset_path, "w") as handle:
        handle.attrs["task_name"] = "pick_cube"
        handle.attrs["language_instruction"] = "pick the cube"
        data_group = handle.create_group("data")
        demo = data_group.create_group("demo_0")
        demo.create_dataset("actions", data=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
        demo.create_dataset("rewards", data=np.asarray([0.0, 1.0], dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset(
            "robot_state",
            data=np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        )
        obs.create_dataset(
            "agentview_rgb",
            data=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        )
    episodes = load_dataset(dataset_path, format="libero")
    assert len(episodes) == 1
    assert episodes[0].task.instruction == "pick the cube"
    assert episodes[0].steps[0].action.shape == (2,)


def test_load_lerobot_dataset_from_synthetic_parquet(tmp_path: Path) -> None:
    """The LeRobot adapter should parse a minimal local parquet dataset."""
    pyarrow = pytest.importorskip("pyarrow")
    from pyarrow import parquet as pq

    dataset_root = tmp_path / "lerobot_sample"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps({"robot_type": "widowx"}),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick block"}) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "task_index": 0, "split": "train"}) + "\n",
        encoding="utf-8",
    )
    table = pyarrow.table(
        {
            "observation.state.x": [0.0, 1.0],
            "observation.state.y": [0.1, 1.1],
            "action.0": [0.5, 0.6],
            "action.1": [0.2, 0.3],
            "reward": [0.0, 1.0],
            "done": [False, True],
            "task": ["pick block", "pick block"],
        }
    )
    parquet_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    pq.write_table(table, parquet_path)

    episodes = load_dataset(dataset_root, format="lerobot")
    assert len(episodes) == 1
    assert episodes[0].task.instruction == "pick block"
    assert episodes[0].steps[-1].terminated is True


def test_report_omits_raw_results_by_default() -> None:
    """Reports must not embed per-step records unless explicitly requested."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    report = benchmark.evaluate_policy(policy=None, n_episodes=1, seed=0)
    payload = report.to_dict()

    assert payload["raw_results"] == []
    summaries = payload["episode_summaries"]
    assert isinstance(summaries, list) and len(summaries) == 1
    assert set(summaries[0]) >= {"episode_id", "task_id", "success", "n_steps", "total_reward"}
    assert len(json.dumps(payload)) < 100_000


def test_raw_results_reference_media_instead_of_inlining_it() -> None:
    """Opt-in raw results must replace image buffers with shape references."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    report = benchmark.evaluate_policy(
        policy=None,
        n_episodes=1,
        seed=0,
        include_raw_results=True,
    )
    payload = report.to_dict()
    raw = payload["raw_results"]
    assert isinstance(raw, list) and len(raw) == 1

    serialized = json.dumps(payload)
    assert '"__kind__": "ndarray_ref"' in serialized
    # A single 224x224x3 frame expands to roughly 600 KB of JSON when inlined.
    assert len(serialized) < 2_000_000

    steps = cast(list[dict[str, object]], cast(dict[str, object], raw[0])["steps"])
    rgb = cast(dict[str, object], cast(dict[str, object], steps[0]["observation"])["rgb"])
    assert rgb["__kind__"] == "ndarray_ref"
    assert rgb["omitted"] is True
    assert rgb["shape"] == [224, 224, 3]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("None", None),
        ("none", None),
        ("NONE", None),
        ("null", None),
        ("", None),
        ("  none  ", None),
        ("garden", "garden"),
        ("C:/scenes/room.ply", "C:/scenes/room.ply"),
    ],
)
def test_scene_argument_normalization(raw: str | None, expected: str | None) -> None:
    """All CLIs must agree on which --scene values mean "no scene"."""
    assert normalize_scene_arg(raw) == expected


def test_evaluate_cli_normalizes_literal_none_scene() -> None:
    """`--scene None` must not reach the benchmark as a literal scene handle."""
    parser = evaluate_cli.build_parser()
    args = parser.parse_args(["--benchmark", "gs_dronegym", "--scene", "None"])
    assert normalize_scene_arg(args.scene) is None
    assert args.include_raw_results is False
