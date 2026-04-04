"""Tests for benchmark adapters and dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gs_dronegym import list_benchmarks, load_dataset, make_benchmark
from gs_dronegym.benchmarks.drone import DroneBenchmark


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
