"""Benchmark registry and dataset dispatch for GS-DroneGym v0.2."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from gs_dronegym.benchmarks.base import BenchmarkAdapter
from gs_dronegym.benchmarks.drone import DroneBenchmark
from gs_dronegym.benchmarks.lerobot import LeRobotBenchmark
from gs_dronegym.benchmarks.libero import LiberoBenchmark
from gs_dronegym.data.dataset import load_dataset as load_dataset_from_format
from gs_dronegym.data.schema import TrajectoryEpisode

BenchmarkName = Literal["gs_dronegym", "drone", "libero", "lerobot"]


def list_benchmarks() -> list[str]:
    """Return the supported benchmark adapter names.

    Returns:
        Sorted benchmark names.
    """
    return ["drone", "gs_dronegym", "libero", "lerobot"]


def make_benchmark(name: BenchmarkName, **kwargs: object) -> BenchmarkAdapter:
    """Instantiate one of the supported benchmark adapters.

    Args:
        name: Benchmark adapter name.
        **kwargs: Adapter constructor kwargs.

    Returns:
        Configured benchmark adapter.

    Raises:
        ValueError: If the benchmark name is unknown.
    """
    if name in {"gs_dronegym", "drone"}:
        return DroneBenchmark(**kwargs)
    if name == "libero":
        return LiberoBenchmark(**kwargs)
    if name == "lerobot":
        return LeRobotBenchmark()
    raise ValueError(f"Unknown benchmark adapter: {name}")


def load_dataset(
    source: str | Path,
    format: Literal["gs_dronegym", "libero", "lerobot"],
) -> list[TrajectoryEpisode]:
    """Dispatch dataset loading by benchmark format.

    Args:
        source: Dataset source path.
        format: Dataset format name.

    Returns:
        Parsed trajectory episodes.
    """
    return load_dataset_from_format(source, format)
