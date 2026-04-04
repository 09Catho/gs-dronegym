"""Benchmark adapters and registry for cross-benchmark evaluation."""

from gs_dronegym.benchmarks.base import BenchmarkAdapter, PolicyLike
from gs_dronegym.benchmarks.drone import DroneBenchmark, trajectory_to_nav_episode
from gs_dronegym.benchmarks.lerobot import LeRobotBenchmark, load_lerobot_dataset
from gs_dronegym.benchmarks.libero import LiberoBenchmark, load_libero_dataset
from gs_dronegym.benchmarks.registry import list_benchmarks, load_dataset, make_benchmark

__all__ = [
    "BenchmarkAdapter",
    "DroneBenchmark",
    "LeRobotBenchmark",
    "LiberoBenchmark",
    "PolicyLike",
    "list_benchmarks",
    "load_dataset",
    "load_lerobot_dataset",
    "load_libero_dataset",
    "make_benchmark",
    "trajectory_to_nav_episode",
]
