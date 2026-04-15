"""Reproduce lightweight resource and baseline measurements for the paper.

This script intentionally measures the CPU/mock path because it is the
universally runnable GS-DroneGym configuration. It does not benchmark real 3DGS
rendering, which depends on a user-provided Gaussian scene and a compatible CUDA
stack.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import gs_dronegym
from gs_dronegym.benchmarks import make_benchmark
from gs_dronegym.renderer.camera_model import CameraModel
from gs_dronegym.renderer.mock_renderer import MockRenderer


def measure_mock_renderer(n_frames: int) -> dict[str, float]:
    """Measure deterministic 224x224 mock-renderer throughput.

    Args:
        n_frames: Number of frames to render after warmup.

    Returns:
        Frames-per-second and milliseconds-per-frame metrics.
    """
    camera = CameraModel(image_width=224, image_height=224)
    renderer = MockRenderer(camera)
    pose = np.eye(4, dtype=np.float32)
    for _ in range(5):
        renderer.render(pose)
    start = time.perf_counter()
    for _ in range(n_frames):
        renderer.render(pose)
    elapsed = time.perf_counter() - start
    return {
        "mock_renderer_fps_224": float(n_frames / elapsed),
        "mock_renderer_ms_per_frame_224": float((elapsed / n_frames) * 1000.0),
    }


def measure_mock_environment(n_steps: int) -> dict[str, float | int]:
    """Measure full environment stepping throughput with mock rendering.

    Args:
        n_steps: Number of Gymnasium steps to run.

    Returns:
        Environment steps-per-second, milliseconds-per-step, and reset count.
    """
    env = gs_dronegym.make("PointNav-v0", scene=None)
    observation, _ = env.reset(seed=0)
    del observation
    resets = 0
    start = time.perf_counter()
    for step_index in range(n_steps):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        del observation
        if terminated or truncated:
            resets += 1
            env.reset(seed=1000 + step_index)
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "env_steps_per_second_mock": float(n_steps / elapsed),
        "env_ms_per_step_mock": float((elapsed / n_steps) * 1000.0),
        "env_resets_during_measurement": resets,
    }


def evaluate_small_baselines(n_episodes: int) -> dict[str, dict[str, float]]:
    """Evaluate zero and random policies on the mock PointNav benchmark.

    Args:
        n_episodes: Number of episodes per policy.

    Returns:
        Nested metric dictionary.
    """
    zero_report = make_benchmark(
        "gs_dronegym", env_id="PointNav-v0", scene=None
    ).evaluate_policy(policy=None, n_episodes=n_episodes, seed=10)

    rng = np.random.default_rng(123)

    def random_policy(_: dict[str, object]) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=4).astype(np.float32)

    random_report = make_benchmark(
        "gs_dronegym", env_id="PointNav-v0", scene=None
    ).evaluate_policy(policy=random_policy, n_episodes=n_episodes, seed=20)

    def summarize(report_dict: dict[str, object]) -> dict[str, float]:
        core = report_dict["core_metrics"]
        benchmark = report_dict["benchmark_metrics"]
        if not isinstance(core, dict) or not isinstance(benchmark, dict):
            raise TypeError("Unexpected benchmark report structure.")
        return {
            "success_rate": float(core["success_rate"]),
            "mean_return": float(core["mean_return"]),
            "mean_episode_length": float(core["mean_episode_length"]),
            "collision_rate": float(benchmark["collision_rate"]),
            "spl": float(benchmark["spl"]),
        }

    return {
        f"zero_policy_{n_episodes}ep": summarize(zero_report.to_dict()),
        f"random_policy_{n_episodes}ep": summarize(random_report.to_dict()),
    }


def main() -> None:
    """Run local paper measurements and write a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("paper/local_measurements.json"))
    args = parser.parse_args()

    results: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    results.update(measure_mock_renderer(args.frames))
    results.update(measure_mock_environment(args.steps))
    results.update(evaluate_small_baselines(args.episodes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
