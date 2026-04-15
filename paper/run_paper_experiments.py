"""Run reproducible small-scale experiments for the GS-DroneGym paper.

The default configuration is intentionally modest so it can run on a laptop.
For a camera-ready empirical section, increase ``--episodes-per-scene`` and
``--eval-episodes`` and run with real Gaussian scenes instead of mock scenes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gs_dronegym.baselines import (
    BehaviorCloningConfig,
    evaluate_behavior_cloning,
    train_behavior_cloning,
)
from gs_dronegym.benchmarks import make_benchmark
from gs_dronegym.data.generation import (
    DatasetGenerationConfig,
    SceneSelectionConfig,
    generate_dataset,
    load_generated_dataset,
    validate_generated_dataset,
)


def _policy_metrics(report_dict: dict[str, object]) -> dict[str, float]:
    """Extract compact live-evaluation metrics from a benchmark report."""
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
        "mean_path_length": float(benchmark["mean_path_length"]),
    }


def _evaluate_live_policies(
    checkpoint_policy: object,
    eval_episodes: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Evaluate zero, random, and BC policies in mock PointNav."""
    benchmark = make_benchmark("gs_dronegym", env_id="PointNav-v0", scene=None)
    zero = benchmark.evaluate_policy(policy=None, n_episodes=eval_episodes, seed=seed)

    rng = np.random.default_rng(seed + 123)

    def random_policy(_: dict[str, object]) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=4).astype(np.float32)

    random = benchmark.evaluate_policy(
        policy=random_policy,
        n_episodes=eval_episodes,
        seed=seed + 1000,
    )
    bc = benchmark.evaluate_policy(
        policy=checkpoint_policy,
        n_episodes=eval_episodes,
        seed=seed + 2000,
    )
    return {
        "zero": _policy_metrics(zero.to_dict()),
        "random": _policy_metrics(random.to_dict()),
        "behavior_cloning": _policy_metrics(bc.to_dict()),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Run dataset generation, BC training, validation, and live evaluation."""
    dataset_root = Path(args.dataset_root)
    if args.force and dataset_root.exists():
        shutil.rmtree(dataset_root)

    results: dict[str, object] = {
        "config": vars(args),
        "timings_s": {},
    }

    if not (dataset_root / "manifest.json").exists():
        start = time.perf_counter()
        manifest = generate_dataset(
            DatasetGenerationConfig(
                output_root=dataset_root,
                scene_selection=SceneSelectionConfig(
                    sources=tuple(str(scene) for scene in args.scenes),
                ),
                episodes_per_scene=args.episodes_per_scene,
                shard_size_episodes=args.shard_size,
                image_size=(args.width, args.height),
                renderer_device=args.renderer_device,
                seed=args.seed,
                allow_mock_rendering=args.allow_mock_rendering,
                dataset_id=args.dataset_id,
            )
        )
        timings = results["timings_s"]
        if not isinstance(timings, dict):
            raise TypeError("Unexpected timings container.")
        timings["generate_dataset"] = time.perf_counter() - start
        results["manifest"] = manifest.to_dict()
    else:
        results["manifest"] = json.loads((dataset_root / "manifest.json").read_text())

    start = time.perf_counter()
    validation = validate_generated_dataset(dataset_root)
    timings = results["timings_s"]
    if not isinstance(timings, dict):
        raise TypeError("Unexpected timings container.")
    timings["validate_dataset"] = time.perf_counter() - start
    results["validation"] = validation.to_dict()

    episodes = load_generated_dataset(dataset_root)
    start = time.perf_counter()
    policy, training = train_behavior_cloning(
        episodes=episodes,
        config=BehaviorCloningConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        ),
        split="train",
        checkpoint_path=args.checkpoint,
    )
    timings["train_behavior_cloning"] = time.perf_counter() - start
    results["training"] = training.to_dict()
    results["dataset_eval"] = {
        "train": evaluate_behavior_cloning(policy, episodes, split="train"),
        "val": evaluate_behavior_cloning(policy, episodes, split="val"),
    }

    start = time.perf_counter()
    results["live_eval"] = _evaluate_live_policies(
        checkpoint_policy=policy,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
    timings["live_eval"] = time.perf_counter() - start
    results["timings_s"] = {key: float(value) for key, value in timings.items()}
    results["training_config"] = asdict(
        BehaviorCloningConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Create the paper experiment parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="outputs/paper_baseline_dataset",
        help="Generated dataset root.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["mock://paper_a", "mock://paper_b", "mock://paper_c"],
        help="Scene handles. Use real .ply paths or built-in scene names for real rendering.",
    )
    parser.add_argument("--episodes-per-scene", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--renderer-device", default="cpu")
    parser.add_argument("--allow-mock-rendering", action="store_true")
    parser.add_argument("--dataset-id", default="paper_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint", default="outputs/paper_baseline_policy.pt")
    parser.add_argument("--output", default="paper/paper_experiment_results.json")
    parser.add_argument("--force", action="store_true", help="Regenerate the dataset root.")
    return parser


def main() -> None:
    """Run the configured paper experiment and write a JSON report."""
    parser = build_parser()
    args = parser.parse_args()
    results = run_experiment(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
