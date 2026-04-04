"""CLI entrypoint for the behavior-cloning baseline."""

from __future__ import annotations

import argparse
import json

from gs_dronegym.baselines import BehaviorCloningConfig, train_behavior_cloning
from gs_dronegym.data.dataset import load_dataset as load_dataset_from_format


def build_parser() -> argparse.ArgumentParser:
    """Create the behavior-cloning training parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Train a behavior-cloning baseline.")
    parser.add_argument("source", help="Dataset path or benchmark-specific handle.")
    parser.add_argument(
        "--format",
        choices=["gs_dronegym", "libero", "lerobot"],
        default="gs_dronegym",
        help="Dataset format.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--split", default="train", help="Dataset split to train on.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint output path.")
    return parser


def main() -> None:
    """Run the behavior-cloning training CLI."""
    parser = build_parser()
    args = parser.parse_args()
    episodes = load_dataset_from_format(args.source, args.format)
    config = BehaviorCloningConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    _, summary = train_behavior_cloning(
        episodes=episodes,
        config=config,
        split=args.split,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
