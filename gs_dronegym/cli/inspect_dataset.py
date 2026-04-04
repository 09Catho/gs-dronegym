"""CLI entrypoint for inspecting benchmark datasets."""

from __future__ import annotations

import argparse
import json

from gs_dronegym.data.dataset import load_dataset as load_dataset_from_format
from gs_dronegym.data.dataset import summarize_dataset


def build_parser() -> argparse.ArgumentParser:
    """Create the dataset inspection CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Inspect a benchmark dataset.")
    parser.add_argument("source", help="Dataset path or benchmark-specific handle.")
    parser.add_argument(
        "--format",
        choices=["gs_dronegym", "libero", "lerobot"],
        default="gs_dronegym",
        help="Dataset format.",
    )
    return parser


def main() -> None:
    """Run the dataset inspection CLI."""
    parser = build_parser()
    args = parser.parse_args()
    episodes = load_dataset_from_format(args.source, args.format)
    print(json.dumps(summarize_dataset(episodes), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
