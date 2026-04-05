"""CLI entrypoint for validating generated synthetic datasets."""

from __future__ import annotations

import argparse
import json

from gs_dronegym.data import validate_generated_dataset


def build_parser() -> argparse.ArgumentParser:
    """Create the dataset validation parser."""
    parser = argparse.ArgumentParser(description="Validate a generated synthetic dataset root.")
    parser.add_argument("dataset_root", help="Generated dataset root directory.")
    return parser


def main() -> None:
    """Run the dataset validation CLI."""
    parser = build_parser()
    args = parser.parse_args()
    report = validate_generated_dataset(args.dataset_root)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
