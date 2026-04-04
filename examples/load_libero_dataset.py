"""Load LIBERO demonstrations into the shared trajectory schema."""

from __future__ import annotations

import json
from pathlib import Path

from gs_dronegym import load_dataset
from gs_dronegym.data import summarize_dataset


def main() -> None:
    """Load a LIBERO dataset directory and print a summary."""
    source = Path("path/to/libero_dataset")
    episodes = load_dataset(source, format="libero")
    print(json.dumps(summarize_dataset(episodes), indent=2))


if __name__ == "__main__":
    main()
