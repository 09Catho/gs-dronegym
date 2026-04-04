"""Train the lightweight behavior-cloning baseline on a normalized dataset."""

from __future__ import annotations

import json
from pathlib import Path

from gs_dronegym import load_dataset
from gs_dronegym.baselines import BehaviorCloningConfig, train_behavior_cloning


def main() -> None:
    """Train a baseline on a saved normalized dataset."""
    episodes = load_dataset(Path("outputs") / "drone_rollout.json", format="gs_dronegym")
    _, summary = train_behavior_cloning(
        episodes,
        config=BehaviorCloningConfig(epochs=1, batch_size=2),
        split="eval",
        checkpoint_path=Path("outputs") / "bc_policy.pt",
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
