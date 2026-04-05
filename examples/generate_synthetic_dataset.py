"""Example synthetic dataset generation script for GS-DroneGym."""

from __future__ import annotations

from pathlib import Path

from gs_dronegym.data import DatasetGenerationConfig, SceneSelectionConfig, generate_dataset


def main() -> None:
    """Generate a tiny mock-backed synthetic dataset."""
    config = DatasetGenerationConfig(
        output_root=Path("outputs") / "synthetic_dataset",
        scene_selection=SceneSelectionConfig(sources=("mock://lab_a", "mock://lab_b")),
        episodes_per_scene=4,
        shard_size_episodes=2,
        renderer_device="cpu",
        allow_mock_rendering=True,
        debug_export_episodes_per_split=1,
        seed=0,
    )
    manifest = generate_dataset(config)
    print(manifest.to_dict())


if __name__ == "__main__":
    main()
