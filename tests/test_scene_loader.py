"""Tests for scene loading and archive extraction.

These tests keep the real-scene code path covered without requiring network
downloads or large Gaussian assets in CI.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gs_dronegym.scene.scene_loader import SceneLoader


def _write_minimal_gaussian_ply(path: Path) -> None:
    """Write a tiny Gaussian-splat-compatible PLY for loader tests."""
    vertex = np.array(
        [
            (
                np.float32(0.0),
                np.float32(0.0),
                np.float32(1.0),
                np.float32(0.5),
                np.float32(0.01),
                np.float32(1.0),
                np.float32(0.2),
            )
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("rot_0", "f4"),
            ("f_dc_0", "f4"),
        ],
    )
    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=True)
    ply.write(path)


def test_scene_loader_extracts_ply_from_zip(tmp_path: Path, monkeypatch) -> None:
    """SceneLoader should resolve a cached archive into a validated PLY path."""
    source_ply = tmp_path / "point_cloud.ply"
    _write_minimal_gaussian_ply(source_ply)
    archive_path = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.write(source_ply, arcname="model/point_cloud.ply")

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(SceneLoader, "CACHE_DIR", cache_dir)
    loader = SceneLoader()
    resolved = loader.load(archive_path)

    assert resolved.name == "point_cloud.ply"
    assert resolved.exists()
    assert loader.validate_ply(resolved)
