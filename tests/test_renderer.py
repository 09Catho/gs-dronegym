"""Tests for renderer interfaces and fallback behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_dronegym.renderer import CameraModel, GSplatRenderer, MockRenderer


def test_mock_renderer_output_shapes_and_dtypes() -> None:
    """Mock renderer should emit correctly shaped arrays."""
    camera = CameraModel(image_width=64, image_height=48)
    renderer = MockRenderer(camera=camera)
    render = renderer.render(np.eye(4, dtype=np.float32))
    assert render["rgb"].shape == (48, 64, 3)
    assert render["rgb"].dtype == np.uint8
    assert render["depth"].shape == (48, 64)
    assert render["depth"].dtype == np.float32
    assert render["alpha"].shape == (48, 64)
    assert render["alpha"].dtype == np.float32


def test_mock_renderer_batch_length() -> None:
    """Batch rendering should return one item per pose."""
    camera = CameraModel(image_width=32, image_height=32)
    renderer = MockRenderer(camera=camera)
    batch = renderer.render_batch([np.eye(4, dtype=np.float32) for _ in range(3)])
    assert len(batch) == 3


def test_gsplat_renderer_smoke(tmp_path: Path) -> None:
    """GSplat renderer should instantiate and render when gsplat is available."""
    pytest.importorskip("gsplat")
    scene_path = tmp_path / "tiny_scene.ply"
    scene_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
                "end_header",
                "0 0 2 5 0 0 0 1 0 0 0 0.5 0.5 0.5",
            ]
        ),
        encoding="utf-8",
    )
    renderer = GSplatRenderer(scene_path=scene_path, camera=CameraModel(32, 32), device="cpu")
    frame = renderer.render(np.eye(4, dtype=np.float32))
    assert frame["rgb"].shape == (32, 32, 3)
