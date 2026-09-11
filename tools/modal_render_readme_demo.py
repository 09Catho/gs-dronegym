"""Render a README flythrough of a Gaussian scene with real gsplat on Modal.

The demo scene is the synthetic test room from ``gs_dronegym.scene.synthetic``,
recoloured with a checkerboard floor and an orange pillar so camera motion reads
clearly. It is rendered through the same ``GSplatRenderer`` the environment
uses, on a Linux GPU where gsplat can build its CUDA extension. It is a
synthetic scene, not a captured reconstruction.

Run with:
    modal run tools/modal_render_readme_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = __file__.rsplit("tools", 1)[0]
OUTPUT_GIF = Path(REPO_ROOT) / "assets" / "real_gsplat_flythrough.gif"

#: Zeroth-order spherical-harmonic basis constant used to encode colour.
SH_C0 = 0.28209479177387814

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.24",
        "plyfile>=0.9",
        "gymnasium>=0.29",
        "Pillow>=10.0",
        "matplotlib>=3.7",
        "pyarrow>=16.0",
        index_url="https://download.pytorch.org/whl/cu124",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install("gsplat==1.5.3")
    .env({"TORCH_CUDA_ARCH_LIST": "7.5"})
    .add_local_dir(f"{REPO_ROOT}gs_dronegym", "/root/gs_dronegym")
)

app = modal.App("gs-dronegym-readme-demo", image=image)


def _texture_demo_room(path: str, pillar_half_extent: float) -> None:
    """Recolour the synthetic room so camera motion is visible in a GIF.

    The fixture colours surfaces only by height, which leaves the floor a flat
    uniform plane. A checkerboard floor and a distinct pillar make parallax
    readable without changing any geometry.

    Args:
        path: Path to the synthetic room PLY, rewritten in place.
        pillar_half_extent: Half width of the central pillar in metres.
    """
    import numpy as np
    from plyfile import PlyData, PlyElement

    vertex = PlyData.read(path)["vertex"].data.copy()
    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    colours = np.stack(
        [np.asarray(vertex[f"f_dc_{i}"], dtype=np.float32) * SH_C0 + 0.5 for i in range(3)],
        axis=1,
    )

    floor = z < 1e-3
    checker = (np.floor(x / 0.5) + np.floor(y / 0.5)).astype(np.int64) % 2 == 0
    colours[floor & checker] = (0.85, 0.85, 0.82)
    colours[floor & ~checker] = (0.25, 0.27, 0.30)

    edge = pillar_half_extent + 1e-3
    pillar = (np.abs(x) <= edge) & (np.abs(y) <= edge) & ~floor
    colours[pillar] = (0.95, 0.55, 0.15)

    for channel in range(3):
        vertex[f"f_dc_{channel}"] = ((colours[:, channel] - 0.5) / SH_C0).astype(np.float32)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


@app.function(gpu="T4", timeout=1800)
def render_flythrough(
    width: int = 256,
    height: int = 192,
    n_frames: int = 48,
    spacing: float = 0.05,
) -> dict[str, object]:
    """Orbit the demo room and render RGB and depth for every frame.

    Args:
        width: Render width in pixels.
        height: Render height in pixels.
        n_frames: Number of frames in the orbit.
        spacing: Gaussian sample spacing on room surfaces.

    Returns:
        Dictionary holding the encoded GIF and render statistics.

    Raises:
        RuntimeError: If the renderer falls back to the mock path.
    """
    import sys

    sys.path.insert(0, "/root")

    import io
    import math
    import time

    import numpy as np
    import torch
    from matplotlib import colormaps
    from PIL import Image, ImageDraw

    from gs_dronegym.renderer.camera_model import CameraModel
    from gs_dronegym.renderer.gsplat_renderer import GSplatRenderer
    from gs_dronegym.scene.synthetic import SyntheticRoomConfig, write_synthetic_room_ply

    room = SyntheticRoomConfig(size=(10.0, 8.0, 3.0), spacing=spacing)
    ply_path = str(write_synthetic_room_ply("/tmp/demo_room.ply", room))
    _texture_demo_room(ply_path, room.pillar_half_extent)

    camera = CameraModel(image_width=width, image_height=height)
    renderer = GSplatRenderer(scene_path=ply_path, camera=camera, device="cuda")
    colormap = colormaps["turbo"]
    max_depth = 9.0

    frames = []
    render_seconds: list[float] = []
    depth_medians: list[float] = []
    for index in range(n_frames):
        angle = 2.0 * math.pi * index / n_frames
        state = np.zeros(12, dtype=np.float32)
        state[:3] = (3.2 * math.cos(angle), 2.5 * math.sin(angle), 1.2)
        state[8] = math.atan2(-float(state[1]), -float(state[0]))
        w2c = camera.get_extrinsics(state)

        torch.cuda.synchronize()
        start = time.perf_counter()
        rendered = renderer.render(w2c)
        torch.cuda.synchronize()
        render_seconds.append(time.perf_counter() - start)
        if getattr(renderer, "_backend_failed", True):
            raise RuntimeError("Renderer fell back to the mock path; refusing to produce a demo.")

        rgb = np.asarray(rendered["rgb"], dtype=np.uint8)
        depth = np.asarray(rendered["depth"], dtype=np.float32)
        depth_medians.append(float(np.median(depth)))
        depth_rgb = (colormap(np.clip(depth / max_depth, 0.0, 1.0))[..., :3] * 255.0).astype(
            np.uint8
        )
        panel = Image.fromarray(np.concatenate([rgb, depth_rgb], axis=1))
        draw = ImageDraw.Draw(panel)
        draw.text((6, 4), "RGB (real gsplat)", fill=(255, 255, 255))
        draw.text((width + 6, 4), "Depth, 0-9 m", fill=(255, 255, 255))
        frames.append(panel.quantize(colors=256, method=Image.Quantize.FASTOCTREE))

    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    # The first frame includes CUDA extension compilation, so exclude it.
    steady = render_seconds[1:] or render_seconds
    return {
        "gif": buffer.getvalue(),
        "stats": {
            "gpu_name": torch.cuda.get_device_name(0),
            "n_gaussians": int(renderer.means.shape[0]),
            "n_frames": n_frames,
            "resolution": [width, height],
            "median_render_ms_including_host_copy": 1000.0 * float(np.median(steady)),
            "depth_median_m_over_frames": float(np.median(depth_medians)),
        },
    }


@app.local_entrypoint()
def main() -> None:
    """Render the flythrough remotely and write the GIF into assets."""
    result = render_flythrough.remote()
    gif = result["gif"]
    assert isinstance(gif, bytes)
    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_GIF.write_bytes(gif)
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {OUTPUT_GIF} ({len(gif) / 1e6:.2f} MB)")
