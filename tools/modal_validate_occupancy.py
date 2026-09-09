"""Validate Gaussian-derived occupancy against real gsplat rendering on Modal.

Windows cannot build the gsplat CUDA extension without MSVC, so the renderer
silently falls back to the mock path locally and the real geometry claim cannot
be checked on this workstation. This job runs the same code on a Linux GPU where
gsplat compiles, renders a synthetic room whose geometry is known exactly, and
checks that the occupancy grid derived from the Gaussians agrees with the depth
the renderer actually produces.

Run with:
    modal run tools/modal_validate_occupancy.py
"""

from __future__ import annotations

import json

import modal

REPO_ROOT = __file__.rsplit("tools", 1)[0]

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

app = modal.App("gs-dronegym-occupancy-validation", image=image)


@app.function(gpu="T4", timeout=1800)
def validate_occupancy_against_depth(
    image_size: int = 96,
    spacing: float = 0.15,
    voxel_size: float = 0.1,
) -> dict[str, object]:
    """Render a synthetic room with gsplat and compare depth to voxel occupancy.

    Args:
        image_size: Square render resolution.
        spacing: Gaussian sample spacing on room surfaces.
        voxel_size: Occupancy voxel edge length.

    Returns:
        Dictionary of agreement statistics.
    """
    import sys

    sys.path.insert(0, "/root")

    import numpy as np
    import torch

    from gs_dronegym.renderer.camera_model import CameraModel
    from gs_dronegym.renderer.gsplat_renderer import _GSPLAT_AVAILABLE, GSplatRenderer
    from gs_dronegym.scene.occupancy import build_occupancy_grid, load_gaussian_cloud
    from gs_dronegym.scene.synthetic import SyntheticRoomConfig, write_synthetic_room_ply

    room = SyntheticRoomConfig(size=(10.0, 8.0, 3.0), spacing=spacing)
    ply_path = write_synthetic_room_ply("/tmp/room.ply", room)

    camera = CameraModel(image_width=image_size, image_height=image_size)
    renderer = GSplatRenderer(scene_path=ply_path, camera=camera, device="cuda")

    # Camera at room centre, level, looking along +x.
    drone_state = np.zeros(12, dtype=np.float32)
    drone_state[:3] = np.array([-3.0, 0.0, 1.5], dtype=np.float32)
    w2c = camera.get_extrinsics(drone_state)
    rendered = renderer.render(w2c)
    depth = np.asarray(rendered["depth"], dtype=np.float32)
    alpha = np.asarray(rendered["alpha"], dtype=np.float32)

    # Ask the renderer itself whether it fell back, rather than inferring it
    # from imports: a CUDA failure silently swaps in the mock renderer.
    used_real_gsplat = bool(
        _GSPLAT_AVAILABLE
        and torch.cuda.is_available()
        and not getattr(renderer, "_backend_failed", True)
    )
    if not used_real_gsplat:
        raise RuntimeError(
            "Renderer fell back to the mock path; this job only validates real gsplat."
        )

    cloud = load_gaussian_cloud(ply_path)
    grid = build_occupancy_grid(cloud, voxel_size=voxel_size)

    intrinsics = camera.get_intrinsics()
    inv_k = np.linalg.inv(intrinsics.astype(np.float64))
    c2w = np.linalg.inv(w2c.astype(np.float64))
    origin = c2w[:3, 3]

    step = 0.5 * voxel_size
    max_range = 20.0
    n_samples = int(max_range / step)

    stride = max(1, image_size // 32)
    rows = range(0, image_size, stride)
    cols = range(0, image_size, stride)

    ray_errors: list[float] = []
    both_hit = 0
    compared = 0
    for v in rows:
        for u in cols:
            pixel = np.array([u + 0.5, v + 0.5, 1.0], dtype=np.float64)
            dir_cam = inv_k @ pixel
            dir_cam = dir_cam / np.linalg.norm(dir_cam)
            dir_world = c2w[:3, :3] @ dir_cam

            t_values = (np.arange(1, n_samples + 1, dtype=np.float64) * step)
            points = origin[None, :] + t_values[:, None] * dir_world[None, :]
            hits = grid.is_occupied(points.astype(np.float32))
            if not np.any(hits):
                continue
            t_hit = float(t_values[int(np.argmax(hits))])
            occupancy_z = t_hit * float(dir_cam[2])

            rendered_z = float(depth[v, u])
            compared += 1
            if rendered_z <= 0.0 or not np.isfinite(rendered_z):
                continue
            both_hit += 1
            ray_errors.append(abs(rendered_z - occupancy_z))

    errors = np.asarray(ray_errors, dtype=np.float64)
    result: dict[str, object] = {
        "used_real_gsplat": used_real_gsplat,
        "gsplat_available": bool(_GSPLAT_AVAILABLE),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "n_gaussians": int(cloud.n_gaussians),
        "grid_shape": list(grid.shape),
        "occupied_fraction": float(grid.occupancy_fraction()),
        "voxel_size": float(voxel_size),
        "rays_compared": int(compared),
        "rays_with_both_hits": int(both_hit),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_median": float(np.median(depth)),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "alpha_median": float(np.median(alpha)),
    }
    if errors.size:
        result.update(
            {
                "median_abs_error_m": float(np.median(errors)),
                "mean_abs_error_m": float(np.mean(errors)),
                "p90_abs_error_m": float(np.percentile(errors, 90)),
                "fraction_within_1_voxel": float(np.mean(errors <= voxel_size)),
                "fraction_within_2_voxels": float(np.mean(errors <= 2.0 * voxel_size)),
            }
        )
    return result


@app.local_entrypoint()
def main() -> None:
    """Run the validation job and print its report."""
    report = validate_occupancy_against_depth.remote()
    print(json.dumps(report, indent=2))
