"""CLI helpers for training and exporting Gaussian splat scenes with Nerfstudio.

The CLI shells out to installed Nerfstudio commands so GS-DroneGym users can
convert real image collections into reusable Gaussian splat scene assets.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run the scene-building CLI."""
    parser = argparse.ArgumentParser(
        description="Train a Gaussian splat scene from an image directory using Nerfstudio."
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing input images.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd() / "outputs" / "scene_build",
        help="Workspace directory for intermediate Nerfstudio outputs.",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="images",
        choices=["images", "video"],
        help="Input data type understood by ns-process-data.",
    )
    parser.add_argument(
        "--export-name",
        type=str,
        default="scene_gaussians.ply",
        help="Filename for the exported Gaussian scene.",
    )
    args = parser.parse_args()

    _ensure_command("ns-process-data")
    _ensure_command("ns-train")
    _ensure_command("ns-export")

    image_dir = args.image_dir.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    processed_dir = workspace / "processed"
    train_dir = workspace / "train"
    export_path = workspace / args.export_name

    workspace.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    process_cmd = [
        "ns-process-data",
        args.data_type,
        "--data",
        str(image_dir),
        "--output-dir",
        str(processed_dir),
    ]
    train_cmd = [
        "ns-train",
        "splatfacto",
        "--data",
        str(processed_dir),
        "--output-dir",
        str(train_dir),
    ]
    export_cmd = [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(_latest_config(train_dir)),
        "--output-path",
        str(export_path),
    ]

    _run(process_cmd)
    _run(train_cmd)
    _run(export_cmd)
    LOGGER.info("Exported Gaussian scene to %s", export_path)


def _ensure_command(name: str) -> None:
    """Ensure a required external command is available.

    Args:
        name: Command name.

    Raises:
        RuntimeError: If the command is missing.
    """
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required Nerfstudio command '{name}' is not available on PATH."
        )


def _latest_config(train_dir: Path) -> Path:
    """Locate the latest Nerfstudio config file in a training workspace.

    Args:
        train_dir: Nerfstudio output directory.

    Returns:
        Path to the newest ``config.yml``.

    Raises:
        FileNotFoundError: If no config was produced by training.
    """
    configs = sorted(train_dir.rglob("config.yml"), key=lambda item: item.stat().st_mtime)
    if not configs:
        raise FileNotFoundError(f"No Nerfstudio config.yml found in {train_dir}")
    return configs[-1]


def _run(command: list[str]) -> None:
    """Execute a subprocess command with logging.

    Args:
        command: Command and arguments.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    LOGGER.info("Running command: %s", " ".join(command))
    subprocess.run(command, check=True)
