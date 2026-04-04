"""Visualization helpers for trajectories and observation frames."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gs_dronegym.utils.metrics import Episode


def plot_trajectory(
    episode: Episode,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a top-down trajectory colored by instantaneous speed.

    Args:
        episode: Episode to visualize.
        save_path: Optional file path for saving the figure.
        show: Whether to show the figure interactively.

    Returns:
        The created matplotlib figure.
    """
    positions = np.asarray(episode.positions, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(positions) > 1:
        speeds = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        scatter = ax.scatter(
            positions[1:, 0],
            positions[1:, 1],
            c=speeds,
            cmap="viridis",
            s=20,
        )
        fig.colorbar(scatter, ax=ax, label="Step speed (m)")
    ax.plot(positions[:, 0], positions[:, 1], color="black", linewidth=1.0, alpha=0.6)
    ax.scatter(positions[0, 0], positions[0, 1], color="green", s=60, label="Start")
    ax.scatter(
        episode.goal_position[0],
        episode.goal_position[1],
        color="red",
        marker="*",
        s=120,
        label="Goal",
    )
    if episode.collision:
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color="red",
            marker="x",
            s=80,
            label="Collision",
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down trajectory")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.2)
    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight")
    if show:
        plt.show()
    return fig


def render_obs_grid(
    obs_list: list[dict[str, object]],
    n_cols: int = 4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Render a grid of RGB observation frames.

    Args:
        obs_list: Observation dictionaries containing ``rgb`` frames.
        n_cols: Number of columns in the grid.
        save_path: Optional output file path.

    Returns:
        The created matplotlib figure.
    """
    n_items = len(obs_list)
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes_array = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for idx, ax in enumerate(axes_array.flat):
        ax.axis("off")
        if idx >= n_items:
            continue
        rgb = np.asarray(obs_list[idx]["rgb"], dtype=np.uint8)
        ax.imshow(rgb)
        ax.text(
            0.02,
            0.95,
            f"step {idx}",
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            verticalalignment="top",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
        )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), bbox_inches="tight")
    return fig
