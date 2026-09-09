"""Shared scene-argument normalization for the command-line entrypoints.

Every CLI that accepts ``--scene`` must treat the textual placeholders for "no
scene" identically. Without this, a value such as ``None`` reaches the renderer
as a literal scene handle and silently takes the Gaussian path before falling
back to the mock renderer.
"""

from __future__ import annotations

#: Textual values that select the mock renderer instead of a scene handle.
NULL_SCENE_TOKENS = frozenset({"", "none", "null"})


def normalize_scene_arg(value: str | None) -> str | None:
    """Convert a raw ``--scene`` value into a scene handle or ``None``.

    Args:
        value: Raw CLI value.

    Returns:
        ``None`` when the value is absent or a null placeholder, otherwise the
        original string.
    """
    if value is None:
        return None
    if value.strip().lower() in NULL_SCENE_TOKENS:
        return None
    return value
