"""Abstract environment base class for GS-DroneGym.

This module provides a small shared base for concrete GS-DroneGym environments.
It exists primarily to keep the public environment surface explicit and stable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np


class BaseDroneEnv(gym.Env, ABC):
    """Abstract base class for drone environments."""

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Reset the environment state.

        Args:
            seed: Optional RNG seed.
            options: Optional reset options.

        Returns:
            Observation dictionary and reset info.
        """

    @abstractmethod
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        """Advance the environment by one step.

        Args:
            action: Action vector.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
