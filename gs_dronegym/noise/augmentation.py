"""Observation augmentation utilities for GS-DroneGym.

These augmentations mimic common perception and state-estimation artifacts
encountered on real drones, helping researchers study sim-to-real robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(slots=True)
class AugmentationConfig:
    """Configuration for visual and IMU augmentations."""

    motion_blur: bool = True
    exposure_jitter: bool = True
    depth_noise: bool = True
    imu_noise: bool = True
    motion_blur_max_kernel: int = 7
    exposure_strength: float = 0.2
    depth_noise_sigma: float = 0.02
    imu_noise_sigma: float = 0.01


class VisualAugmentor:
    """Apply visual and state perturbations to environment observations."""

    def __init__(self, config: AugmentationConfig) -> None:
        """Initialize the augmentor.

        Args:
            config: Augmentation configuration.
        """
        self.config = config
        self._rng = np.random.default_rng()

    def apply(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply all enabled visual augmentations.

        Args:
            rgb: RGB image as ``uint8`` array.
            depth: Depth image as ``float32`` array.
            state: Drone state vector.

        Returns:
            Augmented RGB and depth images.
        """
        rgb_aug = rgb.astype(np.uint8, copy=True)
        depth_aug = depth.astype(np.float32, copy=True)
        speed = float(np.linalg.norm(np.asarray(state, dtype=np.float32)[3:6]))

        if self.config.motion_blur:
            rgb_aug = self.motion_blur(rgb_aug, speed)
        if self.config.exposure_jitter:
            rgb_aug = self.exposure_jitter(rgb_aug, self.config.exposure_strength)
        if self.config.depth_noise:
            depth_aug = self.depth_noise(depth_aug, self.config.depth_noise_sigma)
        return rgb_aug, depth_aug

    def motion_blur(self, rgb: np.ndarray, velocity_magnitude: float) -> np.ndarray:
        """Apply motion blur with strength proportional to speed.

        Args:
            rgb: Input RGB image.
            velocity_magnitude: Drone speed in meters per second.

        Returns:
            Blurred RGB image.
        """
        if velocity_magnitude <= 0.05:
            return rgb
        radius = min(
            self.config.motion_blur_max_kernel,
            max(1, int(round(velocity_magnitude * 1.5))),
        )
        image = Image.fromarray(rgb, mode="RGB")
        blurred = image.filter(ImageFilter.BoxBlur(radius=radius))
        return np.asarray(blurred, dtype=np.uint8)

    def exposure_jitter(self, rgb: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """Apply multiplicative exposure jitter.

        Args:
            rgb: Input RGB image.
            strength: Maximum fractional brightness perturbation.

        Returns:
            Brightness-jittered RGB image.
        """
        factor = float(self._rng.uniform(1.0 - strength, 1.0 + strength))
        jittered = np.clip(rgb.astype(np.float32) * factor, 0.0, 255.0)
        return jittered.astype(np.uint8)

    def depth_noise(self, depth: np.ndarray, sigma: float = 0.02) -> np.ndarray:
        """Apply additive Gaussian depth noise.

        Args:
            depth: Input depth map.
            sigma: Standard deviation of noise in meters.

        Returns:
            Noisy depth map.
        """
        noise = self._rng.normal(loc=0.0, scale=sigma, size=depth.shape).astype(np.float32)
        return np.clip(depth + noise, 0.0, None).astype(np.float32)

    def imu_noise(self, state: np.ndarray, accel_noise: float = 0.01) -> np.ndarray:
        """Apply noise to velocity and angular rate components.

        Args:
            state: Drone state vector.
            accel_noise: Standard deviation of state perturbation.

        Returns:
            Noisy state vector.
        """
        state_noisy = np.asarray(state, dtype=np.float32).copy()
        perturb = self._rng.normal(loc=0.0, scale=accel_noise, size=6).astype(np.float32)
        state_noisy[3:6] += perturb[:3]
        state_noisy[9:12] += perturb[3:]
        return state_noisy
