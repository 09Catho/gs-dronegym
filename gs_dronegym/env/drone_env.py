"""Full GS-DroneGym environment combining dynamics, rendering, and task logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gs_dronegym.dynamics import QuadrotorDynamics, WaypointController
from gs_dronegym.env.base_env import BaseDroneEnv
from gs_dronegym.noise import AugmentationConfig, VisualAugmentor
from gs_dronegym.renderer import CameraModel, GSplatRenderer, MockRenderer
from gs_dronegym.scene import BUILTIN_SCENES, SceneLoader, get_scene
from gs_dronegym.tasks import BaseTask

LOGGER = logging.getLogger(__name__)


class GSDroneEnv(BaseDroneEnv):
    """Gymnasium environment for photorealistic drone navigation."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        task: BaseTask,
        scene_path: str | Path | None = None,
        renderer_device: str = "cuda",
        image_size: tuple[int, int] = (224, 224),
        use_depth: bool = True,
        augmentation: bool = False,
        render_mode: str = "rgb_array",
        action_mode: Literal["waypoint", "direct"] = "waypoint",
    ) -> None:
        """Initialize the drone environment.

        Args:
            task: Task instance defining rewards and termination.
            scene_path: Scene path, URL, built-in scene name, or ``None`` for the
                mock renderer.
            renderer_device: Rendering device string.
            image_size: Observation image size as ``(width, height)``.
            use_depth: Whether to include depth in observations.
            augmentation: Whether to enable observation augmentation.
            render_mode: Gymnasium render mode.
            action_mode: Whether actions are waypoint deltas or direct commands.
        """
        super().__init__()
        self.task = task
        self.scene_path = scene_path
        self.renderer_device = renderer_device
        self.use_depth = bool(use_depth)
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.camera = CameraModel(image_width=image_size[0], image_height=image_size[1])
        self.dynamics = QuadrotorDynamics()
        self.controller = WaypointController()
        self.loader = SceneLoader()
        self.augmentor = VisualAugmentor(AugmentationConfig()) if augmentation else None
        self.scene_bbox = np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 5.0]], dtype=np.float32)
        self.renderer: GSplatRenderer | MockRenderer = MockRenderer(camera=self.camera)
        self.instruction = ""
        self.goal_position = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.latest_obs: dict[str, object] | None = None
        self._seed_value: int | None = None

        observation_space: dict[str, spaces.Space[object]] = {
            "rgb": spaces.Box(
                low=0,
                high=255,
                shape=(self.camera.height, self.camera.width, 3),
                dtype=np.uint8,
            ),
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(12,),
                dtype=np.float32,
            ),
            "instruction": spaces.Text(
                max_length=256,
                charset=" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),.-_",
            ),
        }
        if self.use_depth:
            observation_space["depth"] = spaces.Box(
                low=0.0,
                high=100.0,
                shape=(self.camera.height, self.camera.width),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_space)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._initialize_renderer(scene_path)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Reset the environment.

        Args:
            seed: Optional RNG seed.
            options: Optional reset options, currently unused.

        Returns:
            Initial observation and info dictionary.
        """
        del options
        gym.Env.reset(self, seed=seed)
        self._seed_value = seed
        self.task.seed(seed)
        self.controller.reset()
        self.step_count = 0

        init_state, goal_position, instruction = self.task.reset(self.scene_bbox)
        self.goal_position = goal_position.astype(np.float32)
        self.instruction = instruction
        self.dynamics.set_collision_geometry(self.scene_bbox, self.task.get_obstacles())
        drone_state = self.dynamics.reset(init_state)
        obs = self._build_observation(drone_state)
        info = self._build_info(drone_state, collision=False, success=False)
        self.latest_obs = obs
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        """Advance the environment by one step.

        Args:
            action: Normalized 4D action.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        normalized_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_state = self.dynamics.get_state()
        physical_action = self._action_to_command(normalized_action, prev_state)
        drone_state, collision = self.dynamics.step(physical_action)
        self.step_count += 1
        self.task.update(drone_state, self.step_count, self.dynamics.config.obs_dt)
        self.goal_position = self.task.get_goal_position()

        success = self.task.is_success(drone_state)
        failure = self.task.is_failure(drone_state, collision)
        terminated = bool(success or failure)
        truncated = bool(self.step_count >= self.task.config.max_steps and not terminated)
        reward = self.task.compute_reward(drone_state, prev_state, collision, self.step_count)

        obs = self._build_observation(drone_state)
        info = self._build_info(drone_state, collision=collision, success=success)
        self.latest_obs = obs
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the latest RGB frame.

        Returns:
            Latest RGB array or ``None`` if the environment has not been reset.
        """
        if self.latest_obs is None:
            return None
        rgb = np.asarray(self.latest_obs["rgb"], dtype=np.uint8)
        return rgb

    def close(self) -> None:
        """Close environment resources."""

    def _initialize_renderer(self, scene_path: str | Path | None) -> None:
        """Initialize the renderer and scene bounding box.

        Args:
            scene_path: Scene path, built-in scene name, URL, or ``None``.
        """
        resolved_path: str | Path | None = scene_path
        if isinstance(scene_path, str) and scene_path in BUILTIN_SCENES:
            scene_info = BUILTIN_SCENES[scene_path]
            self.scene_bbox = scene_info.bbox.astype(np.float32)
            try:
                resolved_path = get_scene(scene_path)
            except Exception as exc:  # pragma: no cover - network-dependent path
                LOGGER.warning(
                    "Failed to download built-in scene %s (%s). Falling back to MockRenderer.",
                    scene_path,
                    exc,
                )
                self.renderer = MockRenderer(camera=self.camera)
                return
        elif scene_path is None:
            self.renderer = MockRenderer(camera=self.camera)
            return
        else:
            try:
                resolved_local = self.loader.load(scene_path)
                self.scene_bbox = self.loader.infer_bbox(resolved_local)
                resolved_path = resolved_local
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load scene %s (%s). Falling back to MockRenderer.",
                    scene_path,
                    exc,
                )
                self.renderer = MockRenderer(camera=self.camera)
                return

        self.renderer = GSplatRenderer(
            scene_path=resolved_path,
            camera=self.camera,
            device=self.renderer_device,
        )

    def _action_to_command(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Convert a normalized action into a physical dynamics command.

        Args:
            action: Normalized action in ``[-1, 1]``.
            state: Current drone state.

        Returns:
            Physical command for the dynamics model.
        """
        if self.action_mode == "waypoint":
            delta = np.array([1.5, 1.5, 1.0], dtype=np.float32) * action[:3]
            target_position = state[:3] + delta
            target_yaw = float(state[8] + action[3] * (np.pi / 4.0))
            target = np.concatenate(
                [target_position.astype(np.float32), np.array([target_yaw], dtype=np.float32)]
            )
            return self.controller.compute(state, target)

        hover = self.dynamics.hover_thrust
        thrust = float(np.interp(action[0], [-1.0, 1.0], [-hover, 3.0 * hover]))
        rates = (action[1:] * np.float32(np.pi)).astype(np.float32)
        return np.array([thrust, rates[0], rates[1], rates[2]], dtype=np.float32)

    def _build_observation(self, drone_state: np.ndarray) -> dict[str, object]:
        """Construct the observation dictionary.

        Args:
            drone_state: Current drone state.

        Returns:
            Observation dictionary conforming to ``observation_space``.
        """
        w2c = self.camera.get_extrinsics(drone_state)
        render = self.renderer.render(w2c)
        rgb = render["rgb"].astype(np.uint8)
        depth = render["depth"].astype(np.float32)
        state_out = drone_state.astype(np.float32, copy=True)

        if self.augmentor is not None:
            rgb, depth = self.augmentor.apply(rgb, depth, state_out)
            if self.augmentor.config.imu_noise:
                state_out = self.augmentor.imu_noise(
                    state_out, self.augmentor.config.imu_noise_sigma
                )

        observation: dict[str, object] = {
            "rgb": rgb,
            "state": state_out,
            "instruction": self.instruction,
        }
        if self.use_depth:
            observation["depth"] = depth
        return observation

    def _build_info(
        self,
        drone_state: np.ndarray,
        collision: bool,
        success: bool,
    ) -> dict[str, object]:
        """Build the ``info`` dictionary for the current state.

        Args:
            drone_state: Current drone state.
            collision: Whether a collision occurred.
            success: Whether the task succeeded.

        Returns:
            Info dictionary.
        """
        distance = float(np.linalg.norm(drone_state[:3] - self.goal_position))
        return {
            "collision": collision,
            "success": success,
            "distance_to_goal": distance,
            "step": self.step_count,
            "drone_state": drone_state.astype(np.float32, copy=True),
        }
