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
from gs_dronegym.scene.occupancy import (
    OccupancyGrid,
    build_occupancy_grid,
    derive_scene_bounds,
    load_gaussian_cloud,
)
from gs_dronegym.tasks import BaseTask
from gs_dronegym.utils.instruction_encoder import (
    DEFAULT_INSTRUCTION_DIM,
    encode_instruction,
)

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
        control_hz: float = 10.0,
        physics_hz: float = 200.0,
        camera_hz: float | None = None,
        observation_mode: Literal["full", "state"] = "full",
        episode_time_limit_s: float | None = None,
        instruction_mode: Literal["text", "features", "both"] = "text",
        instruction_dim: int = DEFAULT_INSTRUCTION_DIM,
        scene_collision: bool = True,
        occupancy_voxel_size: float = 0.1,
        body_radius: float = 0.15,
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
            control_hz: Rate at which actions are applied and observations are
                returned. This is the policy's decision rate.
            physics_hz: Internal integration rate. Must be at least
                ``control_hz``; the integrator substeps between decisions.
            camera_hz: Rate at which new frames are rasterized. ``None`` renders
                on every control step. A lower rate holds the most recent frame,
                which is what a real camera slower than the control loop does.
            observation_mode: ``"full"`` returns rgb and depth; ``"state"`` omits
                them entirely and skips rasterization, for state-based training.
            episode_time_limit_s: Truncation horizon in simulated seconds.
                ``None`` keeps the task's step-count limit.
            instruction_mode: ``"text"`` exposes the raw instruction string,
                ``"features"`` exposes only its deterministic fixed-width
                encoding, and ``"both"`` exposes each. Standard RL libraries
                cannot consume a text space inside a dictionary observation, so
                ``"features"`` is the trainable configuration.
            instruction_dim: Width of the encoded instruction features.
            scene_collision: Derive navigation bounds and collision geometry
                from the scene's own Gaussians when a Gaussian scene is loaded.
                Without this, obstacles are hand-authored primitives unrelated
                to the rendered image.
            occupancy_voxel_size: Voxel edge length for the derived occupancy.
            body_radius: Drone collision radius used to probe the occupancy.

        Raises:
            ValueError: If the rate configuration is inconsistent.
        """
        if instruction_dim <= 0:
            raise ValueError(f"instruction_dim must be positive, got {instruction_dim}.")
        if control_hz <= 0.0:
            raise ValueError(f"control_hz must be positive, got {control_hz}.")
        if physics_hz < control_hz:
            raise ValueError(
                f"physics_hz ({physics_hz}) must be at least control_hz ({control_hz})."
            )
        if camera_hz is not None and (camera_hz <= 0.0 or camera_hz > control_hz):
            raise ValueError(
                f"camera_hz must be in (0, control_hz]; got {camera_hz} with "
                f"control_hz {control_hz}."
            )
        super().__init__()
        self.task = task
        self.scene_path = scene_path
        self.renderer_device = renderer_device
        self.use_depth = bool(use_depth)
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.camera = CameraModel(image_width=image_size[0], image_height=image_size[1])
        self.control_hz = float(control_hz)
        self.physics_hz = float(physics_hz)
        self.camera_hz = None if camera_hz is None else float(camera_hz)
        self.observation_mode = observation_mode
        self.instruction_mode = instruction_mode
        self.instruction_dim = int(instruction_dim)
        self.scene_collision = bool(scene_collision)
        self.occupancy_voxel_size = float(occupancy_voxel_size)
        self.body_radius = float(body_radius)
        self.occupancy: OccupancyGrid | None = None
        self._render_interval_steps = (
            1 if camera_hz is None else max(1, int(round(self.control_hz / self.camera_hz)))
        )
        self._max_episode_steps = (
            None
            if episode_time_limit_s is None
            else max(1, int(round(float(episode_time_limit_s) * self.control_hz)))
        )
        self._cached_render: dict[str, np.ndarray] | None = None
        self._steps_since_render = 0
        self.dynamics = QuadrotorDynamics(
            sim_dt=1.0 / self.physics_hz,
            obs_dt=1.0 / self.control_hz,
        )
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
        self._task_rng = np.random.default_rng()
        self._augmentation_rng = np.random.default_rng()
        self._bind_random_streams()

        observation_space: dict[str, spaces.Space[object]] = {
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(12,),
                dtype=np.float32,
            ),
        }
        if self.instruction_mode in {"text", "both"}:
            observation_space["instruction"] = spaces.Text(
                max_length=256,
                charset=" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),.-_",
            )
        if self.instruction_mode in {"features", "both"}:
            observation_space["instruction_features"] = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(self.instruction_dim,),
                dtype=np.float32,
            )
        if self.observation_mode == "full":
            observation_space["rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera.height, self.camera.width, 3),
                dtype=np.uint8,
            )
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
        if seed is not None:
            self._seed_value = seed
            self._reseed_random_streams(seed)
        self.controller.reset()
        self.step_count = 0
        self._cached_render = None
        self._steps_since_render = 0

        init_state, goal_position, instruction = self.task.reset(self.scene_bbox)
        self.goal_position = goal_position.astype(np.float32)
        self.instruction = instruction
        self.dynamics.set_collision_geometry(
            self.scene_bbox,
            self.task.get_obstacles(),
            occupancy=self.occupancy,
            body_radius=self.body_radius if self.occupancy is not None else 0.0,
        )
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
        step_limit = (
            self.task.config.max_steps
            if self._max_episode_steps is None
            else self._max_episode_steps
        )
        truncated = bool(self.step_count >= step_limit and not terminated)
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

        if self.scene_collision and resolved_path is not None:
            self._derive_scene_geometry(resolved_path)

        self.renderer = GSplatRenderer(
            scene_path=resolved_path,
            camera=self.camera,
            device=self.renderer_device,
        )

    def _derive_scene_geometry(self, scene_path: str | Path) -> None:
        """Derive navigation bounds and occupancy from the scene's Gaussians.

        Hand-authored bounds and obstacles describe a different world from the
        one the renderer draws. Deriving both from the same Gaussians keeps
        geometry and pixels consistent.

        Args:
            scene_path: Path to the resolved Gaussian scene.
        """
        try:
            cloud = load_gaussian_cloud(scene_path)
            bounds = derive_scene_bounds(cloud)
            self.occupancy = build_occupancy_grid(
                cloud,
                voxel_size=self.occupancy_voxel_size,
                bounds=bounds,
            )
            self.scene_bbox = bounds
            LOGGER.info(
                "Derived scene geometry from %d Gaussians; %.2f%% of voxels occupied.",
                cloud.n_gaussians,
                100.0 * self.occupancy.occupancy_fraction(),
            )
        except Exception as exc:
            LOGGER.warning(
                "Could not derive scene geometry from %s (%s). "
                "Falling back to task-authored obstacles.",
                scene_path,
                exc,
            )
            self.occupancy = None

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

    def _reseed_random_streams(self, seed: int) -> None:
        """Rebuild the task and augmentation streams from one seed.

        The two streams are spawned independently so that toggling
        augmentation cannot change the episode layout drawn by the task.

        Args:
            seed: Base seed supplied to :meth:`reset`.
        """
        task_seed, augmentation_seed = np.random.SeedSequence(seed).spawn(2)
        self._task_rng = np.random.default_rng(task_seed)
        self._augmentation_rng = np.random.default_rng(augmentation_seed)
        self._bind_random_streams()

    def _bind_random_streams(self) -> None:
        """Hand the owned generators to the task and augmentor.

        Both components draw from these generators directly, so episodes
        continue one stream instead of restarting it on every reset.
        """
        self.task.set_rng(self._task_rng)
        if self.augmentor is not None:
            self.augmentor.set_rng(self._augmentation_rng)

    def _build_observation(self, drone_state: np.ndarray) -> dict[str, object]:
        """Construct the observation dictionary.

        Args:
            drone_state: Current drone state.

        Returns:
            Observation dictionary conforming to ``observation_space``.
        """
        state_out = drone_state.astype(np.float32, copy=True)

        if self.observation_mode == "state":
            if self.augmentor is not None and self.augmentor.config.imu_noise:
                state_out = self.augmentor.imu_noise(
                    state_out, self.augmentor.config.imu_noise_sigma
                )
            observation: dict[str, object] = {"state": state_out}
            self._attach_instruction(observation)
            return observation

        rgb, depth = self._current_frame(drone_state)

        if self.augmentor is not None:
            rgb, depth = self.augmentor.apply(rgb, depth, state_out)
            if self.augmentor.config.imu_noise:
                state_out = self.augmentor.imu_noise(
                    state_out, self.augmentor.config.imu_noise_sigma
                )

        observation = {"rgb": rgb, "state": state_out}
        self._attach_instruction(observation)
        if self.use_depth:
            observation["depth"] = depth
        return observation

    def _attach_instruction(self, observation: dict[str, object]) -> None:
        """Add the instruction to an observation in the configured form.

        Args:
            observation: Observation dictionary to populate in place.
        """
        if self.instruction_mode in {"text", "both"}:
            observation["instruction"] = self.instruction
        if self.instruction_mode in {"features", "both"}:
            observation["instruction_features"] = encode_instruction(
                self.instruction, self.instruction_dim
            )

    def _current_frame(self, drone_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the frame the camera would currently expose.

        A camera slower than the control loop holds its most recent frame, so
        the policy sees a stale image between exposures rather than a fresh
        rasterization on every decision.

        Args:
            drone_state: Current drone state.

        Returns:
            Tuple of RGB and depth arrays.
        """
        if self._cached_render is not None:
            self._steps_since_render += 1
        if (
            self._cached_render is None
            or self._steps_since_render >= self._render_interval_steps
        ):
            w2c = self.camera.get_extrinsics(drone_state)
            render = self.renderer.render(w2c)
            self._cached_render = {
                "rgb": render["rgb"].astype(np.uint8),
                "depth": render["depth"].astype(np.float32),
            }
            self._steps_since_render = 0
        cached = self._cached_render
        return cached["rgb"].copy(), cached["depth"].copy()

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
