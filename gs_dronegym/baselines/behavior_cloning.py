"""A lightweight offline behavior-cloning baseline for shared trajectories.

This baseline is intentionally simple but fully runnable. It supports image,
state, and instruction inputs from the shared trajectory schema and produces a
single continuous action vector, making it useful as a reproducible baseline
across GS-DroneGym, LIBERO, and LeRobot-derived datasets.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gs_dronegym.data.dataset import iter_transitions
from gs_dronegym.data.schema import JsonValue, TrajectoryEpisode

LOGGER = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def _hash_instruction(text: str, dimension: int) -> np.ndarray:
    """Convert free-form text into a fixed-size hashed bag-of-words vector.

    Args:
        text: Instruction string.
        dimension: Output vector dimension.

    Returns:
        Dense float32 feature vector.
    """
    features = np.zeros(dimension, dtype=np.float32)
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return features
    for token in tokens:
        features[hash(token) % dimension] += 1.0
    features /= np.float32(max(len(tokens), 1))
    return features


def _prepare_image(observation: dict[str, object]) -> np.ndarray | None:
    """Extract a model-ready image tensor from one observation.

    Args:
        observation: Observation dictionary.

    Returns:
        Float32 image in CHW layout, or ``None`` if unavailable.
    """
    rgb_value = observation.get("rgb")
    if isinstance(rgb_value, np.ndarray) and rgb_value.ndim == 3:
        rgb = rgb_value.astype(np.float32) / np.float32(255.0)
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    depth_value = observation.get("depth")
    if isinstance(depth_value, np.ndarray) and depth_value.ndim == 2:
        depth = depth_value.astype(np.float32)[None, :, :]
        return depth
    return None


@dataclass(slots=True)
class BehaviorCloningConfig:
    """Training configuration for the behavior-cloning baseline."""

    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-3
    device: str = "cpu"
    hidden_dim: int = 128
    instruction_dim: int = 128
    num_workers: int = 0


@dataclass(slots=True)
class TrainingSummary:
    """Compact training summary suitable for CLI output and reports."""

    n_examples: int
    action_dim: int
    final_train_loss: float
    train_loss_history: list[float]
    checkpoint_path: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the summary for JSON output.

        Returns:
            JSON-safe dictionary.
        """
        return cast(dict[str, JsonValue], json.loads(json.dumps(asdict(self))))


class TrajectoryStepDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset flattening trajectory episodes into step samples."""

    def __init__(
        self,
        episodes: list[TrajectoryEpisode],
        instruction_dim: int,
        split: str | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            episodes: Source episodes.
            instruction_dim: Hashed instruction feature dimension.
            split: Optional split name to filter on.
        """
        filtered = [
            episode
            for episode in episodes
            if split is None or episode.split == split
        ]
        if not filtered:
            raise ValueError("No episodes available for behavior-cloning training.")
        self.samples: list[dict[str, torch.Tensor]] = []
        self.action_dim = filtered[0].action_spec.shape[0]
        self.state_dim = 0
        self.image_channels = 0
        for transition in iter_transitions(filtered):
            observation = transition.observation
            state_value = observation.get("state")
            state = (
                np.asarray(state_value, dtype=np.float32).reshape(-1)
                if isinstance(state_value, np.ndarray)
                else np.zeros(0, dtype=np.float32)
            )
            image = _prepare_image(observation)
            instruction = str(observation.get("instruction", ""))
            hashed_instruction = _hash_instruction(instruction, instruction_dim)
            sample = {
                "state": torch.from_numpy(state),
                "instruction": torch.from_numpy(hashed_instruction),
                "action": torch.from_numpy(
                    np.asarray(transition.action, dtype=np.float32).reshape(-1)
                ),
            }
            if image is not None:
                sample["image"] = torch.from_numpy(image)
                self.image_channels = image.shape[0]
            else:
                sample["image"] = torch.zeros((0, 1, 1), dtype=torch.float32)
            self.state_dim = max(self.state_dim, int(state.shape[0]))
            self.samples.append(sample)
        if not self.samples:
            raise ValueError("Behavior-cloning dataset is empty after flattening.")

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Dataset length.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one training sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary of model tensors.
        """
        sample = self.samples[index]
        state = sample["state"]
        if state.numel() < self.state_dim:
            padding = torch.zeros(self.state_dim - state.numel(), dtype=torch.float32)
            state = torch.cat([state, padding], dim=0)
        return {
            "state": state,
            "instruction": sample["instruction"],
            "action": sample["action"],
            "image": sample["image"],
        }


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a list of step samples into a training batch.

    Args:
        batch: List of samples.

    Returns:
        Batched tensor dictionary.
    """
    states = torch.stack([item["state"] for item in batch], dim=0)
    instructions = torch.stack([item["instruction"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    images = [item["image"] for item in batch]
    if images and images[0].numel() > 0:
        image_batch = torch.stack(images, dim=0)
    else:
        image_batch = torch.zeros((len(batch), 0, 1, 1), dtype=torch.float32)
    return {
        "state": states,
        "instruction": instructions,
        "action": actions,
        "image": image_batch,
    }


class BehaviorCloningPolicy(nn.Module):
    """Simple multimodal behavior-cloning policy."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        instruction_dim: int,
        hidden_dim: int,
        image_channels: int = 0,
    ) -> None:
        """Initialize the policy network.

        Args:
            action_dim: Output action dimension.
            state_dim: State feature dimension.
            instruction_dim: Instruction feature dimension.
            hidden_dim: Shared hidden dimension.
            image_channels: Number of image channels, or zero if unused.
        """
        super().__init__()
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.instruction_dim = int(instruction_dim)
        self.image_channels = int(image_channels)
        self.state_encoder = (
            nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            if self.state_dim > 0
            else None
        )
        self.instruction_encoder = nn.Sequential(
            nn.Linear(self.instruction_dim, hidden_dim),
            nn.ReLU(),
        )
        self.image_encoder = (
            nn.Sequential(
                nn.Conv2d(self.image_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, hidden_dim),
                nn.ReLU(),
            )
            if self.image_channels > 0
            else None
        )
        fused_dim = hidden_dim
        if self.state_encoder is not None:
            fused_dim += hidden_dim
        if self.image_encoder is not None:
            fused_dim += hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        instruction: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Predict actions for a batch of observations.

        Args:
            state: State tensor of shape ``(B, S)``.
            instruction: Hashed instruction tensor of shape ``(B, D)``.
            image: Image tensor of shape ``(B, C, H, W)`` or empty channels.

        Returns:
            Action tensor of shape ``(B, A)``.
        """
        features = [self.instruction_encoder(instruction)]
        if self.state_encoder is not None:
            features.append(self.state_encoder(state))
        if self.image_encoder is not None and image.shape[1] > 0:
            features.append(self.image_encoder(image))
        fused = torch.cat(features, dim=-1)
        return self.head(fused)

    @torch.no_grad()
    def predict(self, observation: dict[str, object]) -> np.ndarray:
        """Run inference on one observation dictionary.

        Args:
            observation: Observation dictionary.

        Returns:
            Predicted action vector.
        """
        device = next(self.parameters()).device
        state_value = observation.get("state")
        state = (
            np.asarray(state_value, dtype=np.float32).reshape(-1)
            if isinstance(state_value, np.ndarray)
            else np.zeros(self.state_dim, dtype=np.float32)
        )
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]))
        image_value = _prepare_image(observation)
        image = (
            image_value
            if image_value is not None
            else np.zeros((self.image_channels, 1, 1), dtype=np.float32)
        )
        instruction = _hash_instruction(
            str(observation.get("instruction", "")),
            self.instruction_dim,
        )
        action = self.forward(
            state=torch.from_numpy(state[None, :]).to(device),
            instruction=torch.from_numpy(instruction[None, :]).to(device),
            image=torch.from_numpy(image[None, :]).to(device),
        )
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def train_behavior_cloning(
    episodes: list[TrajectoryEpisode],
    config: BehaviorCloningConfig | None = None,
    split: str | None = "train",
    checkpoint_path: str | Path | None = None,
) -> tuple[BehaviorCloningPolicy, TrainingSummary]:
    """Train the behavior-cloning baseline on normalized trajectory data.

    Args:
        episodes: Training episodes.
        config: Optional training configuration.
        split: Optional split filter.
        checkpoint_path: Optional output checkpoint path.

    Returns:
        Tuple of trained policy and training summary.
    """
    train_config = config or BehaviorCloningConfig()
    dataset = TrajectoryStepDataset(
        episodes=episodes,
        instruction_dim=train_config.instruction_dim,
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=_collate_batch,
    )
    device = torch.device(train_config.device)
    model = BehaviorCloningPolicy(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        instruction_dim=train_config.instruction_dim,
        hidden_dim=train_config.hidden_dim,
        image_channels=dataset.image_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = nn.MSELoss()

    loss_history: list[float] = []
    model.train()
    for epoch in range(train_config.epochs):
        epoch_losses: list[float] = []
        for batch in dataloader:
            state = batch["state"].to(device)
            instruction = batch["instruction"].to(device)
            image = batch["image"].to(device)
            target = batch["action"].to(device)
            prediction = model(state=state, instruction=instruction, image=image)
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        mean_epoch_loss = float(np.mean(np.asarray(epoch_losses, dtype=np.float32)))
        LOGGER.info("BC epoch %d/%d loss=%.6f", epoch + 1, train_config.epochs, mean_epoch_loss)
        loss_history.append(mean_epoch_loss)

    saved_checkpoint: str | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(train_config),
                "model_spec": {
                    "action_dim": dataset.action_dim,
                    "state_dim": dataset.state_dim,
                    "instruction_dim": train_config.instruction_dim,
                    "hidden_dim": train_config.hidden_dim,
                    "image_channels": dataset.image_channels,
                },
            },
            checkpoint,
        )
        saved_checkpoint = str(checkpoint)

    summary = TrainingSummary(
        n_examples=len(dataset),
        action_dim=dataset.action_dim,
        final_train_loss=loss_history[-1],
        train_loss_history=loss_history,
        checkpoint_path=saved_checkpoint,
    )
    return model, summary


def evaluate_behavior_cloning(
    policy: BehaviorCloningPolicy,
    episodes: list[TrajectoryEpisode],
    split: str | None = None,
) -> dict[str, float]:
    """Compute dataset-level imitation metrics for a trained policy.

    Args:
        policy: Trained policy.
        episodes: Evaluation episodes.
        split: Optional split filter.

    Returns:
        Dictionary containing action MSE and MAE.
    """
    dataset = TrajectoryStepDataset(
        episodes=episodes,
        instruction_dim=policy.instruction_dim,
        split=split,
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for sample in dataset:
        observation = {
            "state": sample["state"].numpy(),
            "instruction": "",
        }
        if sample["image"].numel() > 0:
            image = sample["image"].numpy()
            observation["rgb"] = np.transpose(image[:3], (1, 2, 0))
        predictions.append(policy.predict(observation))
        targets.append(sample["action"].numpy())
    prediction_array = np.stack(predictions).astype(np.float32)
    target_array = np.stack(targets).astype(np.float32)
    return {
        "action_mse": float(np.mean((prediction_array - target_array) ** 2)),
        "action_mae": float(np.mean(np.abs(prediction_array - target_array))),
    }


def load_behavior_cloning_policy(path: str | Path, device: str = "cpu") -> BehaviorCloningPolicy:
    """Load a saved behavior-cloning checkpoint.

    Args:
        path: Checkpoint path.
        device: Torch device to map the checkpoint onto.

    Returns:
        Restored behavior-cloning policy.
    """
    payload = torch.load(Path(path), map_location=device)
    model_spec = cast(dict[str, int], payload["model_spec"])
    model = BehaviorCloningPolicy(
        action_dim=int(model_spec["action_dim"]),
        state_dim=int(model_spec["state_dim"]),
        instruction_dim=int(model_spec["instruction_dim"]),
        hidden_dim=int(model_spec["hidden_dim"]),
        image_channels=int(model_spec["image_channels"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
