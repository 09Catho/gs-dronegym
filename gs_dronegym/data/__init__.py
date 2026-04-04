"""Shared data contracts and dataset utilities for GS-DroneGym v0.2."""

from gs_dronegym.data.dataset import (
    TransitionRecord,
    iter_transitions,
    load_common_dataset,
    load_dataset,
    replay_episode,
    save_dataset,
    summarize_dataset,
)
from gs_dronegym.data.schema import (
    ActionSpec,
    BenchmarkReport,
    ObservationSpec,
    TaskSpec,
    TrajectoryEpisode,
    TrajectoryStep,
    infer_observation_spec,
)

__all__ = [
    "ActionSpec",
    "BenchmarkReport",
    "ObservationSpec",
    "TaskSpec",
    "TrajectoryEpisode",
    "TrajectoryStep",
    "TransitionRecord",
    "infer_observation_spec",
    "iter_transitions",
    "load_common_dataset",
    "load_dataset",
    "replay_episode",
    "save_dataset",
    "summarize_dataset",
]
