from dataclasses import dataclass
from typing import List, Any, Optional

import torch


TERMINATION_REASONS = (
    "all_served",
    "out_of_vehicles",
    "max_steps",
    "no_feasible_action",
    "none",
)


@dataclass
class PPORolloutResult:

    actions: List[int]
    rewards: List[float]
    infos: List[Any]
    served_count: Any
    total_reward: float
    total_distance: float
    in_time_window: float
    positive_battery: float
    done: bool
    termination_reason: str = "none"
    vehicle_distances: Optional[List[float]] = None


@dataclass
class PPORolloutSampleResult:

    actions: List[int]
    log_probs: List[torch.Tensor]
    rewards: List[float]
    infos: List[Any]
    values: List[float]
    dones: List[bool]
    entropies: List[float]
    served_count: Any
    total_reward: float
    total_distance: float
    in_time_window: float
    positive_battery: float
    done: bool
    termination_reason: str = "none"


def termination_reason_ratios(rollouts) -> dict:
    total = len(rollouts)
    ratios = {reason: 0.0 for reason in TERMINATION_REASONS}
    if total == 0:
        return ratios

    for r in rollouts:
        reason = getattr(r, "termination_reason", None)
        if reason not in TERMINATION_REASONS:
            reason = "none"
        ratios[reason] += 1.0

    for reason in ratios:
        ratios[reason] /= total

    return ratios