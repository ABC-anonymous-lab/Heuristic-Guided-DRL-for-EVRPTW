import random
from typing import List, Optional
from dataclasses import dataclass, field
from collections import deque, namedtuple

import numpy as np


Transition = namedtuple(
    "Transition",
    [
        "obs",
        "action",
        "reward",
        "next_obs",
        "done",
        "mask",
        "next_mask",
        "last_idx",
        "next_last_idx",
        "source",
        "episode_id",
    ],
    defaults=["policy", -1],
)


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        mask,
        next_mask,
        last_idx,
        next_last_idx,
        source="policy",
        episode_id=-1,
    ):
        self.buffer.append(
            Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                mask=np.array(mask, copy=True),
                next_mask=np.array(next_mask, copy=True),
                last_idx=int(last_idx),
                next_last_idx=int(next_last_idx),
                source=str(source),
                episode_id=int(episode_id),
            )
        )

    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self.buffer))
        if batch_size == 0:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


@dataclass
class DDQNRolloutResult:

    actions: list
    rewards: list
    infos: list
    served_count: float
    total_reward: float
    total_distance: float
    in_time_window: float
    positive_battery: float
    done: bool
    vehicles_used: int
    node_sequence: List[int] = field(default_factory=list)
    vehicle_distances: Optional[List[float]] = None


__all__ = [
    "Transition",
    "ReplayBuffer",
    "DDQNRolloutResult",
]
