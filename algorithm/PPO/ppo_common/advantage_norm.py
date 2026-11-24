from typing import List, Tuple

import torch


def normalize_advantages(
    advantages_per_graph: List[torch.Tensor], eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    advantages = torch.cat(advantages_per_graph, dim=0)
    if advantages.numel() > 1:
        std = advantages.std(unbiased=False)
        advantages = (advantages - advantages.mean()) / std.clamp_min(eps)

    per_graph = []
    offset = 0
    for a in advantages_per_graph:
        length = a.numel()
        per_graph.append(advantages[offset:offset + length])
        offset += length
    return advantages, per_graph
