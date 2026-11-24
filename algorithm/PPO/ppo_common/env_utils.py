from typing import Any, List

import torch
import numpy as np


def make_envs(env_builder, graph_items: List[Any]):
    """One env per graph."""
    return [env_builder(graph_df) for graph_df in graph_items]


def assert_same_num_nodes(obs_list: List[dict]):
    node_counts = [len(o["xy"]) for o in obs_list]
    if len(set(node_counts)) != 1:
        raise ValueError(
            "Batch forward requires same N in one batch, but got node counts "
            f"{node_counts}. Use separate dataloaders/batches for n50, n100, etc."
        )


def stack_masks(envs, active_indices, device):
    masks = [envs[i].get_action_mask() for i in active_indices]
    return torch.as_tensor(np.stack(masks), device=device, dtype=torch.bool)


def flatten_rollout_tensors(tensor_lists: List[List[torch.Tensor]], device) -> torch.Tensor:
    parts = [torch.stack(xs).view(-1) for xs in tensor_lists if len(xs) > 0]
    if not parts:
        return torch.empty(0, device=device)
    return torch.cat(parts, dim=0)


def encode_static_batch_for_obs_list(model, obs_list, device, requires_grad: bool = False):
    batch_inst = model.obs_list_to_instance_batch(obs_list, device=device)
    if requires_grad:
        return model.encode_static_batch(batch_inst)
    with torch.no_grad():
        return model.encode_static_batch(batch_inst)
