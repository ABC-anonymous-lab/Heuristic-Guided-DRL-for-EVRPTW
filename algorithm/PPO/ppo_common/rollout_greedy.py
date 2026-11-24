from typing import Any, List, Tuple

import torch
import numpy as np

from .rollout_types import TERMINATION_REASONS, PPORolloutResult
from .env_utils import assert_same_num_nodes, encode_static_batch_for_obs_list, make_envs, stack_masks


@torch.no_grad()
def rollout_greedy_serial(model, env, device) -> PPORolloutResult:
    """Greedy rollout of one env."""
    was_training = model.training
    model.eval()
    try:
        obs = env.reset()
        init_inst = model.obs_to_instance(obs, device=device)
        h_static, _, _ = model.encode_static(init_inst)

        done = False
        actions: List[int] = []
        rewards: List[float] = []
        infos: List[Any] = []
        in_time_window = 0
        positive_battery = 0

        while not done:
            mask = env.get_action_mask()
            inst = model.obs_to_instance(obs, device=device)
            last_idx = obs["current_node"]

            probs, _, _, _ = model.forward_step(
                h_static=h_static,
                obs_inst=inst,
                last_node_idx=last_idx,
                mask_np=mask,
            )
            a = int(torch.argmax(probs).item())
            obs, reward, done, info = env.step(a)

            actions.append(a)
            rewards.append(float(reward))
            infos.append(info)
            in_time_window += int(info["step_info"].in_time_window)
            positive_battery += int(info["step_info"].positive_battery)

        num_steps = len(actions)
        final_reason = infos[-1].get("termination_reason") if infos else None
        if final_reason not in TERMINATION_REASONS:
            final_reason = "none"

        return PPORolloutResult(
            actions=actions,
            rewards=rewards,
            infos=infos,
            served_count=env.served_rate,
            total_reward=float(np.sum(rewards)) if rewards else 0.0,
            total_distance=float(env.total_distance),
            in_time_window=float(in_time_window / num_steps if num_steps > 0 else 0.0),
            positive_battery=float(positive_battery / num_steps if num_steps > 0 else 0.0),
            done=bool(done),
            termination_reason=final_reason,
            vehicle_distances=getattr(env, "vehicle_distances", None),
        )
    finally:
        model.train() if was_training else model.eval()


@torch.no_grad()
def rollout_greedy_parallel(
    model, env_builder, graph_items: List[Tuple[Any, bool, Any]], device,
) -> List[PPORolloutResult]:
    was_training = model.training
    model.eval()
    try:
        envs = make_envs(env_builder, graph_items)
        obs_list = [env.reset() for env in envs]
        assert_same_num_nodes(obs_list)

        h_static_batch = encode_static_batch_for_obs_list(model, obs_list, device, requires_grad=False)

        B = len(envs)
        done_flags = [False] * B
        actions = [[] for _ in range(B)]
        rewards = [[] for _ in range(B)]
        infos = [[] for _ in range(B)]
        in_time_window = [0 for _ in range(B)]
        positive_battery = [0 for _ in range(B)]

        while not all(done_flags):
            active = [i for i in range(B) if not done_flags[i]]
            active_obs = [obs_list[i] for i in active]

            batch_inst = model.obs_list_to_instance_batch(active_obs, device=device)
            active_h_static = h_static_batch[torch.as_tensor(active, device=device, dtype=torch.long)]
            last_node_indices = torch.as_tensor(
                [int(obs_list[i]["current_node"]) for i in active],
                device=device,
                dtype=torch.long,
            )
            mask_batch = stack_masks(envs, active, device)

            probs, _, _, _ = model.forward_step_batch(
                h_static_batch=active_h_static,
                batch_inst=batch_inst,
                last_node_indices=last_node_indices,
                mask_batch=mask_batch,
            )
            action_batch = torch.argmax(probs, dim=-1)

            for local_j, env_idx in enumerate(active):
                a = int(action_batch[local_j].item())
                next_obs, reward, done, info = envs[env_idx].step(a)
                obs_list[env_idx] = next_obs
                done_flags[env_idx] = bool(done)

                actions[env_idx].append(a)
                rewards[env_idx].append(float(reward))
                infos[env_idx].append(info)
                in_time_window[env_idx] += int(info["step_info"].in_time_window)
                positive_battery[env_idx] += int(info["step_info"].positive_battery)

        results = []
        for i, env in enumerate(envs):
            num_steps = len(actions[i])
            final_reason = infos[i][-1].get("termination_reason") if infos[i] else None
            if final_reason not in TERMINATION_REASONS:
                final_reason = "none"
            results.append(
                PPORolloutResult(
                    actions=actions[i],
                    rewards=rewards[i],
                    infos=infos[i],
                    served_count=env.served_rate,
                    total_reward=float(np.sum(rewards[i])) if rewards[i] else 0.0,
                    total_distance=float(env.total_distance),
                    in_time_window=float((in_time_window[i] / num_steps) if num_steps > 0 else 0.0),
                    positive_battery=float((positive_battery[i] / num_steps) if num_steps > 0 else 0.0),
                    done=bool(done_flags[i]),
                    termination_reason=final_reason,
                    vehicle_distances=getattr(env, "vehicle_distances", None),
                )
            )
        return results
    finally:
        model.train() if was_training else model.eval()