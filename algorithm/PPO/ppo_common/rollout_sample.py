from typing import Any, Callable, List, Tuple

import torch
import numpy as np

from .env_utils import (
    assert_same_num_nodes,
    encode_static_batch_for_obs_list,
    make_envs,
    stack_masks,
)
from .rollout_types import TERMINATION_REASONS, PPORolloutSampleResult


def _build_sample_result(
    actions, traj_log_probs, rewards, infos, values, dones, entropies,
    in_time_window, positive_battery, env, done,
) -> PPORolloutSampleResult:

    num_steps = len(actions)
    final_reason = infos[-1].get("termination_reason") if infos else None
    if final_reason not in TERMINATION_REASONS:
        final_reason = "none"

    return PPORolloutSampleResult(
        actions=actions,
        log_probs=traj_log_probs,
        rewards=rewards,
        infos=infos,
        values=values,
        dones=dones,
        entropies=entropies,
        served_count=env.served_rate,
        total_reward=float(np.sum(rewards)) if rewards else 0.0,
        total_distance=float(env.total_distance),
        in_time_window=float(in_time_window / num_steps if num_steps > 0 else 0.0),
        positive_battery=float(positive_battery / num_steps if num_steps > 0 else 0.0),
        done=bool(done),
        termination_reason=final_reason,
    )


def _rollout_sample_single(model, env, init_obs, h_static, device, select_action_fn):

    obs = init_obs
    done = False

    actions: List[int] = []
    traj_log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    infos: List[Any] = []
    values: List[float] = []
    dones: List[bool] = []
    entropies: List[float] = []
    in_time_window = 0
    positive_battery = 0

    while not done:
        mask = env.get_action_mask()
        inst = model.obs_to_instance(obs, device=device)
        last_idx = obs["current_node"]

        probs, all_log_probs, value, logits = model.forward_step(
            h_static=h_static,
            obs_inst=inst,
            last_node_idx=last_idx,
            mask_np=mask,
        )

        action, log_prob, entropy = select_action_fn(probs)
        a = int(action.item())
        obs, reward, done, info = env.step(a)

        actions.append(a)
        traj_log_probs.append(log_prob)
        rewards.append(float(reward))
        infos.append(info)
        values.append(float(value.squeeze().item()))
        dones.append(bool(done))
        entropies.append(float(entropy.item()))

        in_time_window += int(info["step_info"].in_time_window)
        positive_battery += int(info["step_info"].positive_battery)

    return _build_sample_result(
        actions, traj_log_probs, rewards, infos, values, dones, entropies,
        in_time_window, positive_battery, env, done,
    )


@torch.no_grad()
def rollout_sample_serial(
    model,
    env_builder,
    graph_items: List[Tuple[Any, bool, Any]],
    device,
    select_action_fn: Callable,
) -> List[PPORolloutSampleResult]:

    """decode_mode='serial': encode all graphs once, then run each episode on its own."""
    envs = make_envs(env_builder, graph_items)
    obs_list = [env.reset() for env in envs]
    assert_same_num_nodes(obs_list)

    h_static_batch = encode_static_batch_for_obs_list(model, obs_list, device, requires_grad=False)

    rollouts = []
    for i, env in enumerate(envs):
        rollouts.append(
            _rollout_sample_single(
                model=model,
                env=env,
                init_obs=obs_list[i],
                h_static=h_static_batch[i],
                device=device,
                select_action_fn=select_action_fn,
            )
        )
    return rollouts


@torch.no_grad()
def rollout_sample_parallel(
    model,
    env_builder,
    graph_items: List[Tuple[Any, bool, Any]],
    device,
) -> List[PPORolloutSampleResult]:
    """decode_mode='parallel': all unfinished episodes advance together, one batched forward per step."""
    envs = make_envs(env_builder, graph_items)
    obs_list = [env.reset() for env in envs]
    assert_same_num_nodes(obs_list)

    h_static_batch = encode_static_batch_for_obs_list(model, obs_list, device, requires_grad=False)

    B = len(envs)
    done_flags = [False] * B

    actions = [[] for _ in range(B)]
    traj_log_probs = [[] for _ in range(B)]
    rewards = [[] for _ in range(B)]
    infos = [[] for _ in range(B)]
    values = [[] for _ in range(B)]
    dones = [[] for _ in range(B)]
    entropies = [[] for _ in range(B)]
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

        probs, _, value_batch, _ = model.forward_step_batch(
            h_static_batch=active_h_static,
            batch_inst=batch_inst,
            last_node_indices=last_node_indices,
            mask_batch=mask_batch,
        )

        dist = torch.distributions.Categorical(probs=probs)
        action_batch = dist.sample()
        log_prob_batch = dist.log_prob(action_batch)
        entropy_batch = dist.entropy()

        for local_j, env_idx in enumerate(active):
            a = int(action_batch[local_j].item())
            next_obs, reward, done, info = envs[env_idx].step(a)

            obs_list[env_idx] = next_obs
            done_flags[env_idx] = bool(done)

            actions[env_idx].append(a)
            traj_log_probs[env_idx].append(log_prob_batch[local_j])
            rewards[env_idx].append(float(reward))
            infos[env_idx].append(info)
            values[env_idx].append(float(value_batch[local_j].squeeze().item()))
            dones[env_idx].append(bool(done))
            entropies[env_idx].append(float(entropy_batch[local_j].item()))

            in_time_window[env_idx] += int(info["step_info"].in_time_window)
            positive_battery[env_idx] += int(info["step_info"].positive_battery)

    rollouts = []
    for i, env in enumerate(envs):
        rollouts.append(
            _build_sample_result(
                actions[i], traj_log_probs[i], rewards[i], infos[i], values[i],
                dones[i], entropies[i], in_time_window[i], positive_battery[i],
                env, done_flags[i],
            )
        )
    return rollouts