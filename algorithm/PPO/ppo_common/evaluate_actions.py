from typing import Any, List

import torch

from .env_utils import assert_same_num_nodes, flatten_rollout_tensors, make_envs, stack_masks


def evaluate_actions_serial_single(
    model, env_builder, graph_df, actions, device,
):
    """One graph, step by step."""
    env = env_builder(graph_df)
    obs = env.reset()

    init_inst = model.obs_to_instance(obs, device=device)
    h_static, _, _ = model.encode_static(init_inst)

    log_probs = []
    values = []
    entropies = []
    done = False

    for a in actions:
        if done:
            break

        mask = env.get_action_mask()
        inst = model.obs_to_instance(obs, device=device)
        last_idx = obs["current_node"]

        log_prob, entropy, value = model.evaluate_action(
            h_static=h_static,
            obs_inst=inst,
            last_node_idx=last_idx,
            mask_np=mask,
            action=a,
        )

        log_probs.append(log_prob)
        values.append(value.squeeze())
        entropies.append(entropy)

        obs, _, done, _ = env.step(int(a))

    log_probs_t = torch.stack(log_probs)
    values_t = torch.stack(values).view(-1)
    entropies_t = torch.stack(entropies)
    return log_probs_t, values_t, entropies_t


def evaluate_actions_serial_batch(
    model, env_builder, graph_items: List[Any], actions_batch: List[List[int]], device,
):
    """Several graphs, one after the other."""
    new_lp_list, new_v_list, ent_list = [], [], []
    for i, graph_df in enumerate(graph_items):
        lp, v, ent = evaluate_actions_serial_single(
            model=model,
            env_builder=env_builder,
            graph_df=graph_df,
            actions=actions_batch[i],
            device=device,
        )
        new_lp_list.append(lp)
        new_v_list.append(v)
        ent_list.append(ent)
    return new_lp_list, new_v_list, ent_list


def evaluate_actions_parallel(
    model,
    env_builder,
    graph_items: List[Any],
    actions_batch: List[List[int]],
    device,
    temperature: float = 1.0,
    return_segments: bool = False,
):
    """Several graphs with batched forward passes."""
    envs = make_envs(env_builder, graph_items)
    obs_list = [env.reset() for env in envs]
    assert_same_num_nodes(obs_list)

    batch_inst = model.obs_list_to_instance_batch(obs_list, device=device)
    h_static_batch = model.encode_static_batch(batch_inst)

    B = len(envs)
    done_flags = [False] * B
    max_T = max((len(a) for a in actions_batch), default=0)

    log_probs = [[] for _ in range(B)]
    values = [[] for _ in range(B)]
    entropies = [[] for _ in range(B)]

    for t in range(max_T):
        active = [i for i in range(B) if (not done_flags[i]) and t < len(actions_batch[i])]
        if not active:
            continue

        active_obs = [obs_list[i] for i in active]
        step_batch_inst = model.obs_list_to_instance_batch(active_obs, device=device)
        active_h_static = h_static_batch[torch.as_tensor(active, device=device, dtype=torch.long)]
        last_node_indices = torch.as_tensor(
            [int(obs_list[i]["current_node"]) for i in active],
            device=device,
            dtype=torch.long,
        )
        mask_batch = stack_masks(envs, active, device)

        probs, _, value_batch, _ = model.forward_step_batch(
            h_static_batch=active_h_static,
            batch_inst=step_batch_inst,
            last_node_indices=last_node_indices,
            mask_batch=mask_batch,
        )

        if temperature != 1.0:
            temperature = max(float(temperature), 1e-8)
            probs = torch.pow(probs.clamp_min(1e-12), 1.0 / temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        action_tensor = torch.as_tensor(
            [int(actions_batch[i][t]) for i in active],
            device=device,
            dtype=torch.long,
        )
        dist = torch.distributions.Categorical(probs=probs)
        log_prob_batch = dist.log_prob(action_tensor)
        entropy_batch = dist.entropy()

        for local_j, env_idx in enumerate(active):
            a = int(action_tensor[local_j].item())

            log_probs[env_idx].append(log_prob_batch[local_j])
            values[env_idx].append(value_batch[local_j].squeeze())
            entropies[env_idx].append(entropy_batch[local_j])

            next_obs, _, done, _ = envs[env_idx].step(a)
            obs_list[env_idx] = next_obs
            done_flags[env_idx] = bool(done)

    flat_log_probs = flatten_rollout_tensors(log_probs, device)
    flat_values = flatten_rollout_tensors(values, device)
    flat_entropies = flatten_rollout_tensors(entropies, device)

    if return_segments:
        log_probs_segs = [torch.stack(xs) if len(xs) > 0 else torch.empty(0, device=device) for xs in log_probs]
        values_segs = [torch.stack(xs) if len(xs) > 0 else torch.empty(0, device=device) for xs in values]
        entropies_segs = [torch.stack(xs) if len(xs) > 0 else torch.empty(0, device=device) for xs in entropies]
        return flat_log_probs, flat_values, flat_entropies, log_probs_segs, values_segs, entropies_segs
    return flat_log_probs, flat_values, flat_entropies