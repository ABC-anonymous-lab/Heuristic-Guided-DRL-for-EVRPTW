from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def ppo_clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_eps):
    """Clipped PPO policy loss."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss, ratio


def ppo_value_loss(new_values, returns, returns_mean, returns_std):
    """MSE between predicted values and returns, both standardized with the return mean/std."""
    return F.mse_loss(
        (new_values - returns_mean) / returns_std,
        (returns - returns_mean) / returns_std,
    )


def ppo_update_pooled(
    *,
    model,
    optimizer,
    evaluate_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    returns_mean: torch.Tensor,
    returns_std: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    extra_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
) -> dict:

    optimizer.zero_grad(set_to_none=True)

    new_log_probs, new_values, entropies = evaluate_fn()
    policy_loss, ratio = ppo_clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_eps)
    value_loss = ppo_value_loss(new_values, returns, returns_mean, returns_std)
    entropy = entropies.mean()

    actor_loss = policy_loss - entropy_coef * entropy
    total_loss = actor_loss + value_coef * value_loss

    total_loss.backward()

    extra_val = torch.zeros((), device=new_log_probs.device)
    if extra_loss_fn is not None:
        extra = extra_loss_fn()
        extra.backward()
        extra_val = extra.detach()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()

    return {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "loss": total_loss.detach() + extra_val,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "grad_norm": torch.as_tensor(grad_norm),
    }


def ppo_update_mean_of_means_accumulated(
    *,
    model,
    optimizer,
    B: int,
    evaluate_fn_per_graph: Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    old_log_probs_per_graph: List[torch.Tensor],
    advantages_per_graph: List[torch.Tensor],
    returns_per_graph: List[torch.Tensor],
    returns_mean: torch.Tensor,
    returns_std: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    extra_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
) -> dict:
    """loss_aggregation='mean_of_means' with decode_mode='serial': per-graph forward and
    backward (gradient accumulation), every graph weighted equally.
    """
    optimizer.zero_grad(set_to_none=True)

    device = old_log_probs_per_graph[0].device
    total_policy_loss = torch.zeros((), device=device)
    total_value_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    total_extra = torch.zeros((), device=device)
    all_ratio = []
    all_new_lp = []

    for i in range(B):
        new_log_probs_i, new_values_i, entropies_i = evaluate_fn_per_graph(i)

        policy_loss_i, ratio_i = ppo_clipped_policy_loss(
            new_log_probs_i, old_log_probs_per_graph[i], advantages_per_graph[i], clip_eps,
        )
        value_loss_i = ppo_value_loss(new_values_i, returns_per_graph[i], returns_mean, returns_std)
        entropy_i = entropies_i.mean()

        graph_loss = policy_loss_i + value_coef * value_loss_i - entropy_coef * entropy_i

        loss_i = graph_loss / B
        loss_i.backward()

        total_policy_loss = total_policy_loss + policy_loss_i.detach() / B
        total_value_loss = total_value_loss + value_loss_i.detach() / B
        total_entropy = total_entropy + entropy_i.detach() / B
        all_ratio.append(ratio_i.detach())
        all_new_lp.append(new_log_probs_i.detach())

    if extra_loss_fn is not None:
        extra_global = extra_loss_fn()
        extra_global.backward()
        total_extra = total_extra + extra_global.detach()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    ratio_cat = torch.cat(all_ratio)
    new_lp_cat = torch.cat(all_new_lp)
    old_lp_cat = torch.cat(old_log_probs_per_graph)

    with torch.no_grad():
        approx_kl = (old_lp_cat - new_lp_cat).mean()
        clip_frac = ((ratio_cat - 1.0).abs() > clip_eps).float().mean()

    return {
        "policy_loss": total_policy_loss,
        "value_loss": total_value_loss,
        "entropy": total_entropy,
        "extra_loss": total_extra,
        "loss": total_policy_loss + value_coef * total_value_loss - entropy_coef * total_entropy + total_extra,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "grad_norm": torch.as_tensor(grad_norm),
    }


def ppo_update_mean_of_means_single_backward(
    *,
    model,
    optimizer,
    B: int,
    evaluate_all_fn: Callable[[], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]],
    old_log_probs_per_graph: List[torch.Tensor],
    advantages_per_graph: List[torch.Tensor],
    returns_per_graph: List[torch.Tensor],
    returns_mean: torch.Tensor,
    returns_std: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    extra_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
) -> dict:
    """loss_aggregation='mean_of_means' with decode_mode='parallel': one batched forward,
    per-graph means, one backward.
    """
    optimizer.zero_grad(set_to_none=True)

    new_lp_list, new_v_list, ent_list = evaluate_all_fn()
    device = old_log_probs_per_graph[0].device

    policy_losses, value_losses, entropy_means = [], [], []
    all_ratio, all_new_lp = [], []

    for i in range(B):
        new_log_probs_i = new_lp_list[i]
        new_values_i = new_v_list[i]
        entropies_i = ent_list[i]

        policy_loss_i, ratio_i = ppo_clipped_policy_loss(
            new_log_probs_i, old_log_probs_per_graph[i], advantages_per_graph[i], clip_eps,
        )
        value_loss_i = ppo_value_loss(new_values_i, returns_per_graph[i], returns_mean, returns_std)
        entropy_i = entropies_i.mean()

        policy_losses.append(policy_loss_i)
        value_losses.append(value_loss_i)
        entropy_means.append(entropy_i)
        all_ratio.append(ratio_i.detach())
        all_new_lp.append(new_log_probs_i.detach())

    total_policy_loss = torch.stack(policy_losses).mean()
    total_value_loss = torch.stack(value_losses).mean()
    total_entropy = torch.stack(entropy_means).mean()

    main_loss = total_policy_loss + value_coef * total_value_loss - entropy_coef * total_entropy

    total_extra = torch.zeros((), device=device)

    # Backward the PPO loss before building the extra (BC) loss, so both graphs are never in memory together
    main_loss.backward()

    if extra_loss_fn is not None:
        extra_global = extra_loss_fn()
        extra_global.backward()
        total_extra = total_extra + extra_global.detach()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    ratio_cat = torch.cat(all_ratio)
    new_lp_cat = torch.cat(all_new_lp)
    old_lp_cat = torch.cat(old_log_probs_per_graph)

    with torch.no_grad():
        approx_kl = (old_lp_cat - new_lp_cat).mean()
        clip_frac = ((ratio_cat - 1.0).abs() > clip_eps).float().mean()

    total_loss_val = (
        total_policy_loss.detach()
        + value_coef * total_value_loss.detach()
        - entropy_coef * total_entropy.detach()
        + total_extra
    )

    return {
        "policy_loss": total_policy_loss.detach(),
        "value_loss": total_value_loss.detach(),
        "entropy": total_entropy.detach(),
        "extra_loss": total_extra,
        "loss": total_loss_val,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "grad_norm": torch.as_tensor(grad_norm),
    }