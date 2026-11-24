import torch
import numpy as np
from typing import Any, Callable, List, Optional

from algorithm.base_trainer import BaseTrainer
from .ppo_common import (
    compute_gae,
    normalize_advantages,
    make_envs,
    rollout_sample_serial,
    rollout_sample_parallel,
    rollout_greedy_serial,
    rollout_greedy_parallel,
    evaluate_actions_serial_single,
    evaluate_actions_serial_batch,
    evaluate_actions_parallel,
    ppo_update_pooled,
    ppo_update_mean_of_means_accumulated,
    ppo_update_mean_of_means_single_backward,
    extract_vehicle_distance_stats,
    summarize_eval_stats,
    termination_reason_ratios,
    PPORolloutResult,
    PPORolloutSampleResult,
)


class PPOTrainer(BaseTrainer):
    """PPO with GAE. decode_mode: 'serial' | 'parallel'; loss_aggregation: 'pooled' | 'mean_of_means'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_cfg = self.cfg["train"]
        self.gamma = train_cfg["gamma"]
        self.gae_lambda = train_cfg["gae_lambda"]
        self.clip_eps = train_cfg["clip_eps"]
        self.value_coef = train_cfg["value_coef"]
        self.entropy_coef = train_cfg["entropy_coef"]
        self.ppo_epochs = train_cfg["ppo_epochs"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.normalize_advantage = train_cfg["normalize_advantage"]

        self.decode_mode = train_cfg["decode_mode"]
        self.loss_aggregation = train_cfg["loss_aggregation"]

        if self.decode_mode not in ("serial", "parallel"):
            raise ValueError(f"Unknown decode_mode: {self.decode_mode!r}")
        if self.loss_aggregation not in ("mean_of_means", "pooled"):
            raise ValueError(f"Unknown loss_aggregation: {self.loss_aggregation!r}")


    def _compute_gae(self, rewards, values, dones, last_value=0.0):
        return compute_gae(rewards, values, dones, self.gamma, self.gae_lambda, last_value)


    def _extract_graph_items(self, batch) -> List[Any]:
        return list(batch["graphs"])


    def _select_action_fn(self, probs):
        return self.select_action(probs, mode="sample", return_entropy=True)


    def _rollout_sample_batch(self, graph_items) -> List[PPORolloutSampleResult]:
        if self.decode_mode == "parallel":
            return rollout_sample_parallel(self.model, self.env_builder, graph_items, self.device)
        return rollout_sample_serial(
            self.model, self.env_builder, graph_items, self.device,
            select_action_fn=self._select_action_fn,
        )


    def rollout_greedy(self, env) -> PPORolloutResult:
        return rollout_greedy_serial(self.model, env, self.device)


    def rollout_greedy_batch(self, graph_items) -> List[PPORolloutResult]:
        """Greedy rollouts for a list of graphs, decoded in parallel."""
        return rollout_greedy_parallel(self.model, self.env_builder, graph_items, self.device)


    def _rollout_greedy_for_batch(self, graph_items) -> List[PPORolloutResult]:
        """Greedy rollouts for a list of graphs, serial or parallel according to decode_mode."""
        if self.decode_mode == "parallel":
            return self.rollout_greedy_batch(graph_items)
        envs = make_envs(self.env_builder, graph_items)
        return [self.rollout_greedy(env) for env in envs]


    def evaluate_actions_on_rollout_batch(
        self, graph_items, actions_batch, return_segments: bool = False, temperature: float = 1.0,
    ):
        """Recompute log-probs, values and entropies of given action sequences (with gradients)."""
        return evaluate_actions_parallel(
            self.model, self.env_builder, graph_items, actions_batch, self.device,
            temperature=temperature, return_segments=return_segments,
        )


    def _normalize_advantages(self, advantages_per_graph_raw):
        """Normalize the advantages over the whole batch (if normalize_advantage)."""
        if not self.normalize_advantage:
            flat = torch.cat(advantages_per_graph_raw, dim=0)
            return flat, advantages_per_graph_raw
        return normalize_advantages(advantages_per_graph_raw)


    def _after_rollout_hook(self, graph_items, rollouts) -> dict:
        """Hook called after the sampled rollouts; returns a context dict passed to the other hooks."""
        return {}


    def _extra_loss_fn(
        self, *, graph_items, actions_batch, after_rollout_ctx,
    ) -> Optional[Callable[[], torch.Tensor]]:
        return None


    def _extra_stats(self, *, graph_items, rollouts, after_rollout_ctx) -> dict:
        return {}


    def _phase_timer(self, phase, cuda_sync=False):
        return self.logger.timer(phase, cuda_sync=cuda_sync)


    def train_step(self, batch):
        """Sample rollouts, compute GAE advantages, run ppo_epochs PPO updates and a greedy evaluation."""
        self.model.train()

        graph_items = self._extract_graph_items(batch)
        B = len(graph_items)
        graphs = graph_items

        with self._phase_timer("rollout", cuda_sync=True):
            rollouts = self._rollout_sample_batch(graph_items)
        actions_batch = [r.actions for r in rollouts]

        advantages_per_graph_raw = []
        returns_per_graph = []
        for rollout in rollouts:
            adv, ret = self._compute_gae(
                rewards=rollout.rewards, values=rollout.values, dones=rollout.dones, last_value=0.0,
            )
            advantages_per_graph_raw.append(torch.as_tensor(adv, dtype=torch.float32, device=self.device))
            returns_per_graph.append(torch.as_tensor(ret, dtype=torch.float32, device=self.device))

        raw_advantages = torch.cat(advantages_per_graph_raw, dim=0)
        returns = torch.cat(returns_per_graph, dim=0)

        advantages, advantages_per_graph = self._normalize_advantages(advantages_per_graph_raw)

        returns_mean = returns.mean()
        if returns.numel() > 1:
            returns_std = returns.std(unbiased=False).clamp_min(1e-8)
        else:
            returns_std = torch.ones((), device=self.device, dtype=returns.dtype)

        old_log_probs_per_graph = [torch.stack(r.log_probs).view(-1).detach() for r in rollouts]
        old_log_probs = torch.cat(old_log_probs_per_graph, dim=0)

        # HG-PPO: OR-Tools repair of the sampled routes
        after_rollout_ctx = self._after_rollout_hook(graph_items, rollouts)

        last = {}
        with self._phase_timer("ppo_update", cuda_sync=True):
          for _ in range(self.ppo_epochs):
            last = self._ppo_update_one_epoch(
                B=B,
                graph_items=graph_items,
                actions_batch=actions_batch,
                old_log_probs=old_log_probs,
                old_log_probs_per_graph=old_log_probs_per_graph,
                advantages=advantages,
                advantages_per_graph=advantages_per_graph,
                returns=returns,
                returns_per_graph=returns_per_graph,
                returns_mean=returns_mean,
                returns_std=returns_std,
                after_rollout_ctx=after_rollout_ctx,
            )

        with self._phase_timer("greedy_eval", cuda_sync=True):
            greedy_results = self._rollout_greedy_for_batch(graph_items)

        stats = self._build_stats(
            rollouts=rollouts, greedy_results=greedy_results, last=last,
            raw_advantages=raw_advantages, B=B,
        )
        stats.update(self._extra_stats(graph_items=graph_items, rollouts=rollouts, after_rollout_ctx=after_rollout_ctx))

        viz_payload = {
            "graph_df": graphs[0],
            "greedy_result": greedy_results[0],
        }

        return stats, viz_payload


    def _ppo_update_one_epoch(
        self, *, B, graph_items, actions_batch, old_log_probs, old_log_probs_per_graph,
        advantages, advantages_per_graph, returns, returns_per_graph, returns_mean, returns_std,
        after_rollout_ctx,
    ):
        self.model.train()

        common_kwargs = dict(
            model=self.model,
            optimizer=self.optimizer,
            clip_eps=self.clip_eps,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm,
        )

        extra_loss_fn = self._extra_loss_fn(
            graph_items=graph_items, actions_batch=actions_batch, after_rollout_ctx=after_rollout_ctx,
        )

        if self.loss_aggregation == "pooled":
            def evaluate_fn():
                if self.decode_mode == "parallel":
                    return evaluate_actions_parallel(
                        self.model, self.env_builder, graph_items, actions_batch, self.device,
                    )
                lp_list, v_list, ent_list = evaluate_actions_serial_batch(
                    self.model, self.env_builder, graph_items, actions_batch, self.device,
                )
                return torch.cat(lp_list), torch.cat(v_list), torch.cat(ent_list)

            return ppo_update_pooled(
                evaluate_fn=evaluate_fn,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
                returns_mean=returns_mean,
                returns_std=returns_std,
                extra_loss_fn=extra_loss_fn,
                **common_kwargs,
            )

        if self.decode_mode == "serial":
            def evaluate_fn_per_graph(i):
                return evaluate_actions_serial_single(
                    self.model, self.env_builder, graph_items[i], actions_batch[i], self.device,
                )

            return ppo_update_mean_of_means_accumulated(
                B=B,
                evaluate_fn_per_graph=evaluate_fn_per_graph,
                old_log_probs_per_graph=old_log_probs_per_graph,
                advantages_per_graph=advantages_per_graph,
                returns_per_graph=returns_per_graph,
                returns_mean=returns_mean,
                returns_std=returns_std,
                extra_loss_fn=extra_loss_fn,
                **common_kwargs,
            )


        def evaluate_all_fn():
            _, _, _, lp_segs, v_segs, ent_segs = evaluate_actions_parallel(
                self.model, self.env_builder, graph_items, actions_batch, self.device,
                return_segments=True,
            )
            return lp_segs, v_segs, ent_segs

        return ppo_update_mean_of_means_single_backward(
            B=B,
            evaluate_all_fn=evaluate_all_fn,
            old_log_probs_per_graph=old_log_probs_per_graph,
            advantages_per_graph=advantages_per_graph,
            returns_per_graph=returns_per_graph,
            returns_mean=returns_mean,
            returns_std=returns_std,
            extra_loss_fn=extra_loss_fn,
            **common_kwargs,
        )


    def _build_stats(self, *, rollouts, greedy_results, last, raw_advantages, B):
        def avg(key, items):
            return float(np.mean([getattr(r, key) for r in items])) if items else 0.0

        stats = {
            "loss": float(last["loss"].item()),
            "policy_loss": float(last["policy_loss"].item()),
            "value_loss": float(last["value_loss"].item()),
            "value_loss/policy_loss": float(
                last["value_loss"].item() / (abs(last["policy_loss"].item()) + 1e-8)
            ),
            "entropy": float(last["entropy"].item()),
            "approx_kl": float(last["approx_kl"].item()),
            "clip_frac": float(last["clip_frac"].item()),
            "grad_norm": float(last["grad_norm"].item()),
            "lr": float(self._get_current_lr()),

            "batch_size": float(B),
            "reward": avg("total_reward", rollouts),
            "len": float(np.mean([len(r.actions) for r in rollouts])) if rollouts else 0.0,
            "served_count": avg("served_count", rollouts),
            "in_time_window_ratio": avg("in_time_window", rollouts),
            "positive_battery_ratio": avg("positive_battery", rollouts),
            "total_distance": avg("total_distance", rollouts),
            "advantage": float(raw_advantages.mean().item()),

            "reward_greedy": avg("total_reward", greedy_results),
            "len_greedy": float(np.mean([len(r.actions) for r in greedy_results])) if greedy_results else 0.0,
            "served_count_greedy": avg("served_count", greedy_results),
            "in_time_window_ratio_greedy": avg("in_time_window", greedy_results),
            "positive_battery_ratio_greedy": avg("positive_battery", greedy_results),
            "total_distance_greedy": avg("total_distance", greedy_results),
        }

        for reason, ratio in termination_reason_ratios(rollouts).items():
            stats[f"term_reason/{reason}"] = ratio

        return stats


    @torch.no_grad()
    def evaluate(self, loader, return_details=False):
        """Greedy evaluation on all graphs of a loader."""
        self.model.eval()
        all_stats = []

        for batch_idx, batch in enumerate(loader):
            graph_items = self._extract_graph_items(batch)
            results = self._rollout_greedy_for_batch(graph_items)

            for local_i, (graph_df, result) in enumerate(zip(graph_items, results)):
                graph_id = graph_df["graph_id"].iloc[0]

                num_vehicles, vd_mean, vd_std, vd_min, vd_max = extract_vehicle_distance_stats(
                    result.vehicle_distances
                )

                all_stats.append({
                    "graph_id": graph_id,
                    "batch_idx": int(batch_idx),
                    "reward": float(result.total_reward),
                    "len": int(len(result.actions)),
                    "served_count": float(result.served_count),
                    "in_time_window_ratio": float(result.in_time_window),
                    "positive_battery_ratio": float(result.positive_battery),
                    "total_distance": float(result.total_distance),
                    "done": bool(result.done),
                    "num_vehicles": num_vehicles,
                    "vehicle_distance_mean": vd_mean,
                    "vehicle_distance_std": vd_std,
                    "vehicle_distance_min": vd_min,
                    "vehicle_distance_max": vd_max,
                })

        result = summarize_eval_stats(all_stats, return_details=return_details)
        self.model.train()
        return result


__all__ = ["PPOTrainer", "PPORolloutResult", "PPORolloutSampleResult"]