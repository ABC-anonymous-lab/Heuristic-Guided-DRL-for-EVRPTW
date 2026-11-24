import random

import torch
import numpy as np
import torch.nn.functional as F

from .DDQN import DDQNTrainer
from .ddqn_common import ReplayBuffer
from algorithm.OR_Tools_GLS_guidance.subprocess_wrapper import IsolatedOrtoolsRepair


class HGDDQNTrainer(DDQNTrainer):
    """DDQNTrainer guided by OR-Tools: the sampled routes are repaired by OR-Tools GLS and the
    repaired trajectories are stored in gls_buffer. The TD loss uses only the policy replay
    buffer; a BC loss (softmax over the masked Q-values) is computed on a batch from gls_buffer."""

    def _default_config(self, ddqn_cfg):
        super()._default_config(ddqn_cfg)

        self.gls_buffer_capacity = int(ddqn_cfg.get("gls_buffer_capacity", 10000))
        self.use_gls_guided = bool(ddqn_cfg.get("use_gls_guided", False))
        self.bc_batch_size = int(ddqn_cfg.get("bc_batch_size", 32))

        repair_fn_file_name = ddqn_cfg["repair_file_name"]
        self.ortools_repair_fn = IsolatedOrtoolsRepair(
            repair_module_name=repair_fn_file_name,
            env_builder_factory_module_name="utils.init_model",
            env_builder_factory_func_name="build_env_builder",
            env_builder_cfg=self.cfg,
            timeout_buffer_s=10.0,
            worker_recycle_every=int(ddqn_cfg.get("ortools_worker_recycle_every", 100)),
            num_workers=int(ddqn_cfg.get("ortools_num_workers", 4)),
        )
        self.ortools_repair_time_limit = int(self.cfg["train"]["ortools_repair_time_limit"])
        self.use_imitation_loss = bool(ddqn_cfg.get("use_imitation_loss", True))
        self.bc_max_weight = float(ddqn_cfg.get("bc_max_weight", 0.5))
        self.bc_min_weight = float(ddqn_cfg.get("bc_min_weight", 0.1))
        self.bc_temperature = float(ddqn_cfg.get("bc_temperature", 1.0))

        self.bc_loss_aggregation = ddqn_cfg.get("bc_loss_aggregation", "pooled")
        if self.bc_loss_aggregation not in ("pooled", "mean_of_means"):
            raise ValueError(
                f"bc_loss_aggregation must be 'pooled' or 'mean_of_means', "
                f"got {self.bc_loss_aggregation!r}"
            )

        self.gls_repair_every_n_graphs = int(ddqn_cfg.get("gls_repair_every_n_graphs", 1))
        if self.gls_repair_every_n_graphs < 1:
            raise ValueError(
                f"gls_repair_every_n_graphs must be >= 1, got {self.gls_repair_every_n_graphs!r}"
            )
        self.gls_repair_prob = 1.0 / self.gls_repair_every_n_graphs


    def _build_buffers(self):
        super()._build_buffers()
        self.gls_buffer = ReplayBuffer(self.gls_buffer_capacity)


    def _total_buffer_size(self):
        return len(self.replay_buffer) + len(self.gls_buffer)


    def _sample_transitions_for_update(self, batch_size):
        if len(self.replay_buffer) == 0:
            return None
        return self.replay_buffer.sample(batch_size)


    def _sample_extra_loss_only_transitions(self):
        if not (self.use_imitation_loss and self.bc_batch_size > 0):
            return []
        if len(self.gls_buffer) == 0:
            return []
        n = min(self.bc_batch_size, len(self.gls_buffer))
        return self.gls_buffer.sample(n)


    def _get_bc_weight(self):
        """BC weight, decreasing linearly from bc_max_weight to bc_min_weight over training."""
        if self.total_rollout_steps <= 0:
            return self.bc_max_weight
        portion = max(0.0, min(1.0, self.global_step / self.total_rollout_steps))
        return self.bc_max_weight - (self.bc_max_weight - self.bc_min_weight) * portion


    def _optimize_on_batch(self, sampled):
        """decode_mode='parallel': backward the TD loss first and then the BC loss (on its own
        batch), so that the two computation graphs are never in memory at the same time.
        The gradients add up, so this equals one backward of td_loss + bc_loss.
        """
        if self.decode_mode != "parallel":
            return super()._optimize_on_batch(sampled)

        self.optimizer.zero_grad()

        td_loss, td_stats = self._td_loss_parallel(sampled)
        td_loss.backward()
        loss_scalar = float(td_loss.detach().item())
        del td_loss

        extra_only_sampled = self._sample_extra_loss_only_transitions()
        extra_loss, extra_stats = self._bc_loss_parallel(extra_only_sampled)
        if extra_loss is not None and extra_loss.requires_grad:
            extra_loss.backward()
            loss_scalar += float(extra_loss.detach().item())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        loss_dict = {
            "loss": loss_scalar,
            "td_loss": td_stats["td_loss"],
            "q_mean": td_stats["q_mean"],
            "target_mean": td_stats["target_mean"],
        }
        loss_dict.update(extra_stats)
        return loss_dict


    def _after_rollout_hook(self, graph_items, sample_results):
        """Repair the sampled routes with OR-Tools and store the repaired trajectories in the gls_buffer."""
        if not self.use_gls_guided:
            return None

        rewards_gls = []
        served_counts_gls = []
        distances_gls = []
        vehicles_used_gls = []
        lens_gls = []
        raw_action_lens = []
        n_skipped = 0

        selected = []
        for graph_df, sample_result in zip(graph_items, sample_results):
            if random.random() >= self.gls_repair_prob:
                n_skipped += 1
                continue
            selected.append((graph_df, sample_result.actions))

        repair_outputs = []
        if selected:
            with self._phase_timer("gls_repair"):
                repair_outputs = self.ortools_repair_fn.repair_batch(
                    selected,
                    time_limit_s=self.ortools_repair_time_limit,
                    balance_coefficient=self.cfg["train"].get("balance_coefficient", 10),
                    vehicle_fixed_cost=self.cfg["train"].get("vehicle_fixed_cost", 100),
                    charge_rate=self.cfg["env"]["charging_rate"],
                    energy_per_km=self.cfg["env"]["energy_rate"],
                )

        for (graph_df, init_actions), (repaired_actions, repair_info) in zip(
            selected, repair_outputs
        ):
            if repaired_actions is None or len(repaired_actions) == 0:
                continue
            if isinstance(repair_info, dict) and str(repair_info.get("status", "")).startswith("subprocess_"):
                continue

            with self._phase_timer("gls_traj_replay"):
                env_for_store = self.env_builder(graph_df)
                repair_result = self._rollout_with_action_sequence(
                    env=env_for_store,
                    action_sequence=repaired_actions,
                    store_buffer=self.gls_buffer,
                    source="gls",
                )
            if repair_result is None:
                continue

            rewards_gls.append(repair_result["total_reward"])
            served_counts_gls.append(repair_result["served_count"])
            distances_gls.append(repair_result["total_distance"])
            vehicles_used_gls.append(repair_result["vehicles_used"])
            lens_gls.append(repair_result["executed_action_len"])
            raw_action_lens.append(repair_result["raw_action_len"])

        if not rewards_gls:
            return None

        return {
            "reward_gls": float(np.mean(rewards_gls)),
            "served_count_gls": float(np.mean(served_counts_gls)),
            "total_distance_gls": float(np.mean(distances_gls)),
            "vehicles_used_gls": float(np.mean(vehicles_used_gls)),
            "len_gls": float(np.mean(lens_gls)),
            "raw_action_lens": float(np.mean(raw_action_lens)),
            "gls_repair_skipped_count": n_skipped,
        }


    def _extra_per_transition_loss(self, tr, q_values):
        """BC loss of one transition: cross-entropy between the masked Q-softmax and the expert action."""
        if not self.use_imitation_loss:
            return None
        if not (0 <= int(tr.action) < len(tr.mask) and bool(tr.mask[tr.action])):
            return None

        a_e = int(tr.action)
        mask = torch.as_tensor(tr.mask, device=q_values.device, dtype=torch.bool)

        log_probs = self._masked_log_probs(q_values, tr.mask, temperature=self.bc_temperature)
        bc_loss = -log_probs[a_e]

        valid_q = q_values[mask]
        adv = (q_values[a_e] - valid_q.mean()).detach()
        return {
            "loss": bc_loss,
            "is_gls": tr.source == "gls",
            "episode_id": tr.episode_id,
            "adv": adv,
        }


    def _combine_extra_losses(self, extra_terms):
        """Weighted BC loss, averaged 'pooled' or 'mean_of_means' (per repaired trajectory)."""
        if len(extra_terms) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {"bc_loss": zero, "n_gls_used_for_bc": 0}

        n_gls_used_for_bc = sum(1 for t in extra_terms if t.get("is_gls"))

        adv_vals = [t["adv"] for t in extra_terms if "adv" in t]
        adv_mean = torch.stack(adv_vals).mean() if adv_vals else torch.tensor(0.0, device=self.device)
        prob_vals = [t["prob_expert"] for t in extra_terms if "prob_expert" in t]
        prob_mean = torch.stack(prob_vals).mean() if prob_vals else torch.tensor(0.0, device=self.device)

        if self.bc_loss_aggregation == "pooled":
            bc_losses = torch.stack([t["loss"] for t in extra_terms])
            bc_loss = bc_losses.mean()

        elif self.bc_loss_aggregation == "mean_of_means":
            by_episode = {}
            for t in extra_terms:
                key = int(t.get("episode_id", -1))
                by_episode.setdefault(key, []).append(t["loss"])
            per_episode_means = [torch.stack(v).mean() for v in by_episode.values()]
            bc_loss = torch.stack(per_episode_means).mean()

        else:
            raise ValueError(f"Unknown bc_loss_aggregation: {self.bc_loss_aggregation!r}")

        return self._get_bc_weight() * bc_loss, {
            "bc_loss": bc_loss.detach(),
            "n_gls_used_for_bc": n_gls_used_for_bc,
            "adv": adv_mean,
            "prob_expert": prob_mean,
        }


    def _extra_stats(self, extra_accum, n_updates):
        n = max(n_updates, 1)
        return {
            "td_loss": extra_accum.get("td_loss", 0.0) / n,
            "bc_loss": extra_accum.get("bc_loss", 0.0) / n,
            "bc_weight": float(self._get_bc_weight()),
            "bc_loss_aggregation": self.bc_loss_aggregation,
            "n_gls_used_for_bc": int(extra_accum.get("n_gls_used_for_bc", 0)),
            "gls_buffer_size": len(self.gls_buffer),
            "total_buffer_size": self._total_buffer_size(),
        }


    def _masked_log_probs(self, q_values: torch.Tensor, mask_np, temperature: float = 1.0):
        mask = torch.as_tensor(mask_np, device=q_values.device, dtype=torch.bool)
        neg_inf = torch.finfo(q_values.dtype).min
        q_masked = q_values.masked_fill(~mask, neg_inf)
        return F.log_softmax(q_masked / max(temperature, 1e-8), dim=-1)


    def _rollout_with_action_sequence(self, env, action_sequence, store_buffer=None, source="gls"):
        episode_id = self._new_episode_id() if store_buffer is not None else -1

        obs = env.reset()
        done = False

        trajectory = []

        actions = []
        rewards = []
        infos = []
        in_time_window = 0
        positive_battery = 0
        node_sequence = [int(obs["current_node"])]

        if len(action_sequence) > 0 and int(action_sequence[0]) == 0:
            action_sequence = action_sequence[1:]
        raw_action_len = len(action_sequence)

        for action in action_sequence:
            if done:
                break

            mask = env.get_action_mask()
            action = int(action)

            last_idx = int(obs["current_node"])
            obs_next, reward, done, info = env.step(action)

            actions.append(action)
            rewards.append(float(reward))
            infos.append(info)
            node_sequence.append(int(obs_next["current_node"]))
            in_time_window += int(info["step_info"].in_time_window)
            positive_battery += int(info["step_info"].positive_battery)

            if store_buffer is not None:
                action_valid_in_mask = (
                    0 <= action < len(mask) and bool(mask[action])
                )
                if action_valid_in_mask:
                    if done:
                        next_mask = mask
                        next_last_idx = (
                            int(obs_next["current_node"])
                            if "current_node" in obs_next
                            else last_idx
                        )
                    else:
                        next_mask = env.get_action_mask()
                        next_last_idx = int(obs_next["current_node"])

                    trajectory.append(dict(
                        obs=self._clone_obs(obs),
                        action=action,
                        reward=float(reward),
                        next_obs=self._clone_obs(obs_next),
                        done=bool(done),
                        mask=mask,
                        next_mask=next_mask,
                        last_idx=last_idx,
                        next_last_idx=next_last_idx,
                        source=source,
                        episode_id=episode_id,
                    ))

            obs = obs_next

        success = done
        if not success:
            return None

        if store_buffer is not None:
            for tr in trajectory:
                store_buffer.push(**tr)

        num_steps = len(actions)

        return {
            "actions": actions,
            "rewards": rewards,
            "infos": infos,
            "served_count": float(env.served_rate),
            "total_reward": float(np.sum(rewards)),
            "total_distance": float(env.total_distance),
            "in_time_window": float(in_time_window / num_steps) if num_steps > 0 else 0.0,
            "positive_battery": float(positive_battery / num_steps) if num_steps > 0 else 0.0,
            "done": bool(done),
            "vehicles_used": env.vehicle_used,
            "node_sequence": node_sequence,
            "vehicle_distances": list(env.vehicle_distances),
            "raw_action_len": raw_action_len,
            "executed_action_len": len(actions),
        }


__all__ = ["HGDDQNTrainer"]