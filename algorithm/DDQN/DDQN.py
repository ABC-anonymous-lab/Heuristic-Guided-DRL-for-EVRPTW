import copy
import random
from contextlib import nullcontext

import torch
import numpy as np
import torch.nn.functional as F

from algorithm.base_trainer import BaseTrainer
from .ddqn_common import ReplayBuffer, DDQNRolloutResult
from algorithm.eval_stats import summarize_eval_stats, extract_vehicle_distance_stats



class DDQNTrainer(BaseTrainer):
    """Double DQN with a target network and epsilon-greedy exploration.
    decode_mode ('serial' | 'parallel') controls the loss computation, rollout_mode
    ('serial' | 'parallel') the epsilon-greedy rollouts, loss_aggregation ('pooled' |
    'mean_of_means') how the TD losses of a sampled batch are averaged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ddqn_cfg = self.cfg["train"]
        self._default_config(ddqn_cfg)
        self._build_buffers()

        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.env_step_count = 0
        self.update_count = 0
        self._episode_counter = 0


    def _default_config(self, ddqn_cfg):
        self.gamma = float(ddqn_cfg.get("gamma", 0.99))
        self.batch_size = int(ddqn_cfg.get("batch_size_for_learn", 32))
        self.buffer_capacity = int(ddqn_cfg.get("buffer_capacity", 50000))
        self.min_buffer_size = int(ddqn_cfg.get("min_buffer_size", 1000))
        self.grad_steps_per_batch = int(ddqn_cfg.get("grad_steps_per_batch", 1))
        self.target_update_freq = int(ddqn_cfg.get("target_update_freq", 2000))
        self.grad_clip = float(ddqn_cfg.get("grad_clip", 1.0))

        self.epsilon_start = float(ddqn_cfg.get("epsilon_start", 1.0))
        self.epsilon_end = float(ddqn_cfg.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = int(ddqn_cfg.get("epsilon_decay_steps", 50000))
        self.epsilon_limit_for_best_model = float(
            ddqn_cfg.get("epsilon_limit_for_best_model", 0.5)
        )

        self.decode_mode = ddqn_cfg.get("decode_mode", "parallel")
        if self.decode_mode not in ("serial", "parallel"):
            raise ValueError(f"decode_mode must be 'serial' or 'parallel', got {self.decode_mode!r}")

        self.loss_aggregation = ddqn_cfg.get("loss_aggregation", "pooled")
        if self.loss_aggregation not in ("pooled", "mean_of_means"):
            raise ValueError(f"loss_aggregation must be 'pooled' or 'mean_of_means', got {self.loss_aggregation!r}")


        self.rollout_mode = ddqn_cfg.get("rollout_mode", "parallel")
        if self.rollout_mode not in ("serial", "parallel"):
            raise ValueError(f"rollout_mode must be 'serial' or 'parallel', got {self.rollout_mode!r}")


    def _build_buffers(self):
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)


    def _clone_obs(self, obs):
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                out[k] = v.copy()
            elif torch.is_tensor(v):
                out[k] = v.detach().cpu().clone()
            else:
                out[k] = copy.deepcopy(v)
        return out


    def _get_epsilon(self):
        ratio = min(1.0, self.env_step_count / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)


    def _masked_argmax(self, q_values: torch.Tensor, mask_np):
        mask = torch.as_tensor(mask_np, device=q_values.device, dtype=torch.bool)
        neg_inf = torch.finfo(q_values.dtype).min
        q_masked = q_values.masked_fill(~mask, neg_inf)
        return int(torch.argmax(q_masked).item())


    def _select_action_from_q(self, q_values: torch.Tensor, mask_np, epsilon: float):
        mask = np.asarray(mask_np).astype(bool)
        valid_actions = np.flatnonzero(mask)

        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")

        if random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        return self._masked_argmax(q_values, mask_np)


    def _sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()


    def _new_episode_id(self):
        self._episode_counter += 1
        return self._episode_counter


    def rollout(
        self,
        env,
        mode="greedy",
        epsilon=None,
        store_transition=False,
        buffer=None,
        source="policy",
    ):
        if store_transition and buffer is None:
            buffer = self.replay_buffer

        episode_id = self._new_episode_id() if store_transition else -1

        obs = env.reset()
        init_inst = self.model.obs_to_instance(obs, device=self.device)
        h_static, _, _ = self.model.encode_static(init_inst)

        done = False
        actions = []
        rewards = []
        infos = []

        in_time_window = 0
        positive_battery = 0
        node_sequence = [int(obs["current_node"])]

        if epsilon is None:
            epsilon = self._get_epsilon()

        ctx = torch.no_grad() if mode in ("greedy", "sample") else nullcontext()

        with ctx:
            while not done:
                mask = env.get_action_mask()
                inst = self.model.obs_to_instance(obs, device=self.device)
                last_idx = int(obs["current_node"])

                q_values = self.model.forward_step(
                    h_static=h_static,
                    obs_inst=inst,
                    last_node_idx=last_idx,
                    mask_np=None,
                )

                if mode == "greedy":
                    action = self._masked_argmax(q_values, mask)
                elif mode == "sample":
                    action = self._select_action_from_q(q_values, mask, epsilon=epsilon)
                else:
                    raise ValueError(f"Unknown rollout mode: {mode}")

                obs_next, reward, done, info = env.step(action)

                actions.append(int(action))
                rewards.append(float(reward))
                infos.append(info)
                node_sequence.append(int(obs_next["current_node"]))

                in_time_window += int(info["step_info"].in_time_window)
                positive_battery += int(info["step_info"].positive_battery)

                if store_transition:
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

                    buffer.push(
                        obs=self._clone_obs(obs),
                        action=int(action),
                        reward=float(reward),
                        next_obs=self._clone_obs(obs_next),
                        done=bool(done),
                        mask=mask,
                        next_mask=next_mask,
                        last_idx=last_idx,
                        next_last_idx=next_last_idx,
                        source=source,
                        episode_id=episode_id,
                    )

                obs = obs_next

                if source == "policy" and store_transition and mode == "sample":
                    self.env_step_count += 1
                    if self.env_step_count % self.target_update_freq == 0:
                        self._sync_target()

        num_steps = len(actions)

        return DDQNRolloutResult(
            actions=actions,
            rewards=rewards,
            infos=infos,
            served_count=float(env.served_rate),
            total_reward=float(np.sum(rewards)),
            total_distance=float(env.total_distance),
            in_time_window=float(in_time_window / num_steps) if num_steps > 0 else 0.0,
            positive_battery=float(positive_battery / num_steps) if num_steps > 0 else 0.0,
            done=bool(done),
            vehicles_used=env.vehicle_used,
            node_sequence=node_sequence,
            vehicle_distances=list(env.vehicle_distances),
        )


    @torch.no_grad()
    def rollout_greedy(self, env):
        return self.rollout(
            env,
            mode="greedy",
            store_transition=False,
            source="eval",
        )


    def _rollout_greedy_for_batch(self, graph_items):
        results = []
        for graph_df in graph_items:
            env_eval = self.env_builder(graph_df)
            results.append(self.rollout_greedy(env_eval))
        return results


    def rollout_sample_parallel(
        self,
        env_builder,
        graph_items,
        epsilon=None,
        store_transition=True,
        buffer=None,
        source="policy",
    ):
        """rollout_mode='parallel': epsilon-greedy rollouts of B graphs advancing together,
        with one batched forward pass per step.
        """
        if store_transition and buffer is None:
            buffer = self.replay_buffer

        B = len(graph_items)
        device = self.device

        if epsilon is None:
            epsilon = self._get_epsilon()

        envs = [env_builder(graph_df) for graph_df in graph_items]
        obs_list = [env.reset() for env in envs]

        episode_ids = (
            [self._new_episode_id() for _ in range(B)] if store_transition else [-1] * B
        )

        init_inst_batch = self.model.obs_list_to_instance_batch(obs_list, device=device)
        h_static_batch = self.model.encode_static_batch(init_inst_batch)

        actions_per_graph       = [[] for _ in range(B)]
        rewards_per_graph        = [[] for _ in range(B)]
        infos_per_graph          = [[] for _ in range(B)]
        node_sequence_per_graph  = [[int(obs_list[b]["current_node"])] for b in range(B)]
        in_time_window_count     = [0 for _ in range(B)]
        positive_battery_count   = [0 for _ in range(B)]
        done_flags               = [False for _ in range(B)]

        active = torch.ones(B, dtype=torch.bool, device=device)
        with torch.no_grad():
            while active.any():
                mask_list = [envs[b].get_action_mask() for b in range(B)]
                mask_batch = torch.as_tensor(
                    np.stack(mask_list), dtype=torch.bool, device=device
                )

                last_node_indices = torch.as_tensor(
                    [int(obs_list[b]["current_node"]) for b in range(B)],
                    dtype=torch.long, device=device,
                )

                batch_inst = self.model.obs_list_to_instance_batch(obs_list, device=device)

                q_values = self.model.forward_step_batch(
                    h_static_batch=h_static_batch,
                    batch_inst=batch_inst,
                    last_node_indices=last_node_indices,
                    mask_batch=mask_batch,
                )

                actions = [None] * B
                for b in range(B):
                    if not bool(active[b].item()):
                        continue
                    mask_b = np.asarray(mask_list[b]).astype(bool)
                    if mask_b.sum() == 0:
                        raise RuntimeError(
                            "[DDQN rollout_sample_parallel] active env has no valid actions. "
                            f"b={b}, "
                            f"current_node={obs_list[b].get('current_node', None)}, "
                            f"steps={len(actions_per_graph[b])}, "
                            f"served_rate={getattr(envs[b], 'served_rate', None)}, "
                            f"vehicle_used={getattr(envs[b], 'vehicle_used', None)}, "
                            f"done_flag={done_flags[b]}"
                        )
                    a = self._select_action_from_q(q_values[b], mask_b, epsilon=epsilon)
                    actions[b] = int(a)

                next_obs_list = list(obs_list)
                num_policy_steps_this_timestep = 0
                for b in range(B):
                    if not active[b]:
                        continue

                    action = actions[b]
                    obs_b = obs_list[b]
                    last_idx = int(obs_b["current_node"])
                    mask_b = mask_list[b]

                    obs_next, reward, done, info = envs[b].step(action)
                    num_policy_steps_this_timestep += 1

                    actions_per_graph[b].append(int(action))
                    rewards_per_graph[b].append(float(reward))
                    infos_per_graph[b].append(info)
                    node_sequence_per_graph[b].append(int(obs_next["current_node"]))

                    in_time_window_count[b]   += int(info["step_info"].in_time_window)
                    positive_battery_count[b] += int(info["step_info"].positive_battery)

                    if store_transition:
                        if done:
                            next_mask = mask_b
                            next_last_idx = (
                                int(obs_next["current_node"])
                                if "current_node" in obs_next
                                else last_idx
                            )
                        else:
                            next_mask = envs[b].get_action_mask()
                            next_last_idx = int(obs_next["current_node"])

                        buffer.push(
                            obs=self._clone_obs(obs_b),
                            action=int(action),
                            reward=float(reward),
                            next_obs=self._clone_obs(obs_next),
                            done=bool(done),
                            mask=mask_b,
                            next_mask=next_mask,
                            last_idx=last_idx,
                            next_last_idx=next_last_idx,
                            source=source,
                            episode_id=episode_ids[b],
                        )

                    next_obs_list[b] = obs_next

                    if done:
                        done_flags[b] = True
                        active[b] = False

                obs_list = next_obs_list

                if source == "policy" and store_transition:
                    old_step = self.env_step_count
                    self.env_step_count += num_policy_steps_this_timestep

                    if old_step // self.target_update_freq < self.env_step_count // self.target_update_freq:
                        self._sync_target()

        results = []
        for b in range(B):
            num_steps = len(actions_per_graph[b])
            results.append(
                DDQNRolloutResult(
                    actions=actions_per_graph[b],
                    rewards=rewards_per_graph[b],
                    infos=infos_per_graph[b],
                    served_count=float(envs[b].served_rate),
                    total_reward=float(np.sum(rewards_per_graph[b])) if num_steps > 0 else 0.0,
                    total_distance=float(envs[b].total_distance),
                    in_time_window=float(in_time_window_count[b] / num_steps) if num_steps > 0 else 0.0,
                    positive_battery=float(positive_battery_count[b] / num_steps) if num_steps > 0 else 0.0,
                    done=bool(done_flags[b]),
                    vehicles_used=envs[b].vehicle_used,
                    node_sequence=node_sequence_per_graph[b],
                    vehicle_distances=list(envs[b].vehicle_distances),
                )
            )

        return results

    def _compute_td_target(self, tr):
        reward_t = torch.tensor(tr.reward, device=self.device, dtype=torch.float32)

        if tr.done:
            return reward_t

        next_obs_inst_online = self.model.obs_to_instance(tr.next_obs, device=self.device)
        next_h_static_online, _, _ = self.model.encode_static(next_obs_inst_online)
        next_q_online = self.model.forward_step(
            h_static=next_h_static_online,
            obs_inst=next_obs_inst_online,
            last_node_idx=tr.next_last_idx,
            mask_np=None,
        )
        next_action = self._masked_argmax(next_q_online, tr.next_mask)

        next_obs_inst_target = self.target_model.obs_to_instance(tr.next_obs, device=self.device)
        next_h_static_target, _, _ = self.target_model.encode_static(next_obs_inst_target)
        next_q_target = self.target_model.forward_step(
            h_static=next_h_static_target,
            obs_inst=next_obs_inst_target,
            last_node_idx=tr.next_last_idx,
            mask_np=None,
        )

        return reward_t + self.gamma * next_q_target[next_action]

    def _aggregate_elementwise_loss(self, elementwise_loss, episode_ids):
        """Average per-transition losses ('pooled') or per-episode means ('mean_of_means')."""
        if self.loss_aggregation == "pooled":
            return elementwise_loss.mean()

        if self.loss_aggregation == "mean_of_means":
            by_episode = {}
            for i, eid in enumerate(episode_ids):
                by_episode.setdefault(int(eid), []).append(i)

            per_episode_means = []
            for idxs in by_episode.values():
                idx_t = torch.tensor(idxs, device=elementwise_loss.device, dtype=torch.long)
                per_episode_means.append(elementwise_loss[idx_t].mean())

            return torch.stack(per_episode_means).mean()

        raise ValueError(f"Unknown loss_aggregation: {self.loss_aggregation!r}")


    def _compute_batch_loss(self, sampled):
        """TD loss (+ extra loss of the hooks) of a sampled batch, serial or parallel."""
        if self.decode_mode == "parallel":
            return self._compute_batch_loss_parallel(sampled)
        if self.decode_mode == "serial":
            return self._compute_batch_loss_serial(sampled)
        raise ValueError(f"Unknown decode_mode: {self.decode_mode!r}")


    def _compute_batch_loss_serial(self, sampled):
        q_pred_list = []
        q_target_list = []
        episode_ids = []

        for tr in sampled:

            obs_inst = self.model.obs_to_instance(tr.obs, device=self.device)
            h_static, _, _ = self.model.encode_static(obs_inst)

            q_values = self.model.forward_step(
                h_static=h_static,
                obs_inst=obs_inst,
                last_node_idx=tr.last_idx,
                mask_np=None,
            )

            q_sa = q_values[tr.action]

            with torch.no_grad():
                target = self._compute_td_target(tr)

            q_pred_list.append(q_sa)
            q_target_list.append(target)
            episode_ids.append(tr.episode_id)

        q_pred = torch.stack(q_pred_list)
        q_target = torch.stack(q_target_list)

        elementwise_loss = F.smooth_l1_loss(q_pred, q_target, reduction="none")
        td_loss = self._aggregate_elementwise_loss(elementwise_loss, episode_ids)

        extra_only_sampled = self._sample_extra_loss_only_transitions()
        extra_terms = []

        for tr in extra_only_sampled:

            obs_inst = self.model.obs_to_instance(tr.obs, device=self.device)
            h_static, _, _ = self.model.encode_static(obs_inst)
            q_values = self.model.forward_step(
                h_static=h_static,
                obs_inst=obs_inst,
                last_node_idx=tr.last_idx,
                mask_np=None,
            )

            extra = self._extra_per_transition_loss(tr, q_values)
            if extra is not None:
                extra_terms.append(extra)

        extra_loss, extra_stats = self._combine_extra_losses(extra_terms)
        total_loss = td_loss + extra_loss

        loss_dict = {
            "loss": total_loss,
            "td_loss": td_loss.detach(),
            "q_mean": q_pred.mean().detach(),
            "target_mean": q_target.mean().detach(),
        }
        loss_dict.update(extra_stats)
        return loss_dict


    def _td_loss_parallel(self, sampled):
        transitions = sampled
        device = self.device

        node_counts = {len(tr.mask) for tr in transitions}
        if len(node_counts) > 1:
            raise ValueError(
                f"decode_mode='parallel' needs all graphs in a TD batch to have the same size N, "
                f"got {sorted(node_counts)}. Use decode_mode='serial' for mixed sizes."
            )

        action = torch.tensor([tr.action for tr in transitions], device=device, dtype=torch.long)
        reward = torch.tensor([tr.reward for tr in transitions], device=device, dtype=torch.float32)
        done = torch.tensor([tr.done for tr in transitions], device=device, dtype=torch.bool)
        last_idx = torch.tensor([tr.last_idx for tr in transitions], device=device, dtype=torch.long)
        next_last_idx = torch.tensor([tr.next_last_idx for tr in transitions], device=device, dtype=torch.long)
        next_mask = torch.tensor(
            np.stack([tr.next_mask for tr in transitions]), device=device, dtype=torch.bool
        )
        cur_mask_td = torch.tensor(
            np.stack([tr.mask for tr in transitions]), device=device, dtype=torch.bool,
        )
        action_is_legal = cur_mask_td.gather(1, action.view(-1, 1)).squeeze(1)
        if not action_is_legal.all():
            bad = (~action_is_legal).nonzero(as_tuple=False).view(-1)
            raise RuntimeError(
                f"Illegal action in the replay buffer, bad indices={bad[:10].tolist()}"
            )

        episode_ids = [tr.episode_id for tr in transitions]

        batch_inst_cur = self.model.obs_list_to_instance_batch(
            [tr.obs for tr in transitions], device=device
        )
        h_cur = self.model.encode_static_batch(batch_inst_cur)
        q_values_cur = self.model.forward_step_batch(
            h_cur, batch_inst_cur, last_idx, mask_batch=cur_mask_td
        )
        q_pred = q_values_cur.gather(1, action.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            batch_inst_next = self.model.obs_list_to_instance_batch(
                [tr.next_obs for tr in transitions], device=device
            )
            h_next_online = self.model.encode_static_batch(batch_inst_next)
            next_q_online = self.model.forward_step_batch(
                h_next_online, batch_inst_next, next_last_idx, mask_batch=next_mask
            )
            next_action = next_q_online.argmax(dim=1)

            h_next_target = self.target_model.encode_static_batch(batch_inst_next)
            next_q_target = self.target_model.forward_step_batch(
                h_next_target, batch_inst_next, next_last_idx, mask_batch=next_mask
            )
            next_q_for_action = next_q_target.gather(1, next_action.view(-1, 1)).squeeze(1)

            target = reward + self.gamma * next_q_for_action * (~done).float()

        elementwise_loss = F.smooth_l1_loss(q_pred, target, reduction="none")
        td_loss = self._aggregate_elementwise_loss(elementwise_loss, episode_ids)

        td_stats = {
            "td_loss": td_loss.detach(),
            "q_mean": q_pred.mean().detach(),
            "target_mean": target.mean().detach(),
        }
        return td_loss, td_stats


    def _bc_loss_parallel(self, extra_only_sampled):
        if not extra_only_sampled:
            return None, {}

        extra_only_transitions = extra_only_sampled
        device = self.device

        node_counts = {len(tr.mask) for tr in extra_only_transitions}
        if len(node_counts) > 1:
            raise ValueError(
                f"decode_mode='parallel' needs all graphs in a BC batch to have the same size N, "
                f"got {sorted(node_counts)}. Use decode_mode='serial' for mixed sizes."
            )

        extra_last_idx = torch.tensor(
            [tr.last_idx for tr in extra_only_transitions], device=device, dtype=torch.long
        )
        extra_mask = torch.tensor(
            np.stack([tr.mask for tr in extra_only_transitions]), device=device, dtype=torch.bool
        )

        batch_inst_extra = self.model.obs_list_to_instance_batch(
            [tr.obs for tr in extra_only_transitions], device=device
        )
        h_extra = self.model.encode_static_batch(batch_inst_extra)
        q_values_extra = self.model.forward_step_batch(
            h_extra, batch_inst_extra, extra_last_idx, mask_batch=extra_mask
        )

        extra_terms = []
        for i, tr in enumerate(extra_only_transitions):
            extra = self._extra_per_transition_loss(tr, q_values_extra[i])
            if extra is not None:
                extra_terms.append(extra)

        return self._combine_extra_losses(extra_terms)


    def _compute_batch_loss_parallel(self, sampled):
        td_loss, td_stats = self._td_loss_parallel(sampled)

        extra_only_sampled = self._sample_extra_loss_only_transitions()
        extra_loss, extra_stats = self._bc_loss_parallel(extra_only_sampled)

        total_loss = td_loss if extra_loss is None else td_loss + extra_loss

        loss_dict = {
            "loss": total_loss,
            "td_loss": td_stats["td_loss"],
            "q_mean": td_stats["q_mean"],
            "target_mean": td_stats["target_mean"],
        }
        loss_dict.update(extra_stats)
        return loss_dict


    def _optimize_on_batch(self, sampled):
        loss_dict = self._compute_batch_loss(sampled)
        loss = loss_dict["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        loss_dict["loss"] = float(loss.detach().item())
        return loss_dict


    def _extra_per_transition_loss(self, tr, q_values):
        return None


    def _combine_extra_losses(self, extra_terms):
        return torch.tensor(0.0, device=self.device), {}


    def _total_buffer_size(self):
        return len(self.replay_buffer)


    def _sample_transitions_for_update(self, batch_size):
        return self.replay_buffer.sample(batch_size)


    def _sample_extra_loss_only_transitions(self):
        return []


    def _after_rollout_hook(self, graph_items, sample_results):
        return None


    def _extra_stats(self, extra_accum, n_updates):
        return {}


    def _extract_graph_items(self, batch):
        return list(batch["graphs"])


    def _phase_timer(self, phase, cuda_sync=False):
        return self.logger.timer(phase, cuda_sync=cuda_sync)


    def train_step(self, batch):
        """Epsilon-greedy rollouts on the batch (filling the replay buffer), then grad_steps_per_batch
        gradient steps on transitions sampled from the buffer, target-network sync and a greedy evaluation.
        """
        self.model.train()

        graph_items = self._extract_graph_items(batch)
        B = len(graph_items)
        epsilon = self._get_epsilon()

        with self._phase_timer("rollout", cuda_sync=True):
          if self.rollout_mode == "parallel":
              sample_results = self.rollout_sample_parallel(
                  self.env_builder,
                  graph_items,
                  epsilon=epsilon,
                  store_transition=True,
                  buffer=self.replay_buffer,
                  source="policy",
              )
          else:
              sample_results = []
              for graph_df in graph_items:
                  env = self.env_builder(graph_df)
                  sample_result = self.rollout(
                      env,
                      mode="sample",
                      epsilon=epsilon,
                      store_transition=True,
                      buffer=self.replay_buffer,
                      source="policy",
                  )
                  sample_results.append(sample_result)

        after_rollout_stats = self._after_rollout_hook(graph_items, sample_results) or {}

        loss_val = 0.0
        q_mean = 0.0
        target_mean = 0.0
        n_updates = 0
        extra_accum = {}
        q_mean_total = 0.0
        target_mean_total = 0.0
        q_mean_avg = 0.0
        target_mean_avg = 0.0
        td_loss_total = 0.0
        td_loss_avg = 0.0

        if self._total_buffer_size() >= self.min_buffer_size:
          with self._phase_timer("td_update", cuda_sync=True):
              self.model.train()
              # Fixed number of gradient steps per batch, on transitions sampled from the replay buffer
              for _ in range(self.grad_steps_per_batch):
                  sampled = self._sample_transitions_for_update(self.batch_size)
                  if not sampled:
                      continue

                  loss_dict = self._optimize_on_batch(sampled)
                  td_loss = loss_dict["td_loss"]
                  td_loss_total += td_loss

                  loss_val += float(loss_dict["loss"])
                  q_mean = float(loss_dict["q_mean"].item())
                  q_mean_total += q_mean
                  target_mean = float(loss_dict["target_mean"].item())
                  target_mean_total += target_mean

                  for k, v in loss_dict.items():
                      if k in ("loss", "q_mean", "target_mean"):
                          continue
                      v_float = float(v.item()) if torch.is_tensor(v) else float(v)
                      extra_accum[k] = extra_accum.get(k, 0.0) + v_float

                  n_updates += 1
                  self.update_count += 1

              loss_val /= max(n_updates, 1)
              q_mean_avg = q_mean_total / max(n_updates, 1)
              target_mean_avg = target_mean_total / max(n_updates, 1)
              td_loss_avg = td_loss_total / max(n_updates, 1)

        greedy_results = []

        with self._phase_timer("greedy_eval", cuda_sync=True):
            with self.model.eval_context():
                greedy_results = self._rollout_greedy_for_batch(graph_items)

        def avg(key, items):
            return float(np.mean([getattr(r, key) for r in items])) if items else 0.0

        stats = {
            "reward": avg("total_reward", sample_results),
            "len": float(np.mean([len(r.actions) for r in sample_results])) if sample_results else 0.0,
            "served_count": avg("served_count", sample_results),
            "in_time_window_ratio": avg("in_time_window", sample_results),
            "positive_battery_ratio": avg("positive_battery", sample_results),
            "total_distance": avg("total_distance", sample_results),
            "done": float(np.mean([float(r.done) for r in sample_results])) if sample_results else 0.0,

            "reward_greedy": avg("total_reward", greedy_results),
            "len_greedy": float(np.mean([len(r.actions) for r in greedy_results])) if greedy_results else 0.0,
            "served_count_greedy": avg("served_count", greedy_results),
            "in_time_window_ratio_greedy": avg("in_time_window", greedy_results),
            "positive_battery_ratio_greedy": avg("positive_battery", greedy_results),
            "total_distance_greedy": avg("total_distance", greedy_results),
            "done_greedy": float(np.mean([float(r.done) for r in greedy_results])) if greedy_results else 0.0,
            "vehicles_used": avg("vehicles_used", greedy_results),

            "loss": loss_val,
            "td_loss": td_loss_avg,
            "q_mean": q_mean_avg,
            "target_mean": target_mean_avg,
            "epsilon": float(epsilon),
            "buffer_size": len(self.replay_buffer),
            "env_step_count": int(self.env_step_count),
            "update_count": int(self.update_count),
            "batch_size": float(B),
        }

        stats.update(self._extra_stats(extra_accum, n_updates))
        stats.update(after_rollout_stats)

        viz_payload = {
            "graph_df": graph_items[0],
            "greedy_result": greedy_results[0],
        }

        return stats, viz_payload


    @torch.no_grad()
    def evaluate(self, loader, return_details=False):
        self.model.eval()
        all_stats = []

        for batch_idx, batch in enumerate(loader):
            graph_items = self._extract_graph_items(batch)
            results = self._rollout_greedy_for_batch(graph_items)

            for local_i, (graph_df, result) in enumerate(zip(graph_items, results)):
                graph_id = graph_df["graph_id"].iloc[0]

                _, vd_mean, vd_std, _, _ = extract_vehicle_distance_stats(
                    getattr(result, "vehicle_distances", None)
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
                    "num_vehicles": float(result.vehicles_used),
                    "vehicle_distance_mean": vd_mean,
                    "vehicle_distance_std": vd_std,
                })

        result = summarize_eval_stats(all_stats, return_details=return_details)
        self.model.train()
        return result


__all__ = ["DDQNTrainer"]