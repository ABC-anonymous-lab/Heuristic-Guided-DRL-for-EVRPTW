from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import numpy as np
from dataclasses import dataclass, field

from algorithm.PN_LSTM.PN_LSTM import PNLSTMTrainer
from algorithm.OR_Tools_GLS_guidance.subprocess_wrapper import IsolatedOrtoolsRepair


@dataclass
class TeacherResult:
    actions: List[int]
    repair_status: str
    hint_valid: bool
    in_time_window: float
    positive_battery: float
    served_count: float
    total_reward: Optional[float] = None
    total_distance: Optional[float] = None
    bc_early_stop_infeasible: bool = False

    n_bc_actions: int = 0

    dropped_nodes: List[int] = field(default_factory=list)


class HGPNLSTMTrainer(PNLSTMTrainer):
    """PNLSTMTrainer plus OR-Tools repair and BC loss."""

    def __init__(self, *args, repair_fn=None, **kwargs):
        super().__init__(*args, **kwargs)

        train_cfg = self.cfg["train"]
        self.bc_max_weight = float(train_cfg.get("bc_max_weight", 0.2))
        self.bc_min_weight = float(train_cfg.get("bc_min_weight", 0.05))
        self.ortools_repair_time_limit = int(train_cfg["ortools_repair_time_limit"])
        self.bc_only_if_teacher_better = bool(train_cfg.get("bc_only_if_teacher_better", True))
        self.bc_better_margin = float(train_cfg.get("bc_better_margin", 0.1))

        self.bc_loss_aggregation = str(
            train_cfg.get("bc_loss_aggregation", "mean_of_means")
        ).lower()

        valid_bc_loss_aggregations = {"mean_of_means", "pooled"}
        if self.bc_loss_aggregation not in valid_bc_loss_aggregations:
            raise ValueError(
                f"Unknown bc_loss_aggregation={self.bc_loss_aggregation}. "
                f"Valid options are: {sorted(valid_bc_loss_aggregations)}"
            )

        if repair_fn is not None:
            self.ortools_repair_fn = repair_fn
        else:

            self.ortools_repair_fn = IsolatedOrtoolsRepair(
                repair_module_name=train_cfg["repair_file_name"],
                env_builder_factory_module_name="utils.init_model",
                env_builder_factory_func_name="build_env_builder",
                env_builder_cfg=self.cfg,
                timeout_buffer_s=10.0,
                worker_recycle_every=int(train_cfg.get("ortools_worker_recycle_every", 100)),
                num_workers=int(train_cfg.get("ortools_num_workers", 4)),
            )


    def _phase_timer(self, phase, cuda_sync=False):
        return self.logger.timer(phase, cuda_sync=cuda_sync)


    def _make_teacher_env(self, graph_df):
        return self._make_env(graph_df)


    def _compute_rl_advantages_from_rewards(
        self,
        sample_results,
        baseline_rewards: List[float],
    ) -> Tuple[List[float], List[float], List[int]]:
        """Batch-normalized advantages from the rewards only (same as PNLSTMTrainer._compute_reinforce_loss)."""
        raw_advantages: List[float] = []
        episode_lens: List[int] = []
        for r, b in zip(sample_results, baseline_rewards):
            raw_advantages.append(float(r.total_reward) - float(b))
            episode_lens.append(len(r.actions))

        advantage_tensor = torch.tensor(
            raw_advantages, device=self.device, dtype=torch.float32
        )

        if advantage_tensor.numel() > 1:
            adv_mean = advantage_tensor.mean()
            adv_std = advantage_tensor.std(unbiased=False)
            advantage_tensor = (advantage_tensor - adv_mean) / (adv_std + 1e-8)

        advantages = [float(a) for a in advantage_tensor]
        return advantages, raw_advantages, episode_lens


    @staticmethod
    def _flush_rl_segment(seg_log_probs: List[torch.Tensor], scale: float):
        if len(seg_log_probs) == 0:
            return []
        seg_sum = torch.stack(seg_log_probs).sum()
        (-scale * seg_sum).backward()
        return []


    def _reinforce_recompute_backward(
        self,
        sample_results,
        advantages: List[float],
        episode_lens: List[int],
        graphs,
        B: int,
        accum_steps: int,
    ):
        rl_loss_value = 0.0
        log_prob_sums: List[float] = []

        for i in range(B):
            actions = list(sample_results[i].actions)
            adv_i = float(advantages[i])
            T_i = int(episode_lens[i])

            if T_i == 0 or len(actions) == 0:
                log_prob_sums.append(0.0)
                continue

            denom = float(max(T_i, 1))

            # Per-graph scale so that the segment gradients add up to the REINFORCE gradient
            scale_i = adv_i / denom / float(B) / float(accum_steps)

            env = self._make_env(graphs[i])
            obs = env.reset()
            lstm_state = None
            done = False
            seg_log_probs: List[torch.Tensor] = []
            graph_log_prob_sum = 0.0

            for a in actions:
                if done:
                    break
                a_int = int(a)
                mask = np.asarray(env.get_action_mask(), dtype=bool)
                if not mask[a_int]:
                    break

                obs_inst = self.model.obs_to_instance(obs, device=self.device)
                encoded, _, _ = self.model.encode_state(obs_inst)
                last_node_idx = int(obs["current_node"])

                out = self.model.forward_step(
                    encoded=encoded,
                    last_node_idx=last_node_idx,
                    mask=mask,
                    lstm_state=lstm_state,
                )
                lstm_state = out["lstm_state"]

                a_tensor = torch.tensor(a_int, device=self.device)
                log_prob = out["log_probs"].gather(0, a_tensor.view(1)).squeeze(0)
                seg_log_probs.append(log_prob)
                graph_log_prob_sum += float(log_prob.detach().item())

                obs, reward, done, step_info = env.step(a_int)

                if a_int == 0 and not done:
                    seg_log_probs = self._flush_rl_segment(seg_log_probs, scale_i)
                    lstm_state = None

            seg_log_probs = self._flush_rl_segment(seg_log_probs, scale_i)

            rl_loss_value += (-adv_i * graph_log_prob_sum / denom) / float(B)
            log_prob_sums.append(graph_log_prob_sum)

        return rl_loss_value, log_prob_sums


    def is_gls_better_for_bc(self, rl_info, gls_info, margin=None):
        if gls_info is None:
            return False
        if margin is None:
            margin = self.bc_better_margin
        get = lambda x, k: x[k] if isinstance(x, dict) else getattr(x, k)
        if get(gls_info, "total_reward") > get(rl_info, "total_reward"):
            return True
        if get(gls_info, "served_count") > get(rl_info, "served_count"):
            return True
        if get(gls_info, "served_count") < get(rl_info, "served_count") - margin:
            return False
        return get(gls_info, "total_distance") < get(rl_info, "total_distance")


    def _replay_teacher_env(self, repaired_actions, info, graph_df):
        repair_status = info.get("status", "unknown") if isinstance(info, dict) else "unknown"
        hint_valid = bool(info.get("hint_valid", False)) if isinstance(info, dict) else False
        repair_failed_fallback = str(repair_status).startswith("subprocess_")
        if (
            repaired_actions is None or len(repaired_actions) == 0
            or repair_failed_fallback
        ):
            return TeacherResult(
                actions=[], repair_status=repair_status, hint_valid=hint_valid,
                total_reward=0.0, total_distance=0.0, in_time_window=0.0,
                positive_battery=0.0, served_count=0.0, n_bc_actions=0,
            )

        repaired_actions = list(repaired_actions)
        if len(repaired_actions) > 0 and int(repaired_actions[0]) == 0:
            repaired_actions = repaired_actions[1:]

        env = self._make_teacher_env(graph_df)
        obs = env.reset()
        done = False
        executed, total_reward = [], 0.0
        itw = pb = 0
        early = False
        import numpy as _np
        for a in repaired_actions:
            if done:
                break
            a_int = int(a)
            mask = _np.asarray(env.get_action_mask(), dtype=bool)
            if not mask[a_int]:
                early = True
                break
            obs, reward, done, step_info = env.step(a_int)
            executed.append(a_int)
            total_reward += float(reward)
            si = step_info.get("step_info") if isinstance(step_info, dict) else None
            if si is not None:
                itw += int(getattr(si, "in_time_window", 0))
                pb += int(getattr(si, "positive_battery", 0))
        n = max(len(executed), 1)
        return TeacherResult(
            actions=executed, repair_status=repair_status, hint_valid=hint_valid,
            total_reward=total_reward,
            total_distance=float(getattr(env, "total_distance", 0.0)),
            in_time_window=itw / n, positive_battery=pb / n,
            served_count=float(getattr(env, "served_rate", 0.0)),
            bc_early_stop_infeasible=early, n_bc_actions=len(executed),
        )


    def _get_bc_weight(self):
        if self.total_rollout_steps <= 0:
            return self.bc_max_weight
        portion = max(0.0, min(1.0, self.global_step / self.total_rollout_steps))
        return self.bc_max_weight - (self.bc_max_weight - self.bc_min_weight) * portion


    @staticmethod
    def _compute_grad_alignment_metrics(named_params, rl_grads, bc_grads):
        group_matchers = {
            "all": lambda name: True,
            "encoder": lambda name: name.startswith("encoder."),
            "lstm": lambda name: name.startswith("decoder.lstm_cell."),
            "attention": lambda name: name.startswith("decoder.attention."),
        }

        metrics = {}
        eps_ratio = 1e-8
        eps_cos = 1e-12

        for group_name, matcher in group_matchers.items():
            rl_sq = 0.0
            bc_sq = 0.0
            dot = 0.0
            n_params = 0

            for (name, _p), g_rl, g_bc in zip(named_params, rl_grads, bc_grads):
                if not matcher(name):
                    continue

                n_params += 1

                if g_rl is not None:
                    rl_sq += float(g_rl.detach().pow(2).sum().item())

                if g_bc is not None:
                    bc_sq += float(g_bc.detach().pow(2).sum().item())

                if g_rl is not None and g_bc is not None:
                    dot += float((g_rl.detach() * g_bc.detach()).sum().item())

            rl_norm = rl_sq ** 0.5
            bc_norm = bc_sq ** 0.5
            ratio = bc_norm / (rl_norm + eps_ratio)

            if rl_norm > eps_cos and bc_norm > eps_cos:
                cosine = dot / (rl_norm * bc_norm)
                cosine = max(-1.0, min(1.0, cosine))
            else:
                cosine = 0.0

            prefix = "" if group_name == "all" else f"{group_name}_"
            metrics[f"{prefix}rl_grad_norm"] = float(rl_norm)
            metrics[f"{prefix}bc_grad_norm"] = float(bc_norm)
            metrics[f"{prefix}bc_rl_grad_ratio"] = float(ratio)
            metrics[f"{prefix}grad_cosine_rl_bc"] = float(cosine)
            metrics[f"{prefix}grad_param_tensors"] = float(n_params)

        return metrics


    def _teacher_forcing_bc(
        self,
        repaired_actions,
        info,
        graph_df,
    ):
        repair_status = info.get("status", "unknown")
        hint_valid = bool(info.get("hint_valid", False))

        repair_failed_fallback = str(repair_status).startswith("subprocess_")
        if repair_failed_fallback:
            empty_result = TeacherResult(
                actions=[],
                repair_status=repair_status,
                hint_valid=hint_valid,
                total_reward=0.0,
                total_distance=0.0,
                in_time_window=0.0,
                positive_battery=0.0,
                served_count=0.0,
                n_bc_actions=0,
            )
            return torch.tensor(0.0, device=self.device), empty_result

        env = self._make_env(graph_df)
        obs = env.reset()
        lstm_state = None
        done = False

        teacher_log_probs: List[torch.Tensor] = []
        executed_actions: List[int] = []
        total_reward = 0.0
        in_time_window = 0
        positive_battery = 0
        bc_early_stop_infeasible = False

        repaired_actions = list(repaired_actions)
        if len(repaired_actions) > 0 and int(repaired_actions[0]) == 0:
            repaired_actions = repaired_actions[1:]

        for a in repaired_actions:
            if done:
                break

            a_int = int(a)
            mask = np.asarray(env.get_action_mask(), dtype=bool)

            if not mask[a_int]:
                bc_early_stop_infeasible = True
                break

            obs_inst = self.model.obs_to_instance(obs, device=self.device)
            encoded, _, _ = self.model.encode_state(obs_inst)
            last_node_idx = int(obs["current_node"])

            out = self.model.forward_step(
                encoded=encoded,
                last_node_idx=last_node_idx,
                mask=mask,
                lstm_state=lstm_state,
            )
            lstm_state = out["lstm_state"]

            a_tensor = torch.tensor(a_int, device=self.device)
            log_prob = out["log_probs"].gather(0, a_tensor.view(1)).squeeze(0)
            teacher_log_probs.append(log_prob)
            executed_actions.append(a_int)

            obs, reward, done, step_info = env.step(a_int)

            if a_int == 0 and not done:
                lstm_state = None

            total_reward += float(reward)

            si = step_info.get("step_info") if isinstance(step_info, dict) else None
            if si is not None:
                in_time_window += int(getattr(si, "in_time_window", 0))
                positive_battery += int(getattr(si, "positive_battery", 0))

        loss_bc = (
            torch.tensor(0.0, device=self.device)
            if len(teacher_log_probs) == 0
            else -torch.stack(teacher_log_probs).mean()
        )

        n_exec = max(len(executed_actions), 1)
        teacher_result = TeacherResult(
            actions=executed_actions,
            repair_status=repair_status,
            hint_valid=hint_valid,
            total_reward=total_reward,
            total_distance=float(getattr(env, "total_distance", 0.0)),
            in_time_window=in_time_window / n_exec,
            positive_battery=positive_battery / n_exec,
            served_count=float(getattr(env, "served_rate", 0.0)),
            bc_early_stop_infeasible=bc_early_stop_infeasible,
            n_bc_actions=len(teacher_log_probs),
        )
        return loss_bc, teacher_result


    def train_step(self, batch):
        """Sample routes (no gradient), repair them with OR-Tools, then accumulate the RL gradient
        and the weighted BC gradient, and take an optimizer step.
        """
        self.model.train()
        self.baseline_model.eval()

        graphs = batch["graphs"]
        B = len(graphs)

        accum_steps = max(int(self.accum_steps), 1)

        if self._accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        self.model.eval()

        sample_results = []
        for i in range(B):
            env_sample = self._make_env(graphs[i])
            result = self.rollout_with_model(
                model=self.model, env=env_sample,
                mode="sample", keep_graph=False,
            )
            sample_results.append(result)

        baseline_results = []
        baseline_rewards = []
        with torch.no_grad():
            for i in range(B):
                env_base = self._make_env(graphs[i])
                base_result = self.rollout_with_model(
                    model=self.baseline_model, env=env_base,
                    mode="greedy", keep_graph=False,
                )
                baseline_results.append(base_result)
                baseline_rewards.append(float(base_result.total_reward))

        advantages, raw_advantages, episode_lens = (
            self._compute_rl_advantages_from_rewards(
                sample_results, baseline_rewards,
            )
        )

        _probe_named_params = list(self.model.named_parameters())
        _probe_params = [p for _, p in _probe_named_params]
        _prev_grads = [
            (p.grad.detach().clone() if p.grad is not None else None)
            for p in _probe_params
        ]

        with self._phase_timer("rl_recompute_backward"):
            rl_loss_value, log_prob_sums = self._reinforce_recompute_backward(
                sample_results=sample_results,
                advantages=advantages,
                episode_lens=episode_lens,
                graphs=graphs,
                B=B,
                accum_steps=accum_steps,
            )

        self.model.train()

        _rl_grads = []
        for p, prev in zip(_probe_params, _prev_grads):
            if p.grad is None:
                _rl_grads.append(None)
                continue
            g_rl = p.grad.detach() if prev is None else (p.grad.detach() - prev)
            _rl_grads.append(g_rl.clone())

        _after_rl_grads = [
            (p.grad.detach().clone() if p.grad is not None else None)
            for p in _probe_params
        ]

        del _prev_grads

        self.optimizer.zero_grad(set_to_none=True)


        items = [
            (graphs[i], sample_results[i].actions)
            for i in range(B)
        ]
        with self._phase_timer("gls_repair"):
            repair_outputs = self.ortools_repair_fn.repair_batch(
                items,
                time_limit_s=self.ortools_repair_time_limit,
                balance_coefficient=self.cfg["train"].get("balance_coefficient", 10),
                vehicle_fixed_cost=self.cfg["train"].get("vehicle_fixed_cost", 100),
                charge_rate=self.cfg["env"]["charging_rate"],
                energy_per_km=self.cfg["env"]["energy_rate"],
            )

        teacher_results: List = []
        triggered: List[bool] = []
        with self._phase_timer("gls_traj_replay"):
            for i in range(B):
                repaired_actions_i, repair_info_i = repair_outputs[i]
                teacher_result = self._replay_teacher_env(
                    repaired_actions_i, repair_info_i,
                    graphs[i],
                )
                teacher_results.append(teacher_result)
                if teacher_result.n_bc_actions == 0:
                    triggered.append(False)
                elif self.bc_only_if_teacher_better:
                    triggered.append(
                        self.is_gls_better_for_bc(
                            rl_info=sample_results[i], gls_info=teacher_result,
                        )
                    )
                else:
                    triggered.append(True)

        trig_idx = [i for i in range(B) if triggered[i]]
        gls_bc_triggered_rate = float(len(trig_idx)) / max(B, 1)
        bc_weight = float(self._get_bc_weight())

        bc_coefficients = {}
        if len(trig_idx) > 0:
            if self.bc_loss_aggregation == "mean_of_means":
                coefficient = 1.0 / len(trig_idx)
                for i in trig_idx:
                    bc_coefficients[i] = coefficient
            elif self.bc_loss_aggregation == "pooled":
                total_bc_actions_for_weight = float(
                    sum(teacher_results[i].n_bc_actions for i in trig_idx)
                )
                if total_bc_actions_for_weight > 0:
                    for i in trig_idx:
                        bc_coefficients[i] = (
                            float(teacher_results[i].n_bc_actions)
                            / total_bc_actions_for_weight
                        )
                else:
                    trig_idx = []
                    bc_coefficients = {}
            else:
                raise ValueError(
                    f"Unknown bc_loss_aggregation={self.bc_loss_aggregation}"
                )

        bc_loss_value = 0.0
        with self._phase_timer("bc_replay", cuda_sync=True):
            for i in trig_idx:
                repaired_actions_i, repair_info_i = repair_outputs[i]
                bc_loss_i, teacher_full_i = self._teacher_forcing_bc(
                    repaired_actions_i, repair_info_i,
                    graph_df=graphs[i],
                )
                teacher_results[i] = teacher_full_i
                coefficient_i = float(bc_coefficients[i])
                bc_loss_i_value = float(bc_loss_i.detach().item())
                bc_loss_value += coefficient_i * bc_loss_i_value

                if isinstance(bc_loss_i, torch.Tensor) and bc_loss_i.requires_grad:
                    scaled_bc_loss = (
                        bc_weight * coefficient_i * bc_loss_i / accum_steps
                    )
                    scaled_bc_loss.backward()
                    del scaled_bc_loss
                del bc_loss_i

        _bc_grads = [
            (p.grad.detach() if p.grad is not None else None)
            for p in _probe_params
        ]

        grad_alignment = self._compute_grad_alignment_metrics(
            named_params=_probe_named_params,
            rl_grads=_rl_grads,
            bc_grads=_bc_grads,
        )

        rl_grad_norm = grad_alignment["rl_grad_norm"]
        bc_grad_norm = grad_alignment["bc_grad_norm"]
        bc_rl_grad_ratio = grad_alignment["bc_rl_grad_ratio"]
        grad_cosine_rl_bc = grad_alignment["grad_cosine_rl_bc"]

        for p, after_rl in zip(_probe_params, _after_rl_grads):
            if after_rl is None:
                continue
            if p.grad is None:
                p.grad = after_rl
            else:
                p.grad = p.grad + after_rl

        del _rl_grads, _bc_grads, _after_rl_grads

        loss_value = rl_loss_value + bc_weight * bc_loss_value

        self._accum_counter += 1
        optimizer_stepped = False
        if self._accum_counter >= accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_counter = 0
            optimizer_stepped = True

        sample_lens = [len(r.actions) for r in sample_results]
        if baseline_results is not None and len(baseline_results) > 0:
            reward_baseline = float(
                np.mean([r.total_reward for r in baseline_results])
            )
        else:
            reward_baseline = (
                float(np.mean(baseline_rewards)) if baseline_rewards else 0.0
            )

        stats = {
            "reward": float(np.mean([r.total_reward for r in sample_results])),
            "len": float(np.mean(sample_lens)),
            "served_count": float(np.mean([
                float(getattr(r, "served_count", 0.0)) for r in sample_results
            ])),
            "in_time_window_ratio": float(np.mean([
                float(getattr(r, "in_time_window", 0.0)) for r in sample_results
            ])),
            "positive_battery_ratio": float(np.mean([
                float(getattr(r, "positive_battery", 0.0)) for r in sample_results
            ])),
            "total_distance": float(np.mean([
                float(getattr(r, "total_distance", 0.0)) for r in sample_results
            ])),

            "loss": float(loss_value),
            "loss_rl": float(rl_loss_value),
            "loss_bc": float(bc_loss_value),
            "bc_weight": float(bc_weight),
            "bc_loss_aggregation": self.bc_loss_aggregation,

            "advantage": float(np.mean(advantages)) if advantages else 0.0,
            "advantage_abs": float(np.mean(np.abs(advantages))) if advantages else 0.0,
            "raw_advantage": float(np.mean(raw_advantages)) if raw_advantages else 0.0,
            "log_prob_sum": float(np.mean(log_prob_sums)) if log_prob_sums else 0.0,

            "rl_grad_norm": float(rl_grad_norm),
            "bc_grad_norm": float(bc_grad_norm),
            "bc_rl_grad_ratio": float(bc_rl_grad_ratio),
            "grad_cosine_rl_bc": float(grad_cosine_rl_bc),

            "encoder_rl_grad_norm": float(grad_alignment["encoder_rl_grad_norm"]),
            "encoder_bc_grad_norm": float(grad_alignment["encoder_bc_grad_norm"]),
            "encoder_bc_rl_grad_ratio": float(grad_alignment["encoder_bc_rl_grad_ratio"]),
            "encoder_grad_cosine_rl_bc": float(grad_alignment["encoder_grad_cosine_rl_bc"]),

            "lstm_rl_grad_norm": float(grad_alignment["lstm_rl_grad_norm"]),
            "lstm_bc_grad_norm": float(grad_alignment["lstm_bc_grad_norm"]),
            "lstm_bc_rl_grad_ratio": float(grad_alignment["lstm_bc_rl_grad_ratio"]),
            "lstm_grad_cosine_rl_bc": float(grad_alignment["lstm_grad_cosine_rl_bc"]),

            "attention_rl_grad_norm": float(grad_alignment["attention_rl_grad_norm"]),
            "attention_bc_grad_norm": float(grad_alignment["attention_bc_grad_norm"]),
            "attention_bc_rl_grad_ratio": float(grad_alignment["attention_bc_rl_grad_ratio"]),
            "attention_grad_cosine_rl_bc": float(grad_alignment["attention_grad_cosine_rl_bc"]),

            "optimizer_stepped": float(optimizer_stepped),
            "reward_baseline": float(reward_baseline),

            "teacher_reward": float(np.mean([
                float(t.total_reward or 0.0) for t in teacher_results
            ])),
            "teacher_served_count": float(np.mean([
                float(t.served_count) for t in teacher_results
            ])),
            "teacher_total_distance": float(np.mean([
                float(t.total_distance or 0.0) for t in teacher_results
            ])),
            "teacher_in_time_window_ratio": float(np.mean([
                float(t.in_time_window) for t in teacher_results
            ])),
            "teacher_positive_battery_ratio": float(np.mean([
                float(t.positive_battery) for t in teacher_results
            ])),

            "gls_bc_triggered": float(gls_bc_triggered_rate),
            "bc_early_stop_rate": float(np.mean([
                bool(t.bc_early_stop_infeasible) for t in teacher_results
            ])),
            "bc_num_actions": float(np.mean([
                int(getattr(t, "n_bc_actions", len(t.actions))) for t in teacher_results
            ])),
            "bc_total_actions": float(np.sum([
                int(getattr(t, "n_bc_actions", len(t.actions))) for t in teacher_results
            ])),
            "num_dropped_nodes": float(np.mean([
                len(getattr(t, "dropped_nodes", [])) for t in teacher_results
            ])),
        }


        env_greedy = self._make_env(graphs[0])
        with self._phase_timer("greedy_eval", cuda_sync=True):
            greedy_result = self.rollout_greedy(env_greedy)

        viz_payload = {
            "graph_df": graphs[0],
            "greedy_result": greedy_result,
        }

        return stats, viz_payload