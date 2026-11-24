from __future__ import annotations
from typing import List, Optional

import torch
import numpy as np
from dataclasses import dataclass, field

from algorithm.GAT_Transformer.GAT_Transformer import GATTransformerTrainer
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


class HGGATTransformerTrainer(GATTransformerTrainer):
    """GATTransformerTrainer plus an OR-Tools-guided BC loss: each sampled route is repaired by
    OR-Tools GLS. If the repaired route is better (bc_only_if_teacher_better), the policy is
    trained to imitate it by teacher forcing."""

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
        if self.bc_loss_aggregation not in ("mean_of_means", "pooled"):
            raise ValueError(
                f"Unknown bc_loss_aggregation={self.bc_loss_aggregation}. "
                f"Valid options are: ['mean_of_means', 'pooled']"
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
        return self.env_builder(graph_df)

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


    def _teacher_forcing_bc(
        self,
        repaired_actions,
        info,
        graph_df,
    ):
        repair_status = info.get("status", "unknown") if isinstance(info, dict) else "unknown"
        hint_valid = bool(info.get("hint_valid", False)) if isinstance(info, dict) else False

        repair_failed_fallback = str(repair_status).startswith("subprocess_")
        if repair_failed_fallback:
            empty_result = TeacherResult(
                actions=[], repair_status=repair_status, hint_valid=hint_valid,
                total_reward=0.0, total_distance=0.0,
                in_time_window=0.0, positive_battery=0.0, served_count=0.0,
                n_bc_actions=0,
            )
            return torch.tensor(0.0, device=self.device), empty_result

        env = self.env_builder(graph_df)
        obs = env.reset()
        done = False

        init_inst = self.model.obs_to_instance(obs, device=self.device)
        h, h_graph, _, _ = self.model.encode(init_inst)
        if h_graph.dim() == 1:
            h_graph = h_graph.unsqueeze(0)

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
            last_node_idx = int(obs["current_node"])

            probs, log_prob_all = self.model.forward_step(
                h=h,
                h_graph=h_graph,
                obs_inst=obs_inst,
                last_node_idx=last_node_idx,
                mask_np=mask,
            )

            log_prob_all = log_prob_all.view(-1)
            log_prob = log_prob_all[a_int]
            teacher_log_probs.append(log_prob)
            executed_actions.append(a_int)

            obs, reward, done, step_info = env.step(a_int)
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
        """REINFORCE step plus the weighted BC loss. The RL and the per-graph BC losses are
        backpropagated one after the other, so their computation graphs are never all in memory.
        """
        self.model.train()
        self.baseline_model.eval()

        graphs = batch["graphs"]
        B = len(graphs)

        if self._accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        accum_steps = max(int(self.accum_steps), 1)

        envs_sample = []
        envs_base = []
        init_obs_list = []

        for i in range(B):
            env_s = self.env_builder(graphs[i])
            env_b = self.env_builder(graphs[i])

            obs = env_s.reset()
            env_b.reset()

            envs_sample.append(env_s)
            envs_base.append(env_b)
            init_obs_list.append(obs)

        batch_inst = self.model.obs_list_to_instance_batch(
            init_obs_list,
            device=self.device,
        )

        h_batch, h_graph_batch, _, _ = self.model.encode_batch(batch_inst)

        with torch.no_grad():
            batch_inst_b = self.baseline_model.obs_list_to_instance_batch(
                init_obs_list,
                device=self.device,
            )

            h_batch_b, h_graph_batch_b, _, _ = (
                self.baseline_model.encode_batch(batch_inst_b)
            )

        sample_results = []
        baseline_results = []

        with self._phase_timer("rollout", cuda_sync=True):
            for i in range(B):

                sample_result = self._rollout_with_h(
                    model=self.model,
                    env=envs_sample[i],
                    h=h_batch[i],
                    h_graph=h_graph_batch[i],
                    mode="sample",
                    keep_graph=True,
                )

                sample_results.append(sample_result)

                with torch.no_grad():
                    baseline_result = self._rollout_with_h(
                        model=self.baseline_model,
                        env=envs_base[i],
                        h=h_batch_b[i],
                        h_graph=h_graph_batch_b[i],
                        mode="greedy",
                        keep_graph=False,
                    )

                baseline_results.append(baseline_result)

        baseline_rewards = [
            float(r.total_reward)
            for r in baseline_results
        ]

        rl_loss, _rl_aux = self._compute_reinforce_loss(
            sample_results,
            baseline_rewards,
        )

        advantages = _rl_aux["raw_advantages"]

        rl_loss_value = float(rl_loss.detach().item())

        (rl_loss / accum_steps).backward()

        for r in sample_results:
            if hasattr(r, "log_probs"):
                r.log_probs.clear()

        del rl_loss
        del h_batch
        del h_graph_batch
        del batch_inst

        del h_batch_b
        del h_graph_batch_b
        del batch_inst_b

        items = [
            (
                graphs[i],
                sample_results[i].actions,
            )
            for i in range(B)
        ]

        with self._phase_timer("gls_repair"):
            repair_outputs = self.ortools_repair_fn.repair_batch(
                items,
                time_limit_s=self.ortools_repair_time_limit,
                balance_coefficient=self.cfg["train"].get(
                    "balance_coefficient", 10
                ),
                vehicle_fixed_cost=self.cfg["train"].get(
                    "vehicle_fixed_cost", 100
                ),
                charge_rate=self.cfg["env"]["charging_rate"],
                energy_per_km=self.cfg["env"]["energy_rate"],
            )

        teacher_results: List[TeacherResult] = []
        triggered: List[bool] = []

        with self._phase_timer("gls_traj_replay"):
            for i in range(B):

                repaired_actions_i, repair_info_i = repair_outputs[i]

                teacher_result = self._replay_teacher_env(
                    repaired_actions_i,
                    repair_info_i,
                    graphs[i],
                )

                teacher_results.append(teacher_result)
                if teacher_result.n_bc_actions == 0:
                    triggered.append(False)
                elif self.bc_only_if_teacher_better:
                    triggered.append(
                        self.is_gls_better_for_bc(
                            rl_info=sample_results[i],
                            gls_info=teacher_result,
                        )
                    )
                else:
                    triggered.append(True)

        trig_idx = [
            i
            for i in range(B)
            if triggered[i]
        ]
        gls_bc_triggered_rate = (
            float(len(trig_idx)) / max(B, 1)
        )
        bc_weight = float(self._get_bc_weight())
        bc_coefficients = {}

        if len(trig_idx) > 0:

            if self.bc_loss_aggregation == "mean_of_means":
                coeff = 1.0 / len(trig_idx)
                for i in trig_idx:
                    bc_coefficients[i] = coeff
            else:
                total_bc_actions_for_weight = float(
                    sum(
                        teacher_results[i].n_bc_actions
                        for i in trig_idx
                    )
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

        bc_loss_value = 0.0

        with self._phase_timer("bc_replay", cuda_sync=True):
            for i in trig_idx:
                repaired_actions_i, repair_info_i = repair_outputs[i]
                bc_loss_i, teacher_full_i = self._teacher_forcing_bc(
                    repaired_actions_i,
                    repair_info_i,
                    graph_df=graphs[i],
                )

                teacher_results[i] = teacher_full_i
                coeff_i = float(bc_coefficients[i])

                bc_loss_value += (
                    coeff_i
                    * float(bc_loss_i.detach().item())
                )

                if bc_loss_i.requires_grad:
                    scaled_bc_loss = (
                        bc_weight
                        * coeff_i
                        * bc_loss_i
                        / accum_steps
                    )

                    scaled_bc_loss.backward()
                    del scaled_bc_loss
                del bc_loss_i

        self._accum_counter += 1

        if self._accum_counter >= accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_counter = 0

        total_loss_value = (
            rl_loss_value
            + bc_weight * bc_loss_value
        )

        stats = {
            "reward": float(
                np.mean([
                    r.total_reward
                    for r in sample_results
                ])
            ),

            "len": float(
                np.mean([
                    len(r.actions)
                    for r in sample_results
                ])
            ),

            "served_count": float(
                np.mean([
                    float(getattr(r, "served_count", 0.0))
                    for r in sample_results
                ])
            ),

            "in_time_window_ratio": float(
                np.mean([
                    float(getattr(r, "in_time_window", 0.0))
                    for r in sample_results
                ])
            ),

            "positive_battery_ratio": float(
                np.mean([
                    float(getattr(r, "positive_battery", 0.0))
                    for r in sample_results
                ])
            ),

            "total_distance": float(
                np.mean([
                    float(getattr(r, "total_distance", 0.0))
                    for r in sample_results
                ])
            ),

            "loss": float(total_loss_value),
            "loss_rl": float(rl_loss_value),
            "loss_bc": float(bc_loss_value),
            "bc_weight": float(bc_weight),

            "advantage": float(
                np.mean(advantages)
            ),

            "reward_baseline": float(
                np.mean([
                    r.total_reward
                    for r in baseline_results
                ])
            ),

            "teacher_reward": float(
                np.mean([
                    float(t.total_reward or 0.0)
                    for t in teacher_results
                ])
            ),

            "teacher_served_count": float(
                np.mean([
                    float(t.served_count)
                    for t in teacher_results
                ])
            ),

            "teacher_total_distance": float(
                np.mean([
                    float(t.total_distance or 0.0)
                    for t in teacher_results
                ])
            ),

            "teacher_in_time_window_ratio": float(
                np.mean([
                    float(t.in_time_window)
                    for t in teacher_results
                ])
            ),

            "teacher_positive_battery_ratio": float(
                np.mean([
                    float(t.positive_battery)
                    for t in teacher_results
                ])
            ),

            "gls_bc_triggered": float(
                gls_bc_triggered_rate
            ),

            "bc_early_stop_rate": float(
                np.mean([
                    bool(t.bc_early_stop_infeasible)
                    for t in teacher_results
                ])
            ),

            "bc_num_actions": float(
                np.mean([
                    int(t.n_bc_actions)
                    for t in teacher_results
                ])
            ),

            "bc_total_actions": float(
                np.sum([
                    int(t.n_bc_actions)
                    for t in teacher_results
                ])
            ),
        }

        env_greedy = self.env_builder(graphs[0])

        with self._phase_timer(
            "greedy_eval",
            cuda_sync=True,
        ):
            greedy_result = self.rollout_greedy(
                env_greedy
            )

        viz_payload = {
            "graph_df": graphs[0],
            "greedy_result": greedy_result,
        }

        return stats, viz_payload