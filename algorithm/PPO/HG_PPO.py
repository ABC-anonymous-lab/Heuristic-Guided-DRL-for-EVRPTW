from collections import Counter

import torch
import numpy as np
from typing import List, Optional

from algorithm.PPO.PPO import PPOTrainer
from .ppo_common import PPORolloutResult, PPORolloutSampleResult
from algorithm.OR_Tools_GLS_guidance.subprocess_wrapper import IsolatedOrtoolsRepair



class HGPPOTrainer(PPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_cfg = self.cfg["train"]
        self.use_gls_guided = bool(train_cfg.get("use_gls_guided", False))
        repair_fn_file_name = train_cfg["repair_file_name"]
        self.ortools_repair_fn = IsolatedOrtoolsRepair(
            repair_module_name=repair_fn_file_name,
            env_builder_factory_module_name="utils.init_model",
            env_builder_factory_func_name="build_env_builder",
            env_builder_cfg=self.cfg,
            timeout_buffer_s=10.0,
            worker_recycle_every=int(train_cfg.get("ortools_worker_recycle_every", 100)),
            num_workers=int(train_cfg.get("ortools_num_workers", 4)),
        )
        self.ortools_repair_time_limit = int(self.cfg["train"]["ortools_repair_time_limit"])

        self.use_imitation_loss = bool(train_cfg.get("use_imitation_loss", True))
        self.bc_max_weight = float(train_cfg.get("bc_max_weight", 0.5))
        self.bc_min_weight = float(train_cfg.get("bc_min_weight", 0.1))
        self.bc_temperature = float(train_cfg.get("bc_temperature", 1.0))

        self.bc_loss_aggregation = train_cfg.get("bc_loss_aggregation", "pooled")
        if self.bc_loss_aggregation not in ("pooled", "mean_of_means"):
            raise ValueError(f"Unknown bc_loss_aggregation: {self.bc_loss_aggregation!r}")

        self._last_bc_loss = torch.tensor(0.0, device=self.device)


    def _get_bc_weight(self):
        if self.total_rollout_steps <= 0:
            return self.bc_max_weight
        portion = max(0.0, min(1.0, self.global_step / self.total_rollout_steps))
        return self.bc_max_weight - (self.bc_max_weight - self.bc_min_weight) * portion


    @staticmethod
    def _rollout_field(x, name):
        if isinstance(x, dict):
            return x[name]
        return getattr(x, name)


    def is_gls_better_for_bc(self, rl_info, gls_info, margin=0.1):
        if gls_info is None:
            return False
        get = self._rollout_field
        if get(gls_info, "total_reward") > get(rl_info, "total_reward"):
            return True
        if get(gls_info, "served_count") > get(rl_info, "served_count"):
            return True
        if get(gls_info, "served_count") < get(rl_info, "served_count") - margin:
            return False
        return get(gls_info, "total_distance") < get(rl_info, "total_distance")


    def _after_rollout_hook(self, graph_items, rollouts) -> dict:
        B = len(graph_items)
        gls_rollouts: List[Optional[dict]] = [None] * B
        gls_better_flags = [False] * B

        if self.use_gls_guided:
            items = [
                (graph_df, rollouts[i].actions)
                for i, graph_df in enumerate(graph_items)
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

            for i, graph_df in enumerate(graph_items):
                repaired_actions, repair_info = repair_outputs[i]
                if repaired_actions is None or len(repaired_actions) == 0:
                    continue
                if all(int(a) == 0 for a in repaired_actions):
                    continue
                if isinstance(repair_info, dict) and str(repair_info.get("status", "")).startswith("subprocess_"):
                    continue

                with self._phase_timer("gls_traj_replay"):
                    env_for_rollout = self.env_builder(graph_df)
                    gls_rollout = self.rollout_with_action_sequence(
                        env=env_for_rollout, action_sequence=repaired_actions,
                    )
                gls_rollouts[i] = gls_rollout
                gls_better_flags[i] = self.is_gls_better_for_bc(rl_info=rollouts[i], gls_info=gls_rollout)

        return {"gls_rollouts": gls_rollouts, "gls_better_flags": gls_better_flags}


    def _extra_loss_fn(self, *, graph_items, actions_batch, after_rollout_ctx):
        return self._build_bc_loss_global(graph_items, after_rollout_ctx)


    def _collect_bc_triggered(self, graph_items, after_rollout_ctx):
        gls_rollouts = after_rollout_ctx["gls_rollouts"]
        gls_better_flags = after_rollout_ctx["gls_better_flags"]
        if not (self.use_gls_guided and any(gls_better_flags)):
            return [], gls_rollouts, gls_better_flags
        triggered_indices = [
            i for i, flag in enumerate(gls_better_flags)
            if flag and gls_rollouts[i] is not None
        ]
        return triggered_indices, gls_rollouts, gls_better_flags


    def _build_bc_loss_global(self, graph_items, after_rollout_ctx):
        triggered_indices, gls_rollouts, _ = self._collect_bc_triggered(graph_items, after_rollout_ctx)
        if not triggered_indices:
            self._last_bc_loss = torch.tensor(0.0, device=self.device)
            return None

        bc_graph_items = [graph_items[i] for i in triggered_indices]
        bc_actions_batch = [list(gls_rollouts[i]["actions"]) for i in triggered_indices]

        def extra_loss_fn():
          with self._phase_timer("bc_replay", cuda_sync=True):
              if self.bc_loss_aggregation == "mean_of_means":
                  (
                      bc_log_probs, bc_values, bc_entropies,
                      bc_lp_segs, bc_v_segs, bc_ent_segs,
                  ) = self.evaluate_actions_on_rollout_batch(
                      graph_items=bc_graph_items, actions_batch=bc_actions_batch,
                      temperature=self.bc_temperature, return_segments=True,
                  )
                  per_graph_bc_means = torch.stack(
                      [-lp.mean() for lp in bc_lp_segs if lp.numel() > 0]
                  )
                  bc_loss = per_graph_bc_means.mean()
              else:
                  bc_log_probs, bc_values, bc_entropies = self.evaluate_actions_on_rollout_batch(
                      graph_items=bc_graph_items, actions_batch=bc_actions_batch,
                      temperature=self.bc_temperature,
                  )
                  bc_loss = -bc_log_probs.mean()

              self._last_bc_loss = bc_loss.detach()
              return self._get_bc_weight() * bc_loss

        return extra_loss_fn


    def _extra_stats(self, *, graph_items, rollouts, after_rollout_ctx) -> dict:
        gls_rollouts = after_rollout_ctx["gls_rollouts"]
        gls_better_flags = after_rollout_ctx["gls_better_flags"]

        stats = {"bc_loss": float(self._last_bc_loss.item())}

        valid = []
        if self.use_gls_guided and any(r is not None for r in gls_rollouts):
            valid = [i for i, r in enumerate(gls_rollouts) if r is not None]
            stats.update({
                "reward_gls": float(np.mean([gls_rollouts[i]["total_reward"] for i in valid])),
                "served_count_gls": float(np.mean([gls_rollouts[i]["served_count"] for i in valid])),
                "in_time_window_ratio_gls": float(np.mean([gls_rollouts[i]["in_time_window"] for i in valid])),
                "positive_battery_ratio_gls": float(np.mean([gls_rollouts[i]["positive_battery"] for i in valid])),
                "total_distance_gls": float(np.mean([gls_rollouts[i]["total_distance"] for i in valid])),
                "gls_bc_triggered": float(np.mean([float(x) for x in gls_better_flags])),
                "len_gls_raw": float(np.mean([gls_rollouts[i]["raw_action_len"] for i in valid])),
                "len_gls_executed": float(np.mean([gls_rollouts[i]["executed_action_len"] for i in valid])),
            })

        reasons = [
            gls_rollouts[i].get(
                "termination_reason",
                "unknown"
            )
            for i in valid
        ]
        reason_counter = Counter(reasons)
        total_valid = len(valid)
        for reason, count in reason_counter.items():
            reason_name = str(reason) if reason is not None else "none"
            stats[f"gls_done_reason/{reason_name}"] = count / total_valid

        stats["bc_weight"] = self._get_bc_weight()

        return stats


    def rollout_with_action_sequence(self, env, action_sequence):
        obs = env.reset()
        done = False

        actions, rewards, infos, dones = [], [], [], []

        if len(action_sequence) > 0 and int(action_sequence[0]) == 0:
            action_sequence = action_sequence[1:]
            actions = [0]

        in_time_window = 0
        positive_battery = 0

        for a in action_sequence:
            if done:
                break
            obs, reward, done, info = env.step(int(a))
            actions.append(int(a))
            rewards.append(float(reward))
            infos.append(info)
            dones.append(bool(done))
            in_time_window += int(info["step_info"].in_time_window)
            positive_battery += int(info["step_info"].positive_battery)

        num_steps = len(actions)
        raw_action_len = len(action_sequence)

        if infos:
            termination_reason = infos[-1].get("termination_reason", "unknown")
        else:
            termination_reason = "no_step_executed"

        return {
            "actions": actions,
            "rewards": rewards,
            "infos": infos,
            "dones": dones,
            "served_count": env.served_rate,
            "total_reward": float(np.sum(rewards)),
            "total_distance": float(env.total_distance),
            "in_time_window": float((in_time_window / num_steps) if num_steps > 0 else 0.0),
            "positive_battery": float((positive_battery / num_steps) if num_steps > 0 else 0.0),
            "done": bool(done),
            "raw_action_len": raw_action_len,
            "executed_action_len": len(actions),
            "termination_reason": termination_reason,
        }


__all__ = ["HGPPOTrainer", "PPORolloutResult", "PPORolloutSampleResult"]
