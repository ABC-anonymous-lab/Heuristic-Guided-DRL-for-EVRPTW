from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

from algorithm.base_trainer import BaseTrainer


BASELINE_EVAL_SIZE = 1000
BASELINE_ALPHA = 0.05


@dataclass
class PNLSTMRolloutResult:

    actions: List[int]
    log_probs: List[torch.Tensor]
    rewards: List[float]
    infos: List[Any]
    served_count: float
    total_reward: float
    total_distance: float
    in_time_window: float
    positive_battery: float
    done: bool
    vehicle_distances: Optional[List[float]] = None
    vehicle_used: Optional[int] = None
    station_visits: Optional[int] = None
    negative_battery_sum: Optional[float] = None
    termination_reason: Optional[str] = None


class PNLSTMTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.baseline_model = copy.deepcopy(self.model).to(self.device)
        self.baseline_model.eval()
        for p in self.baseline_model.parameters():
            p.requires_grad_(False)

        train_cfg = self.cfg["train"]
        self.accum_steps = int(train_cfg.get("reinforce_accum_steps", 1))
        self._accum_counter = 0


    @staticmethod
    def _extract_graph_id_from_df(graph_df, fallback: int) -> Any:
        if hasattr(graph_df, "columns") and "graph_id" in graph_df.columns:
            graph_id = graph_df["graph_id"].iloc[0]
            try:
                return int(graph_id)
            except Exception:
                return str(graph_id)
        return int(fallback)


    def _make_env(self, graph_df):
        return self.env_builder(graph_df)


    def rollout_with_model(
        self,
        model: torch.nn.Module,
        env,
        mode: str = "sample",
        keep_graph: bool = True,
    ) -> PNLSTMRolloutResult:
        """Run one episode with the given model (sample or greedy)."""
        obs = env.reset()
        done = False
        lstm_state = None

        actions: List[int] = []
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        infos: List[Any] = []
        in_time_window = 0
        positive_battery = 0
        termination_reason: Optional[str] = None

        ctx = torch.enable_grad() if keep_graph else torch.no_grad()

        with ctx:
            while not done:
                mask = env.get_action_mask()
                mask = np.asarray(mask, dtype=bool)

                if not np.any(mask):
                    env.done = True
                    done = True
                    termination_reason = "no_feasible_action_before_step"
                    try:
                        rewards.append(float(env.get_reward()))
                    except Exception:
                        rewards.append(0.0)
                    break

                obs_inst = model.obs_to_instance(obs, device=self.device)
                encoded, _, _ = model.encode_state(obs_inst)
                last_node_idx = int(obs["current_node"])

                if mode in ("sample", "stochastic"):
                    action_t, log_prob_t, entropy_t, lstm_state, aux = model.sample_action(
                        encoded=encoded,
                        last_node_idx=last_node_idx,
                        mask=mask,
                        lstm_state=lstm_state,
                    )
                    if keep_graph:
                        log_probs.append(log_prob_t)
                elif mode == "greedy":
                    action_t, log_prob_t, lstm_state, aux = model.greedy_action(
                        encoded=encoded,
                        last_node_idx=last_node_idx,
                        mask=mask,
                        lstm_state=lstm_state,
                    )
                else:
                    raise ValueError(f"Unknown rollout mode: {mode}")

                action = int(action_t.item())
                obs, reward, done, info = env.step(action)

                if action == 0 and not done:
                    lstm_state = None

                actions.append(action)
                rewards.append(float(reward))
                infos.append(info)

                step_info = info.get("step_info") if isinstance(info, dict) else None
                if step_info is not None:
                    in_time_window += int(getattr(step_info, "in_time_window", 0))
                    positive_battery += int(getattr(step_info, "positive_battery", 0))

                if isinstance(info, dict):
                    termination_reason = info.get("termination_reason", termination_reason)

        num_steps = len(actions)


        vehicle_distances = getattr(env, "vehicle_distances", None)
        if vehicle_distances is not None:
            vehicle_distances = list(vehicle_distances)

        return PNLSTMRolloutResult(
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            infos=infos,
            served_count=float(getattr(env, "served_rate", 0.0)),
            total_reward=float(np.sum(rewards)),
            total_distance=float(getattr(env, "total_distance", 0.0)),
            in_time_window=float(in_time_window / num_steps) if num_steps > 0 else 0.0,
            positive_battery=float(positive_battery / num_steps) if num_steps > 0 else 0.0,
            done=bool(getattr(env, "done", done)),
            vehicle_distances=vehicle_distances,
            vehicle_used=int(getattr(env, "vehicle_used", 0)) if hasattr(env, "vehicle_used") else None,
            station_visits=int(getattr(env, "station_visits", 0)) if hasattr(env, "station_visits") else None,
            negative_battery_sum=float(getattr(env, "negative_battery_sum", 0.0)),
            termination_reason=termination_reason,
        )


    def rollout_greedy(self, env) -> PNLSTMRolloutResult:
        return self.rollout_with_model(self.model, env=env, mode="greedy", keep_graph=False)


    def _compute_reinforce_loss(
        self,
        sample_results: List[PNLSTMRolloutResult],
        baseline_rewards: List[float],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        per_graph_log_prob_terms: List[torch.Tensor] = []
        raw_advantages: List[float] = []
        log_prob_sums: List[float] = []
        episode_lens: List[int] = []

        for sample_result, baseline_reward in zip(sample_results, baseline_rewards):
            R = float(sample_result.total_reward)
            b = float(baseline_reward)
            raw_advantages.append(R - b)

            T = len(sample_result.log_probs)
            episode_lens.append(T)

            if T == 0:
                log_prob_sum = torch.zeros((), device=self.device, requires_grad=True)
                log_prob_term = log_prob_sum
            else:
                log_probs = torch.stack(sample_result.log_probs)
                log_prob_sum = log_probs.sum()
                log_prob_term = log_prob_sum / max(T, 1)

            log_prob_sums.append(float(log_prob_sum.detach().item()))
            per_graph_log_prob_terms.append(log_prob_term)

        advantage_tensor = torch.tensor(
            raw_advantages, device=self.device, dtype=torch.float32
        )

        if advantage_tensor.numel() > 1:
            adv_mean = advantage_tensor.mean()
            adv_std = advantage_tensor.std(unbiased=False)
            advantage_tensor = (advantage_tensor - adv_mean) / (adv_std + 1e-8)

        advantages = [float(a.detach().item()) for a in advantage_tensor]

        per_graph_loss: List[torch.Tensor] = []
        for log_prob_term, advantage in zip(per_graph_log_prob_terms, advantage_tensor):
            per_graph_loss.append(-log_prob_term * advantage)

        loss = torch.stack(per_graph_loss).mean()

        aux = {
            "advantages": advantages,
            "raw_advantages": raw_advantages,
            "log_prob_sums": log_prob_sums,
            "episode_lens": episode_lens,
        }
        return loss, aux


    def train_step(self, batch) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Sample one route per graph, compute baseline rewards with the baseline model, and take a
        REINFORCE step (gradients accumulated over reinforce_accum_steps batches).
        """
        self.model.train()
        self.baseline_model.eval()

        graphs = batch["graphs"]
        B = len(graphs)

        sample_results: List[PNLSTMRolloutResult] = []

        for i in range(B):
            env_sample = self._make_env(graphs[i])
            result = self.rollout_with_model(
                model=self.model,
                env=env_sample,
                mode="sample",
                keep_graph=True,
            )
            sample_results.append(result)

        baseline_results: List[PNLSTMRolloutResult] = []
        baseline_rewards = []
        with torch.no_grad():
            for i in range(B):
                env_base = self._make_env(graphs[i])
                base_result = self.rollout_with_model(
                    model=self.baseline_model,
                    env=env_base,
                    mode="greedy",
                    keep_graph=False,
                )
                baseline_results.append(base_result)
                baseline_rewards.append(float(base_result.total_reward))

        if self._accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        loss, _rl_aux = self._compute_reinforce_loss(sample_results, baseline_rewards)
        advantages = _rl_aux["advantages"]
        log_prob_sums = _rl_aux["log_prob_sums"]
        (loss / max(self.accum_steps, 1)).backward()

        self._accum_counter += 1
        optimizer_stepped = False

        if self._accum_counter >= self.accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_counter = 0
            optimizer_stepped = True

        def avg_sample(attr: str) -> float:
            return float(np.mean([getattr(r, attr) for r in sample_results]))

        if baseline_results is not None and len(baseline_results) > 0:
            def avg_base(attr: str) -> float:
                return float(np.mean([getattr(r, attr) for r in baseline_results]))
            reward_baseline = avg_base("total_reward")
            len_baseline = float(np.mean([len(r.actions) for r in baseline_results]))
            served_baseline = avg_base("served_count")
            distance_baseline = avg_base("total_distance")
            done_baseline = float(np.mean([r.done for r in baseline_results]))
            in_tw_baseline = avg_base("in_time_window")
            positive_batt_baseline = avg_base("positive_battery")
        else:
            reward_baseline = float(np.mean(baseline_rewards)) if baseline_rewards else 0.0
            len_baseline = 0.0
            served_baseline = 0.0
            distance_baseline = 0.0
            done_baseline = 0.0
            in_tw_baseline = 0.0
            positive_batt_baseline = 0.0

        term_counts: Dict[str, int] = {}
        for r in sample_results:
            key = r.termination_reason or "none"
            term_counts[key] = term_counts.get(key, 0) + 1

        env_greedy = self._make_env(graphs[0])
        greedy_result = self.rollout_greedy(env_greedy)

        stats: Dict[str, float] = {
            "loss": float(loss.detach().item()),
            "reward": avg_sample("total_reward"),
            "reward_baseline": reward_baseline,
            "advantage": float(np.mean(advantages)) if advantages else 0.0,
            "advantage_abs": float(np.mean(np.abs(advantages))) if advantages else 0.0,
            "log_prob_sum": float(np.mean(log_prob_sums)) if log_prob_sums else 0.0,
            "len": float(np.mean([len(r.actions) for r in sample_results])),
            "served_count": avg_sample("served_count"),
            "total_distance": avg_sample("total_distance"),
            "done": float(np.mean([r.done for r in sample_results])),
            "station_visits": float(np.mean([r.station_visits or 0 for r in sample_results])),
            "negative_battery_sum": float(np.mean([r.negative_battery_sum or 0.0 for r in sample_results])),
            "vehicle_used": float(np.mean([r.vehicle_used or 0 for r in sample_results])),
            "optimizer_stepped": float(optimizer_stepped),

            "len_baseline": len_baseline,
            "served_count_baseline": served_baseline,
            "total_distance_baseline": distance_baseline,
            "done_baseline": done_baseline,

            "reward_greedy": float(greedy_result.total_reward),
            "len_greedy": float(len(greedy_result.actions)),
            "served_count_greedy": float(greedy_result.served_count),
            "total_distance_greedy": float(greedy_result.total_distance),

            "in_time_window_ratio": avg_sample("in_time_window"),
            "positive_battery_ratio": avg_sample("positive_battery"),
            "in_time_window_ratio_greedy": in_tw_baseline,
            "positive_battery_ratio_greedy": positive_batt_baseline,
        }

        for key, value in term_counts.items():
            stats[f"termination_{key}"] = float(value / max(B, 1))

        viz_payload = {
            "graph_df": graphs[0],
            "greedy_result": greedy_result,
        }

        return stats, viz_payload


    @torch.no_grad()
    def _rollout_baseline_rewards(self, loader, max_instances: Optional[int] = None) -> np.ndarray:
        self.baseline_model.eval()

        if max_instances is None:
            max_instances = BASELINE_EVAL_SIZE

        rewards: List[float] = []
        count = 0

        for batch in loader:
            graphs = batch["graphs"]

            for i in range(len(graphs)):
                env = self._make_env(graphs[i])
                result = self.rollout_with_model(
                    model=self.baseline_model,
                    env=env,
                    mode="greedy",
                    keep_graph=False,
                )
                rewards.append(float(result.total_reward))
                count += 1
                if count >= max_instances:
                    break
            if count >= max_instances:
                break

        return np.asarray(rewards, dtype=np.float64)


    def _baseline_t_test(self, policy_rewards: np.ndarray, baseline_rewards: np.ndarray) -> Dict[str, float]:
        n = min(len(policy_rewards), len(baseline_rewards))
        policy_rewards = np.asarray(policy_rewards[:n], dtype=np.float64)
        baseline_rewards = np.asarray(baseline_rewards[:n], dtype=np.float64)

        if n == 0:
            return {
                "baseline_updated": 0.0,
                "baseline_mean_policy_reward": 0.0,
                "baseline_mean_baseline_reward": 0.0,
                "baseline_mean_diff": 0.0,
                "baseline_t_stat": 0.0,
                "baseline_eval_n": 0,
            }

        diff = policy_rewards - baseline_rewards
        mean_diff = float(diff.mean())
        std_diff = float(diff.std(ddof=1)) if n > 1 else 0.0

        if n <= 1 or std_diff < 1e-12:
            should_update = mean_diff > 0.0
            t_stat = float("inf") if should_update else 0.0
        else:
            t_stat = mean_diff / (std_diff / math.sqrt(n))
            critical = float(NormalDist().inv_cdf(1.0 - BASELINE_ALPHA))
            should_update = t_stat > critical

        if should_update:
            tqdm.write("PN_LSTM: update rollout baseline model.")
            self.baseline_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self.baseline_model.eval()
            for p in self.baseline_model.parameters():
                p.requires_grad_(False)

        return {
            "baseline_updated": float(should_update),
            "baseline_mean_policy_reward": float(policy_rewards.mean()),
            "baseline_mean_baseline_reward": float(baseline_rewards.mean()),
            "baseline_mean_diff": mean_diff,
            "baseline_t_stat": float(t_stat),
            "baseline_eval_n": int(n),
        }


    @torch.no_grad()
    def maybe_update_baseline_from_eval(
        self,
        eval_details: List[Dict[str, Any]],
        loader,
        max_instances: Optional[int] = None,
    ) -> Dict[str, float]:
        if max_instances is None:
            max_instances = BASELINE_EVAL_SIZE

        policy_rewards = np.asarray(
            [float(x["reward"]) for x in eval_details[:max_instances]],
            dtype=np.float64,
        )
        baseline_rewards = self._rollout_baseline_rewards(loader, max_instances=max_instances)
        return self._baseline_t_test(policy_rewards, baseline_rewards)


    def _extract_vehicle_distances_from_rollout(self, env, result: PNLSTMRolloutResult) -> List[float]:
        if result.vehicle_distances is not None:
            return list(result.vehicle_distances)

        actions = result.actions
        if actions is None or len(actions) == 0:
            return []

        depot = int(getattr(env, "depot_idx", 0))
        dist = getattr(env, "dist")
        if torch.is_tensor(dist):
            dist = dist.detach().cpu().numpy()

        vehicle_distances: List[float] = []
        current_node = depot
        current_route_distance = 0.0

        for action in actions:
            next_node = int(action)
            current_route_distance += float(dist[current_node, next_node])
            if next_node == depot:
                if current_route_distance > 1e-8:
                    vehicle_distances.append(float(current_route_distance))
                current_route_distance = 0.0
            current_node = next_node

        if current_route_distance > 1e-8:
            vehicle_distances.append(float(current_route_distance))

        return vehicle_distances


    @torch.no_grad()
    def evaluate(self, loader, return_details: bool = False):
        self.model.eval()
        all_stats: List[Dict[str, Any]] = []
        global_graph_idx = 0

        for batch_idx, batch in enumerate(loader):
            graphs = batch["graphs"]

            for i in range(len(graphs)):
                env = self._make_env(graphs[i])
                result = self.rollout_with_model(
                    model=self.model,
                    env=env,
                    mode="greedy",
                    keep_graph=False,
                )

                graph_id = self._extract_graph_id_from_df(graphs[i], fallback=global_graph_idx)
                global_graph_idx += 1

                vehicle_distances = self._extract_vehicle_distances_from_rollout(env, result)
                if len(vehicle_distances) > 0:
                    vd = np.asarray(vehicle_distances, dtype=np.float32)
                    num_vehicles = int(len(vd))
                    vehicle_distance_mean = float(vd.mean())
                    vehicle_distance_std = float(vd.std(ddof=0))
                    vehicle_distance_min = float(vd.min())
                    vehicle_distance_max = float(vd.max())
                else:
                    num_vehicles = 0
                    vehicle_distance_mean = 0.0
                    vehicle_distance_std = 0.0
                    vehicle_distance_min = 0.0
                    vehicle_distance_max = 0.0

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
                    "vehicle_used": int(result.vehicle_used or 0),
                    "station_visits": int(result.station_visits or 0),
                    "negative_battery_sum": float(result.negative_battery_sum or 0.0),
                    "termination_reason": result.termination_reason,

                    "num_vehicles": num_vehicles,
                    "vehicle_distance_mean": vehicle_distance_mean,
                    "vehicle_distance_std": vehicle_distance_std,
                    "vehicle_distance_min": vehicle_distance_min,
                    "vehicle_distance_max": vehicle_distance_max,
                })

        if not all_stats:
            empty = {"num_graphs": 0}
            return (empty, []) if return_details else empty

        def values_of(key: str) -> List[float]:
            return [float(x[key]) for x in all_stats if key in x and isinstance(x[key], (int, float, bool, np.number))]

        def mean_of(key: str) -> float:
            values = values_of(key)
            return float(sum(values) / max(len(values), 1)) if values else 0.0

        def min_of(key: str) -> float:
            values = values_of(key)
            return float(min(values)) if values else 0.0

        def max_of(key: str) -> float:
            values = values_of(key)
            return float(max(values)) if values else 0.0

        term_counts: Dict[str, int] = {}
        for x in all_stats:
            key = x.get("termination_reason") or "none"
            term_counts[key] = term_counts.get(key, 0) + 1

        summary: Dict[str, Any] = {
            "num_graphs": int(len(all_stats)),
            "reward": mean_of("reward"),
            "len": mean_of("len"),
            "served_count": mean_of("served_count"),
            "in_time_window_ratio": mean_of("in_time_window_ratio"),
            "positive_battery_ratio": mean_of("positive_battery_ratio"),
            "total_distance": mean_of("total_distance"),
            "total_distance_min": min_of("total_distance"),
            "total_distance_max": max_of("total_distance"),
            "total_distance_mean": mean_of("total_distance"),
            "done": mean_of("done"),
            "done_rate": mean_of("done"),
            "vehicle_used": mean_of("vehicle_used"),
            "station_visits": mean_of("station_visits"),
            "negative_battery_sum": mean_of("negative_battery_sum"),
            "num_vehicles_min": min_of("num_vehicles"),
            "num_vehicles_max": max_of("num_vehicles"),
            "num_vehicles_mean": mean_of("num_vehicles"),
            "vehicle_distance_std_min": min_of("vehicle_distance_std"),
            "vehicle_distance_std_max": max_of("vehicle_distance_std"),
            "vehicle_distance_std_mean": mean_of("vehicle_distance_std"),
            "vehicle_distance_mean_min": min_of("vehicle_distance_mean"),
            "vehicle_distance_mean_max": max_of("vehicle_distance_mean"),
            "vehicle_distance_mean_mean": mean_of("vehicle_distance_mean"),
        }

        for key, value in term_counts.items():
            summary[f"termination_{key}"] = float(value / max(len(all_stats), 1))

        return (summary, all_stats) if return_details else summary