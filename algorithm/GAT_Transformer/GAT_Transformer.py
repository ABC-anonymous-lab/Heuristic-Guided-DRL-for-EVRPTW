import copy
import math
from dataclasses import dataclass
from typing import List, Any, Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm
from algorithm.base_trainer import BaseTrainer


@dataclass
class ReinforceRolloutResult:

    actions:           List[int]
    log_probs:         List[torch.Tensor]
    rewards:           List[float]
    infos:             List[Any]
    served_count:      Any
    total_reward:      float
    total_distance:    float
    in_time_window:    float
    positive_battery:  float
    done:              bool
    vehicle_distances: List[float] = None


BASELINE_EVAL_SIZE = 1000


class GATTransformerTrainer(BaseTrainer):
    """REINFORCE trainer for GATTransformerPolicy. The B graphs of a batch are encoded in
    one batched call; decoding runs graph by graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.baseline_model = copy.deepcopy(self.model).to(self.device)
        self.baseline_model.eval()

        train_cfg = self.cfg["train"]
        self.accum_steps = train_cfg.get("reinforce_accum_steps", 1)
        self._accum_counter = 0


    def _rollout_with_h(self, model, env, h, h_graph, mode="sample", keep_graph=True):
        if h_graph.dim() == 1:
            h_graph = h_graph.unsqueeze(0)

        obs = env.reset()
        done = False

        log_probs = []
        rewards = []
        infos = []
        actions = []
        in_time_window   = 0
        positive_battery = 0

        ctx = torch.enable_grad() if keep_graph else torch.no_grad()

        with ctx:
            while not done:
                mask = env.get_action_mask()

                inst = model.obs_to_instance(obs, device=self.device)
                last_idx = obs["current_node"]

                probs, log_prob_all = model.forward_step(
                    h=h,
                    h_graph=h_graph,
                    obs_inst=inst,
                    last_node_idx=last_idx,
                    mask_np=mask,
                )

                action, log_prob = self.select_action(probs, mode=mode)
                a = int(action.item())
                obs, reward, done, info = env.step(a)

                actions.append(a)
                rewards.append(float(reward))
                infos.append(info)

                if keep_graph and mode == "sample":
                    log_probs.append(log_prob)

                in_time_window   += int(info["step_info"].in_time_window)
                positive_battery += int(info["step_info"].positive_battery)

        num_steps = len(actions)
        return ReinforceRolloutResult(
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            infos=infos,
            served_count=float(env.served_rate),
            total_reward=float(np.sum(rewards)),
            total_distance=float(env.total_distance),
            in_time_window=float(in_time_window / num_steps if num_steps > 0 else 0.0),
            positive_battery=float(positive_battery / num_steps if num_steps > 0 else 0.0),
            done=bool(done),
            vehicle_distances= env.vehicle_distances if hasattr(env, "vehicle_distances") else None,
        )


    def rollout_with_model(self, model, env, mode="sample", keep_graph=True):
        obs = env.reset()
        init_inst = model.obs_to_instance(obs, device=self.device)

        if keep_graph:
            h, h_graph, _, _ = model.encode(init_inst)
        else:
            with torch.no_grad():
                h, h_graph, _, _ = model.encode(init_inst)

        return self._rollout_with_h(
            model=model, env=env, h=h, h_graph=h_graph,
            mode=mode, keep_graph=keep_graph,
        )


    def rollout_greedy(self, env):
        return self.rollout_with_model(self.model, env, "greedy")


    def _compute_reinforce_loss(
        self,
        sample_results: List[ReinforceRolloutResult],
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


    def train_step(self, batch):
        """Sample one route per graph, compute baseline rewards with the baseline model, and take a
        REINFORCE step (gradients accumulated over reinforce_accum_steps batches).
        """
        self.model.train()
        self.baseline_model.eval()

        graphs             = batch["graphs"]
        B = len(graphs)

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

        batch_inst = self.model.obs_list_to_instance_batch(init_obs_list, device=self.device)
        h_batch, h_graph_batch, _, _ = self.model.encode_batch(batch_inst)

        with torch.no_grad():
            batch_inst_b = self.baseline_model.obs_list_to_instance_batch(
                init_obs_list, device=self.device
            )
            h_batch_b, h_graph_batch_b, _, _ = self.baseline_model.encode_batch(batch_inst_b)

        sample_results = []
        baseline_results = []

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


        if self._accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        accum_steps = max(int(self.accum_steps), 1)

        baseline_rewards = [float(r.total_reward) for r in baseline_results]
        loss, _rl_aux = self._compute_reinforce_loss(
            sample_results,
            baseline_rewards,
        )
        advantages = _rl_aux["raw_advantages"]

        (loss / accum_steps).backward()

        self._accum_counter += 1

        if self._accum_counter >= accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_counter = 0

        def avg_sample(key):
            return float(np.mean([getattr(r, key) for r in sample_results]))

        def avg_baseline(key):
            return float(np.mean([getattr(r, key) for r in baseline_results]))

        stats = {
            "loss": float(loss.detach().item()),
            "reward": avg_sample("total_reward"),
            "reward_baseline": avg_baseline("total_reward"),
            "advantage": float(np.mean(advantages)),
            "len": float(np.mean([len(r.actions) for r in sample_results])),
            "served_count": avg_sample("served_count"),
            "total_distance": avg_sample("total_distance"),
            "done": float(np.mean([r.done for r in sample_results])),

            "len_baseline": float(np.mean([len(r.actions) for r in baseline_results])),
            "served_count_baseline": avg_baseline("served_count"),
            "total_distance_baseline": avg_baseline("total_distance"),
            "done_baseline": float(np.mean([r.done for r in baseline_results])),

            "reward_greedy": avg_baseline("total_reward"),
            "len_greedy": float(np.mean([len(r.actions) for r in baseline_results])),
            "served_count_greedy": avg_baseline("served_count"),
            "total_distance_greedy": avg_baseline("total_distance"),

            "in_time_window_ratio": avg_sample("in_time_window"),
            "positive_battery_ratio": 0.0,
            "in_time_window_ratio_greedy": avg_baseline("in_time_window"),
            "positive_battery_ratio_greedy": 0.0,
        }

        env_greedy = self.env_builder(graphs[0])
        greedy_result = self.rollout_greedy(env_greedy)

        viz_payload = {
            "graph_df":        graphs[0],
            "greedy_result":   greedy_result,
        }

        return stats, viz_payload


    @torch.no_grad()
    def _rollout_baseline_rewards(self, loader, max_instances=None):
        self.baseline_model.eval()

        if max_instances is None:
            max_instances = BASELINE_EVAL_SIZE

        baseline_rewards = []
        count = 0

        for batch in loader:
            graphs             = batch["graphs"]

            for i in range(len(graphs)):
                env_base = self.env_builder(graphs[i])

                baseline_result = self.rollout_with_model(
                    model=self.baseline_model, env=env_base, mode="greedy", keep_graph=False,
                )

                baseline_rewards.append(baseline_result.total_reward)

                count += 1
                if count >= max_instances:
                    break
            if count >= max_instances:
                break

        return np.asarray(baseline_rewards, dtype=np.float64)


    def _baseline_t_test(self, policy_rewards, baseline_rewards):
        diff = policy_rewards - baseline_rewards

        mean_diff = float(diff.mean())
        std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0

        if len(diff) <= 1 or std_diff < 1e-12:
            should_update = mean_diff > 0.0
            t_stat = float("inf") if should_update else 0.0
        else:
            t_stat = mean_diff / (std_diff / math.sqrt(len(diff)))
            should_update = t_stat > 1.645

        if should_update:
            tqdm.write("Baseline Model Updated")
            self.baseline_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

        return {
            "baseline_updated": float(should_update),
            "baseline_mean_policy_reward": float(policy_rewards.mean()),
            "baseline_mean_baseline_reward": float(baseline_rewards.mean()),
            "baseline_mean_diff": mean_diff,
            "baseline_t_stat": float(t_stat),
            "baseline_eval_n": int(len(diff)),
        }


    @torch.no_grad()
    def maybe_update_baseline_from_eval(self, eval_details, loader, max_instances=None):
        """Baseline update using the policy rewards already computed by evaluate()."""
        if max_instances is None:
            max_instances = BASELINE_EVAL_SIZE

        policy_rewards = np.asarray(
            [x["reward"] for x in eval_details[:max_instances]], dtype=np.float64
        )
        baseline_rewards = self._rollout_baseline_rewards(loader, max_instances=max_instances)

        n = min(len(policy_rewards), len(baseline_rewards))
        if len(policy_rewards) != len(baseline_rewards):
            policy_rewards = policy_rewards[:n]
            baseline_rewards = baseline_rewards[:n]

        return self._baseline_t_test(policy_rewards, baseline_rewards)


    def _extract_vehicle_distances_from_rollout(self, env, result):
        actions = getattr(result, "actions", None)
        if actions is None or len(actions) == 0:
            return []

        if hasattr(env, "depot_idx"):
            depot_idx = env.depot_idx
            if isinstance(depot_idx, (list, tuple, np.ndarray)):
                depot = int(depot_idx[0])
            else:
                depot = int(depot_idx)
        elif hasattr(env, "node_type") and hasattr(env, "DEPOT"):
            depot_candidates = np.where(env.node_type == env.DEPOT)[0]
            depot = int(depot_candidates[0]) if len(depot_candidates) > 0 else 0
        else:
            depot = 0

        dist = env.dist
        if torch.is_tensor(dist):
            dist = dist.detach().cpu().numpy()

        vehicle_distances = []

        current_node = depot
        current_route_distance = 0.0

        for action in actions:
            next_node = int(action)

            step_distance = float(dist[current_node, next_node])
            current_route_distance += step_distance

            if next_node == depot:
                if current_route_distance > 1e-8:
                    vehicle_distances.append(current_route_distance)
                current_route_distance = 0.0

            current_node = next_node

        if current_route_distance > 1e-8:
            vehicle_distances.append(current_route_distance)

        return vehicle_distances


    @torch.no_grad()
    def evaluate(self, loader, return_details=False):
        self.model.eval()
        all_stats = []

        global_graph_idx = 0

        for batch_idx, batch in enumerate(loader):
            graphs             = batch["graphs"]

            for i in range(len(graphs)):
                env = self.env_builder(graphs[i])

                result = self.rollout_with_model(
                    model=self.model,
                    env=env,
                    mode="greedy",
                    keep_graph=False,
                )

                if "graph_id" in graphs[i].columns:
                    graph_id = graphs[i]["graph_id"].iloc[0]
                    try:
                        graph_id = int(graph_id)
                    except Exception:
                        graph_id = str(graph_id)
                else:
                    graph_id = int(global_graph_idx)

                global_graph_idx += 1

                vehicle_distances = self._extract_vehicle_distances_from_rollout(
                    env=env,
                    result=result,
                )

                if len(vehicle_distances) > 0:
                    vd = np.asarray(vehicle_distances, dtype=np.float32)

                    num_vehicles = int(len(vd))
                    vehicle_distance_mean = float(vd.mean())
                    vehicle_distance_std  = float(vd.std(ddof=0))
                    vehicle_distance_min  = float(vd.min())
                    vehicle_distance_max  = float(vd.max())
                else:
                    num_vehicles = 0
                    vehicle_distance_mean = 0.0
                    vehicle_distance_std  = 0.0
                    vehicle_distance_min  = 0.0
                    vehicle_distance_max  = 0.0

                all_stats.append({
                    "graph_id":               graph_id,
                    "batch_idx":              batch_idx,
                    "reward":                 float(result.total_reward),
                    "len":                    int(len(result.actions)),
                    "served_count":           float(result.served_count),
                    "in_time_window_ratio":   float(result.in_time_window),
                    "positive_battery_ratio": float(result.positive_battery),
                    "total_distance":         float(result.total_distance),
                    "done":                   bool(result.done),

                    "num_vehicles":           num_vehicles,
                    "vehicle_distance_mean":  vehicle_distance_mean,
                    "vehicle_distance_std":   vehicle_distance_std,
                    "vehicle_distance_min":   vehicle_distance_min,
                    "vehicle_distance_max":   vehicle_distance_max,
                })

        if not all_stats:
            empty = {"num_graphs": 0}
            return (empty, []) if return_details else empty

        def values_of(key):
            return [x[key] for x in all_stats if key in x]

        def mean_of(key):
            v = values_of(key)
            return float(sum(v) / max(len(v), 1)) if v else 0.0

        def min_of(key):
            v = values_of(key)
            return float(min(v)) if v else 0.0

        def max_of(key):
            v = values_of(key)
            return float(max(v)) if v else 0.0

        summary = {
            "num_graphs":             int(len(all_stats)),

            "reward":                 mean_of("reward"),
            "len":                    mean_of("len"),
            "served_count":           mean_of("served_count"),
            "in_time_window_ratio":   mean_of("in_time_window_ratio"),
            "positive_battery_ratio": mean_of("positive_battery_ratio"),

            "total_distance":         mean_of("total_distance"),
            "total_distance_min":     min_of("total_distance"),
            "total_distance_max":     max_of("total_distance"),
            "total_distance_mean":    mean_of("total_distance"),

            "done":                   mean_of("done"),
            "done_rate":              mean_of("done"),

            "num_vehicles_min":       min_of("num_vehicles"),
            "num_vehicles_max":       max_of("num_vehicles"),
            "num_vehicles_mean":      mean_of("num_vehicles"),

            "vehicle_distance_std_min":    min_of("vehicle_distance_std"),
            "vehicle_distance_std_max":    max_of("vehicle_distance_std"),
            "vehicle_distance_std_mean":   mean_of("vehicle_distance_std"),

            "vehicle_distance_mean_min":   min_of("vehicle_distance_mean"),
            "vehicle_distance_mean_max":   max_of("vehicle_distance_mean"),
            "vehicle_distance_mean_mean":  mean_of("vehicle_distance_mean"),
        }

        return (summary, all_stats) if return_details else summary