import numpy as np


def reward_function(env, **kwargs):
    step_distance = float(kwargs.get("step_distance", 0.0))
    eps = 1e-8
    # Mask of the previous state (computed in env.step before moving)
    valid_mask = env.last_valid_mask

    # Normalize by the longest feasible move from the previous node
    candidate_dists = env.dist[env.prev_node][valid_mask]
    local_max = candidate_dists.max() if candidate_dists.size > 0 else eps
    local_max = max(local_max, eps)

    norm_distance = float(np.clip(step_distance / local_max, 0.0, 1.0))

    # Distance penalty, scaled by distance_weight (the "w" in the experiment names)
    distance_weight = env.distance_weight
    reward = -distance_weight * norm_distance

    # Serving a customer: +1, plus a bonus for choosing a near customer
    if env.node_type[env.current_node] == env.CUSTOMER:
        base_service_reward = 1.0
        short_distance_bonus = 1.0 * (1.0 - norm_distance)
        reward += base_service_reward + short_distance_bonus

    return reward
