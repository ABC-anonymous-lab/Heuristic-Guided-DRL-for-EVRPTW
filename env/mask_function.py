import numpy as np


def mask_function(env) -> np.ndarray:
    mask = np.ones(env.N, dtype=bool)

    is_depot = env.node_type == env.DEPOT
    is_customer = env.node_type == env.CUSTOMER
    is_station = env.node_type == env.STATION

    depot_idx = getattr(env, "depot_idx", 0)

    # Battery left on arrival at every node
    travel_energy = env.dist[env.current_node] / env.energy_rate
    batt_after_arrival = env.battery - travel_energy
    reachable = batt_after_arrival >= 0

    # No self-loop
    mask[env.current_node] = False

    mask &= reachable

    # Each customer is served at most once
    for i in env.customer_idx:
        if env.visited_customer[i]:
            mask[i] = False

    all_customers_served = all(
        env.visited_customer[i]
        for i in env.customer_idx
    )

    depot_reachable = reachable[depot_idx]

    # All customers served
    if (
        all_customers_served
        and depot_reachable
        and env.current_node != depot_idx
    ):
        mask[:] = False
        mask[depot_idx] = True
        return mask

    # Battery lookahead
    # after serving a customer the vehicle must still be able to reach the nearest station or the depot
    safe_nodes = np.where(is_station | is_depot)[0]

    if len(safe_nodes) == 0:
        raise ValueError("No depot/station found.")

    energy_to_safe = env.dist[:, safe_nodes] / env.energy_rate
    min_escape_energy = np.min(energy_to_safe, axis=1)

    mask[is_customer] &= (
        batt_after_arrival[is_customer]
        >= env.min_battery + min_escape_energy[is_customer]
    )

    # Time windows
    # a customer can only be chosen if the vehicle arrives before its due date
    travel_time = env.dist[env.current_node] / env.speed
    arrival = env.time + travel_time

    mask[is_customer] &= (
        arrival[is_customer] <= env.due[is_customer]
    )

    # The depot is allowed whenever it is reachable
    mask[is_depot] = reachable[is_depot]

    if is_depot[env.current_node]:
        mask[env.current_node] = False

    # Charging rule
    # stations are allowed only when the current battery is at most station_allow_battery_ratio of battery_max
    battery_ratio = env.battery / max(env.battery_max, 1e-8)

    if battery_ratio > env.station_allow_battery_ratio:
        mask[is_station] = False

    return mask