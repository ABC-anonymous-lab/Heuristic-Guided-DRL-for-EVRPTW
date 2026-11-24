import numpy as np
from .helpers import (
    build_graph_df_from_env,
    build_expanded_hint_routes,
    solve_state_expanded_evrp,
    extract_routes_as_original_nodes,
    extract_routes_with_events,
)
from .helpers_vectorized import build_state_expanded_data_model_from_df_vectorized


def ortools_repair_fn(
    env,
    init_actions,
    time_limit_s,
    *,
    balance_coefficient=10,
    vehicle_fixed_cost=0,
    charge_rate=1.0,
    energy_per_km=1.0,
    graph_id=0,
):
    """
    Use the DRL route as a hint for OR-Tools GLS on the state-expanded model and return repaired_actions.
    """
    graph_df = build_graph_df_from_env(
        env,
        graph_id=graph_id,
    )
    num_vehicles = int(env.n_vehicles)
    vehicle_speed = float(env.speed)
    max_battery = int(env.battery_max)
    charged_battery_max = int(env.charged_battery_max)
    target_charge_pct = round(float(charged_battery_max / max_battery), 2)
    battery_levels = [n for n in range(0, max_battery, 3)]
    battery_capacity = int(max_battery)

    data = build_state_expanded_data_model_from_df_vectorized(
        graph_df=graph_df,
        battery_levels=battery_levels,
        num_vehicles=num_vehicles,
        vehicle_speed=vehicle_speed,
        battery_capacity=battery_capacity,
        charge_rate=charge_rate,
        energy_per_km=energy_per_km,
        target_charge_pct=target_charge_pct,
    )

    hint_routes, hint_info = build_expanded_hint_routes(
        data,
        init_actions,
    )

    solver_hint_routes = (
        hint_routes
        if hint_info["valid"]
        else None
    )

    (
        manager,
        routing,
        solution,
        solve_info,
    ) = solve_state_expanded_evrp(
        data,
        time_limit_sec=time_limit_s,
        hint_routes=solver_hint_routes,
        balance_coefficient=balance_coefficient,
        vehicle_fixed_cost=vehicle_fixed_cost,
    )

    info = {
        "status": "failed",
        "reward": None,
        "used_time_limit_s": time_limit_s,
        "hint_valid": hint_info["valid"],
        "hint_invalid_reasons": (
            hint_info["invalid_reasons"]
        ),
        **solve_info,
    }

    if solution is None:
        return list(init_actions), info

    extracted_routes = extract_routes_as_original_nodes(
        data,
        manager,
        routing,
        solution,
    )

    repaired_actions = []
    used_routes = []

    for route_result in extracted_routes:
        route = route_result["node_sequence"]

        if len(route) <= 2:
            continue
        used_routes.append(route)
        if not repaired_actions:
            repaired_actions.extend(route)
        else:
            repaired_actions.extend(route[1:])

    depot_orig_id = int(
        data["nodes"][data["depot"]]["orig_node_id"]
    )

    if not repaired_actions:
        repaired_actions = [depot_orig_id]

    extracted = extract_routes_with_events(
        data,
        manager,
        routing,
        solution,
    )

    distance_dim = routing.GetDimensionOrDie(
        "Distance"
    )

    route_distances = []
    for vehicle_id in range(num_vehicles):
        end_index = routing.End(vehicle_id)

        distance_scaled = solution.Value(
            distance_dim.CumulVar(end_index)
        )

        route_distances.append(
            distance_scaled / solve_info["distance_scale"]
        )

    nonzero_distances = [
        distance
        for distance in route_distances
        if distance > 1e-9
    ]

    info.update({
        "status": "success",
        "initial_assignment_valid": (
            solve_info["initial_assignment_valid"]
        ),
        "objective_value": solution.ObjectiveValue(),
        "used_routes": used_routes,
        "num_used_vehicles": len(used_routes),
        "route_distances": route_distances,
        "route_distance_mean": (
            float(np.mean(nonzero_distances))
            if nonzero_distances else 0.0
        ),
        "route_distance_std": (
            float(np.std(nonzero_distances))
            if nonzero_distances else 0.0
        ),
        "route_distance_max": (
            float(np.max(nonzero_distances))
            if nonzero_distances else 0.0
        ),
        "route_distance_min": (
            float(np.min(nonzero_distances))
            if nonzero_distances else 0.0
        ),
        "global_summary": extracted["global_summary"],
    })

    return repaired_actions, info