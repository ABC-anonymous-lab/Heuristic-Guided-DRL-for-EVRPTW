import bisect
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


SCALE = 1000


def build_graph_df_from_env(env, graph_id=0):
    if hasattr(env, "xy_km"):
        xy_km = np.asarray(env.xy_km)
    else:
        raise AttributeError(
            "env has no xy_km (node coordinates in km)"
        )

    node_ids = (
        np.asarray(env.node_id)
        if hasattr(env, "node_id")
        else np.arange(env.N)
    )

    return pd.DataFrame({
        "graph_id": np.full(env.N, graph_id, dtype=int),
        "node_id": node_ids.astype(int),
        "type": np.asarray(env.node_type).astype(int),
        "x_km": xy_km[:, 0].astype(float),
        "y_km": xy_km[:, 1].astype(float),
        "service_time": np.asarray(env.service).astype(float),
        "ready_time": np.asarray(env.ready).astype(float),
        "due_date": np.asarray(env.due).astype(float),
    })


def solve_state_expanded_evrp(
    data,
    time_limit_sec=10,
    hint_routes=None,
    balance_coefficient=0,
    vehicle_fixed_cost=0,
):
    manager = pywrapcp.RoutingIndexManager(
        data["num_nodes"],
        data["num_vehicles"],
        data["depot"],
    )
    routing = pywrapcp.RoutingModel(manager)

    BIG_M = 10 ** 9
    DIST_SCALE = 1000

    allowed_arcs = data["allowed_arcs"]
    arc_time_dict = data["arc_time_dict"]
    nodes = data["nodes"]
    depot = data["depot"]
    vehicle_speed = float(data["vehicle_speed"])

    N = int(data["num_nodes"])

    real_dist = np.asarray(data["real_distance_matrix"], dtype=float)
    energy_int = np.asarray(data["energy_matrix_int"], dtype=np.int64)

    batt = np.fromiter(
        (int(nodes[k]["battery_level"]) for k in range(N)),
        dtype=np.int64, count=N,
    )
    svc = np.fromiter(
        (int(nodes[k]["service_time"]) for k in range(N)),
        dtype=np.int64, count=N,
    )

    depot_ok = energy_int[:, depot] <= batt

    allowed_mask = np.zeros((N, N), dtype=bool)
    if allowed_arcs:
        aidx = np.array(list(allowed_arcs), dtype=np.int64)
        allowed_mask[aidx[:, 0], aidx[:, 1]] = True

    dist_scaled = np.rint(real_dist * DIST_SCALE).astype(np.int64)
    D = np.where(allowed_mask, dist_scaled, BIG_M)

    ret_dist = np.rint(real_dist[:, depot] * DIST_SCALE).astype(np.int64)
    D[:, depot] = np.where(depot_ok, ret_dist, BIG_M)
    D[depot, depot] = 0

    T = np.full((N, N), BIG_M, dtype=np.int64)
    if arc_time_dict:
        ks = np.array(list(arc_time_dict.keys()), dtype=np.int64)
        vs = np.fromiter(arc_time_dict.values(), dtype=np.int64,
                         count=len(arc_time_dict))
        T[ks[:, 0], ks[:, 1]] = vs

    travel_to_depot = np.rint(real_dist[:, depot] / vehicle_speed).astype(np.int64)
    T[:, depot] = np.where(depot_ok, svc + travel_to_depot, BIG_M)
    T[depot, depot] = 0

    dist_cb_idx = routing.RegisterTransitMatrix(D.tolist())
    time_cb_idx = routing.RegisterTransitMatrix(T.tolist())

    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    if vehicle_fixed_cost > 0:
        fixed_cost_scaled = int(round(vehicle_fixed_cost * DIST_SCALE))
        for vehicle_id in range(data["num_vehicles"]):
            routing.SetFixedCostOfVehicle(fixed_cost_scaled, vehicle_id)

    max_arc_distance = float(np.max(real_dist))
    max_route_distance = int(
        np.ceil(max_arc_distance * max(N, 1) * DIST_SCALE)
    )

    routing.AddDimension(
        dist_cb_idx,
        0,
        max_route_distance,
        True,
        "Distance",
    )
    distance_dim = routing.GetDimensionOrDie("Distance")
    if balance_coefficient > 0:
        distance_dim.SetGlobalSpanCostCoefficient(int(balance_coefficient))

    routing.AddDimension(
        time_cb_idx,
        3600,
        200000,
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    for node_idx in range(N):
        index = manager.NodeToIndex(node_idx)
        ready = int(nodes[node_idx]["ready_time"])
        due = int(nodes[node_idx]["due_date"])
        time_dim.CumulVar(index).SetRange(ready, due)

    depot_ready = int(nodes[depot]["ready_time"])
    depot_due = int(nodes[depot]["due_date"])
    for vehicle_id in range(data["num_vehicles"]):
        start = routing.Start(vehicle_id)
        end = routing.End(vehicle_id)
        time_dim.CumulVar(start).SetRange(depot_ready, depot_due)
        time_dim.CumulVar(end).SetRange(depot_ready, 200000)
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(start))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(end))

    customer_drop_penalty = int(1_000_000 * DIST_SCALE)
    for state_indices in data["customer_groups"].values():
        routing.AddDisjunction(
            [manager.NodeToIndex(i) for i in state_indices],
            customer_drop_penalty,
        )

    for state_indices in data["charger_groups"].values():
        for expanded_idx in state_indices:
            routing.AddDisjunction([manager.NodeToIndex(expanded_idx)], 0)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(max(1, int(time_limit_sec)))
    search_parameters.log_search = False

    routing.CloseModelWithParameters(search_parameters)

    initial_assignment = None
    if hint_routes is not None:
        try:
            initial_assignment = routing.ReadAssignmentFromRoutes(
                hint_routes, True,
            )
        except Exception:
            initial_assignment = None

    if initial_assignment is not None:
        solution = routing.SolveFromAssignmentWithParameters(
            initial_assignment, search_parameters,
        )
    else:
        solution = routing.SolveWithParameters(search_parameters)

    solve_info = {
        "initial_assignment_valid": initial_assignment is not None,
        "balance_coefficient": balance_coefficient,
        "vehicle_fixed_cost": vehicle_fixed_cost,
        "distance_scale": DIST_SCALE,
    }

    return manager, routing, solution, solve_info


def extract_routes_with_events(data, manager, routing, solution):
    if solution is None:
        return {
            "routes": [],
            "global_summary": {
                "total_distance": None,
                "total_travel_time": None,
                "total_charge_time": None,
                "total_charge_amount": None,
                "num_charges": None,
                "skipped_customers": None,
                "tardiness_min": None,
                "charge_time_pct_of_travel_time": None,
            },
        }

    nodes         = data["nodes"]
    real_dist     = data["real_distance_matrix"]
    vehicle_speed = data["vehicle_speed"]
    energy_per_km = data["energy_per_km"]
    charge_rate   = data["charge_rate"]
    target_batt   = float(data["target_charge_level"])
    depot         = data["depot"]
    depot_orig_id = int(nodes[depot]["orig_node_id"])

    routes = []
    served_customers = set()

    g_distance = g_travel_time = g_charge_time = g_charge_amount = 0.0
    g_num_charges = 0
    g_tardiness = 0.0

    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)

        cur_time      = float(nodes[depot]["ready_time"])
        cur_batt_real = float(data["battery_capacity"])

        events = []
        node_sequence = [depot_orig_id]

        r_dist = r_travel = r_charge_time = r_charge_amount = 0.0
        r_num_charges = 0
        r_tardiness = 0.0

        events.append({
            "orig_node_id":      depot_orig_id,
            "state_node":        nodes[depot]["name"],
            "node_type":         0,
            "arrival_time":      cur_time,
            "service_start_time": cur_time,
            "departure_time":    cur_time,
            "arrival_battery":   cur_batt_real,
            "departure_battery": cur_batt_real,
            "charge_amount":     0.0,
            "charge_time":       0.0,
            "travel_from_prev":  0.0,
            "travel_time_from_prev": 0.0,
            "wait_time":         0.0,
            "service_time":      0.0,
            "due_date":          float(nodes[depot]["due_date"]),
            "tardiness":         0.0,
        })

        while not routing.IsEnd(idx):
            next_idx = solution.Value(routing.NextVar(idx))

            if routing.IsStart(idx):
                i, ni, from_kind = depot, nodes[depot], "depot_start"
            else:
                i  = manager.IndexToNode(idx)
                ni = nodes[i]
                from_kind = ni["kind"]

            charge_amt  = 0.0
            charge_time_here = 0.0
            if from_kind == "charger_state":
                if target_batt > cur_batt_real:
                    charge_amt        = target_batt - cur_batt_real
                    charge_time_here  = charge_amt * charge_rate
                    cur_batt_real     = target_batt
                    cur_time         += charge_time_here
                    r_charge_amount  += charge_amt
                    r_charge_time    += charge_time_here
                    if charge_amt > 1e-9:
                        r_num_charges += 1

                    assert events[-1]["orig_node_id"] == ni["orig_node_id"], (
                        f"Charger event node mismatch: "
                        f"events[-1]={events[-1]['orig_node_id']} vs ni={ni['orig_node_id']}"
                    )
                    events[-1]["charge_amount"]    = charge_amt
                    events[-1]["charge_time"]      = charge_time_here
                    events[-1]["departure_time"]   = cur_time
                    events[-1]["departure_battery"] = cur_batt_real

            dep_time = cur_time
            dep_batt = cur_batt_real

            if routing.IsEnd(next_idx):
                back_km     = float(real_dist[i][depot])
                back_energy = back_km * energy_per_km
                back_travel = back_km / vehicle_speed

                arr_time = dep_time + back_travel
                arr_batt = dep_batt - back_energy

                r_dist   += back_km
                r_travel += back_travel

                node_sequence.append(depot_orig_id)
                events.append({
                    "orig_node_id":      depot_orig_id,
                    "state_node":        "depot_return",
                    "node_type":         0,
                    "arrival_time":      arr_time,
                    "service_start_time": arr_time,
                    "departure_time":    arr_time,
                    "arrival_battery":   arr_batt,
                    "departure_battery": arr_batt,
                    "charge_amount":     0.0,
                    "charge_time":       0.0,
                    "travel_from_prev":  back_km,
                    "travel_time_from_prev": back_travel,
                    "wait_time":         0.0,
                    "service_time":      0.0,
                    "due_date":          float(nodes[depot]["due_date"]),
                    "tardiness":         0.0,
                })
                break

            j  = manager.IndexToNode(next_idx)
            nj = nodes[j]

            travel_km   = float(real_dist[i][j])
            energy_used = travel_km * energy_per_km
            travel_time = travel_km / vehicle_speed

            arr_time_raw = dep_time + travel_time
            arr_batt     = dep_batt - energy_used

            ready_j   = float(nj["ready_time"])
            due_j     = float(nj["due_date"])
            service_j = float(nj["service_time"])

            wait_time     = max(0.0, ready_j - arr_time_raw)
            service_start = arr_time_raw + wait_time
            finish_service = service_start + service_j
            tardiness     = max(0.0, service_start - due_j)

            r_dist      += travel_km
            r_travel    += travel_time
            r_tardiness += tardiness

            events.append({
                "orig_node_id":      int(nj["orig_node_id"]),
                "state_node":        nj["name"],
                "node_type":         int(nj["orig_type"]),
                "arrival_time":      arr_time_raw,
                "service_start_time": service_start,
                "departure_time":    finish_service,
                "arrival_battery":   arr_batt,
                "departure_battery": arr_batt,
                "charge_amount":     0.0,
                "charge_time":       0.0,
                "travel_from_prev":  travel_km,
                "travel_time_from_prev": travel_time,
                "wait_time":         wait_time,
                "service_time":      service_j,
                "due_date":          due_j,
                "tardiness":         tardiness,
            })

            node_sequence.append(int(nj["orig_node_id"]))
            if int(nj["orig_type"]) == 1:
                served_customers.add(int(nj["orig_node_id"]))

            cur_time      = finish_service
            cur_batt_real = arr_batt
            idx = next_idx

        charge_pct = (100.0 * r_charge_time / r_travel) if r_travel > 1e-9 else 0.0

        routes.append({
            "vehicle_id":    v,
            "node_sequence": node_sequence,
            "events":        events,
            "summary": {
                "distance":                      r_dist,
                "travel_time":                   r_travel,
                "num_charges":                   r_num_charges,
                "total_charge_amount":           r_charge_amount,
                "total_charge_time":             r_charge_time,
                "charge_time_pct_of_travel_time": charge_pct,
                "tardiness_min":                 r_tardiness,
            },
        })

        g_distance     += r_dist
        g_travel_time  += r_travel
        g_charge_time  += r_charge_time
        g_charge_amount += r_charge_amount
        g_num_charges  += r_num_charges
        g_tardiness    += r_tardiness

    all_customers = {
        int(n["orig_node_id"]) for n in nodes if int(n["orig_type"]) == 1
    }
    skipped_ids = sorted(list(all_customers - served_customers))

    global_charge_pct = (
        100.0 * g_charge_time / g_travel_time if g_travel_time > 1e-9 else 0.0
    )

    return {
        "routes": routes,
        "global_summary": {
            "total_distance":                g_distance,
            "total_travel_time":             g_travel_time,
            "num_charges":                   g_num_charges,
            "total_charge_amount":           g_charge_amount,
            "total_charge_time":             g_charge_time,
            "charge_time_pct_of_travel_time": global_charge_pct,
            "tardiness_min":                 g_tardiness,
            "skipped_customers":             len(skipped_ids),
            "skipped_customer_ids":          skipped_ids,
        },
    }


def extract_routes_as_original_nodes(data, manager, routing, solution):
    """Solution as a flat action sequence of original node ids (depot = 0 between routes)."""
    results = []
    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        route_nodes  = [data["nodes"][data["depot"]]["orig_node_id"]]
        route_states = []
        visited_customers = set()

        while not routing.IsEnd(idx):
            next_idx = solution.Value(routing.NextVar(idx))
            if routing.IsEnd(next_idx):
                break
            j = manager.IndexToNode(next_idx)
            node_info = data["nodes"][j]
            orig_id   = node_info["orig_node_id"]
            node_type = node_info["orig_type"]
            route_states.append(node_info["name"])
            if node_type == 1 and orig_id not in visited_customers:
                route_nodes.append(orig_id)
                visited_customers.add(orig_id)
            elif node_type == 2:
                route_nodes.append(orig_id)
            idx = next_idx

        route_nodes.append(data["nodes"][data["depot"]]["orig_node_id"])
        results.append({
            "vehicle_id":    v,
            "node_sequence": route_nodes,
            "state_sequence": route_states,
        })
    return results


def round_down_level(x, levels):
    if x < 0:
        return None
    idx = bisect.bisect_right(levels, x)
    if idx == 0:
        return None
    return levels[idx - 1]


def build_expanded_hint_routes(data, init_actions):
    nodes = data["nodes"]
    depot = data["depot"]
    num_vehicles = data["num_vehicles"]
    target_battery = int(data["target_charge_level"])
    battery_capacity = int(data["battery_capacity"])
    battery_levels = data["battery_levels"]
    energy_matrix = data["energy_matrix_int"]

    depot_orig_id = int(nodes[depot]["orig_node_id"])

    customer_state_map = {}
    station_clone_map = {}
    orig_type_map = {}

    for expanded_idx, node in enumerate(nodes):
        orig_id = int(node["orig_node_id"])
        orig_type = int(node["orig_type"])

        orig_type_map[orig_id] = orig_type

        if node["kind"] == "customer_state":
            key = (
                orig_id,
                int(node["battery_level"]),
            )
            customer_state_map[key] = expanded_idx

        elif node["kind"] == "charger_state":
            station_clone_map.setdefault(
                orig_id,
                [],
            ).append(expanded_idx)

    station_clone_cursor = {
        station_id: 0
        for station_id in station_clone_map
    }

    hint_routes = []
    current_route = []

    current_expanded_idx = depot
    current_battery = battery_capacity

    seen_customers = set()
    invalid_reasons = []

    def close_current_route():
        nonlocal current_route
        nonlocal current_expanded_idx
        nonlocal current_battery

        if current_route:
            hint_routes.append(current_route)
            current_route = []

        current_expanded_idx = depot
        current_battery = battery_capacity

    for raw_action in init_actions:
        orig_id = int(raw_action)

        if orig_id == depot_orig_id:
            close_current_route()
            continue

        if orig_id not in orig_type_map:
            invalid_reasons.append(
                f"unknown original node id: {orig_id}"
            )
            continue

        orig_type = orig_type_map[orig_id]

        if orig_type == 1:
            if orig_id in seen_customers:
                continue

            candidate_states = [
                expanded_idx
                for (customer_id, _), expanded_idx
                in customer_state_map.items()
                if customer_id == orig_id
            ]

            if not candidate_states:
                invalid_reasons.append(
                    f"customer {orig_id} has no expanded states"
                )
                continue

            representative_idx = candidate_states[0]
            energy_used = int(
                energy_matrix[
                    current_expanded_idx
                ][
                    representative_idx
                ]
            )

            arrival_battery = current_battery - energy_used
            mapped_battery = round_down_level(
                arrival_battery,
                battery_levels,
            )

            if mapped_battery is None:
                invalid_reasons.append(
                    f"battery infeasible: "
                    f"{nodes[current_expanded_idx]['name']} "
                    f"-> customer {orig_id}, "
                    f"current={current_battery}, "
                    f"energy={energy_used}"
                )
                continue

            expanded_idx = customer_state_map.get(
                (orig_id, mapped_battery)
            )
            if expanded_idx is None:
                invalid_reasons.append(
                    f"missing customer state: "
                    f"customer={orig_id}, "
                    f"battery={mapped_battery}"
                )
                continue
            if (
                current_expanded_idx,
                expanded_idx,
            ) not in data["allowed_arcs"]:
                invalid_reasons.append(
                    f"illegal expanded arc: "
                    f"{nodes[current_expanded_idx]['name']} "
                    f"-> {nodes[expanded_idx]['name']}"
                )
                continue

            current_route.append(expanded_idx)
            current_expanded_idx = expanded_idx
            current_battery = mapped_battery
            seen_customers.add(orig_id)

        elif orig_type == 2:
            clones = station_clone_map.get(orig_id, [])

            cursor = station_clone_cursor.get(orig_id, 0)

            if cursor >= len(clones):
                invalid_reasons.append(
                    f"station {orig_id} has no unused clone"
                )
                continue

            expanded_idx = clones[cursor]
            energy_used = int(
                energy_matrix[
                    current_expanded_idx
                ][
                    expanded_idx
                ]
            )

            arrival_battery = current_battery - energy_used

            if arrival_battery < 0:
                invalid_reasons.append(
                    f"battery infeasible before station "
                    f"{orig_id}: current={current_battery}, "
                    f"energy={energy_used}"
                )
                continue

            if (
                current_expanded_idx,
                expanded_idx,
            ) not in data["allowed_arcs"]:
                invalid_reasons.append(
                    f"illegal charger arc: "
                    f"{nodes[current_expanded_idx]['name']} "
                    f"-> {nodes[expanded_idx]['name']}"
                )
                continue

            station_clone_cursor[orig_id] += 1

            current_route.append(expanded_idx)
            current_expanded_idx = expanded_idx
            current_battery = target_battery

    close_current_route()

    if len(hint_routes) > num_vehicles:
        base_routes = hint_routes[:num_vehicles]

        for extra_route in hint_routes[num_vehicles:]:
            target = min(
                range(num_vehicles),
                key=lambda v: len(base_routes[v]),
            )
            base_routes[target].extend(extra_route)

        hint_routes = base_routes

    while len(hint_routes) < num_vehicles:
        hint_routes.append([])

    return hint_routes, {
        "valid": len(invalid_reasons) == 0,
        "invalid_reasons": invalid_reasons,
    }