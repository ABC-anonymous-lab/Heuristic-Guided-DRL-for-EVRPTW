import os
import csv
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from helper import METRIC_COLUMNS, extract_single_graph_metrics, summarize_graph_metrics
from utils.plot_functions import (
    plot_all_vehicle_battery_and_distance,
    plot_all_vehicle_distance_and_time_profiles,
    plot_all_vehicle_figures,
    plot_all_vehicle_timeline_and_gantt,
)


CRITERIA = "ortools" 


def round_down_level(x, levels):
    if x < 0:
        return None
    feasible = [lv for lv in levels if lv <= x]
    if not feasible:
        return None
    return max(feasible)


# -------------
# Model
# -------------
def build_state_expanded_data_model_from_df(
    graph_df: pd.DataFrame,
    num_vehicles: int,
    vehicle_speed: float,        # km per minute
    battery_capacity: int,
    battery_levels,
    charge_rate: float,          # minutes per battery unit
    energy_per_km: float,        # battery units per km
    target_charge_pct: float,    # charge-to level as a fraction of battery_capacity
):
    required_cols = [
        "graph_id", "node_id", "type",
        "x_km", "y_km",
        "service_time",
        "ready_time", "due_date",
    ]
    missing = [c for c in required_cols if c not in graph_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    graph_df = graph_df.copy().reset_index(drop=True)
    graph_df["type"] = graph_df["type"].astype(int)
    graph_df["node_id"] = graph_df["node_id"].astype(int)

    depot_df   = graph_df[graph_df["type"] == 0].copy()
    cust_df    = graph_df[graph_df["type"] == 1].copy()
    station_df = graph_df[graph_df["type"] == 2].copy()

    if len(depot_df) != 1:
        raise ValueError(f"Each graph needs exactly one depot, found {len(depot_df)}")

    depot_row = depot_df.iloc[0]

    # Copies per station, so that several vehicles can charge at the same station
    station_clone_limit = max(num_vehicles, 3)

    data = {}
    data["graph_id"]            = int(graph_df["graph_id"].iloc[0])
    data["num_vehicles"]        = int(num_vehicles)
    data["vehicle_speed"]       = float(vehicle_speed)
    data["battery_capacity"]    = int(battery_capacity)
    data["battery_levels"]      = list(sorted(set(int(x) for x in battery_levels)))
    data["charge_rate"]         = float(charge_rate)
    data["energy_per_km"]       = float(energy_per_km)
    data["target_charge_level"] = int(round(data["battery_capacity"] * target_charge_pct))

    expanded_nodes = []

    expanded_nodes.append({
        "name": f"depot_start_{int(depot_row['node_id'])}",
        "kind": "depot_start",
        "orig_node_id": int(depot_row["node_id"]),
        "orig_type": 0,
        "x_km": float(depot_row["x_km"]),
        "y_km": float(depot_row["y_km"]),
        "service_time": int(round(float(depot_row["service_time"]))),
        "ready_time":   int(round(float(depot_row["ready_time"]))),
        "due_date":     int(round(float(depot_row["due_date"]))),
        "battery_level": data["battery_capacity"],
        "physical_station_id": None,
        "group_key": "DEPOT_START",
    })

    for _, row in cust_df.iterrows():
        for b in data["battery_levels"]:
            expanded_nodes.append({
                "name": f"cust_{int(row['node_id'])}_b{b}",
                "kind": "customer_state",
                "orig_node_id": int(row["node_id"]),
                "orig_type": 1,
                "x_km": float(row["x_km"]),
                "y_km": float(row["y_km"]),
                "service_time": int(round(float(row["service_time"]))),
                "ready_time":   int(round(float(row["ready_time"]))),
                "due_date":     int(round(float(row["due_date"]))),
                "battery_level": int(b),
                "physical_station_id": None,
                "group_key": f"CUST_{int(row['node_id'])}",
            })

    for _, row in station_df.iterrows():
        sid = int(row["node_id"])
        for c in range(station_clone_limit):
            expanded_nodes.append({
                "name": f"chg_{sid}_clone{c}",
                "kind": "charger_state",
                "orig_node_id": sid,
                "orig_type": 2,
                "x_km": float(row["x_km"]),
                "y_km": float(row["y_km"]),
                "service_time": 0,
                "ready_time":   int(round(float(row["ready_time"]))),
                "due_date":     int(round(float(row["due_date"]))),
                "battery_level": int(data["target_charge_level"]),
                "physical_station_id": sid,
                "clone_id": c,
                "group_key": f"CHG_{sid}",
            })

    data["nodes"]     = expanded_nodes
    data["num_nodes"] = len(expanded_nodes)
    data["depot"]     = 0

    coords = np.array([[n["x_km"], n["y_km"]] for n in expanded_nodes], dtype=float)
    diff   = coords[:, None, :] - coords[None, :, :]
    euclid = np.linalg.norm(diff, axis=-1)                       # km, (N, N)
    energy_matrix_int = np.rint(euclid * data["energy_per_km"]).astype(int)

    data["real_distance_matrix"] = euclid.tolist()
    data["energy_matrix_int"]    = energy_matrix_int.tolist()

    # ------------------------------------------------------------------
    # Feasible arcs and their transit times (minutes).
    # arc_time[(i, j)] = service time at i + travel time,
    # so the Time cumul of a station is the time charging finishes.
    # ------------------------------------------------------------------
    allowed_arcs: set = set()
    arc_time_dict: dict = {}

    target_batt = int(data["target_charge_level"])

    for i, ni in enumerate(expanded_nodes):
        bi = int(ni["battery_level"])
        from_service = int(ni["service_time"])

        for j in range(1, data["num_nodes"]):
            if i == j:
                continue
            nj = expanded_nodes[j]

            e_ij       = int(energy_matrix_int[i][j])
            travel_min = int(round(euclid[i][j] / data["vehicle_speed"]))
            arrival_batt = bi - e_ij

            if nj["kind"] == "customer_state":
                # No jumps between copies of the same customer
                if ni["orig_type"] == 1 and ni["orig_node_id"] == nj["orig_node_id"]:
                    continue
                # Only the copy that matches the arrival battery level
                if round_down_level(arrival_batt, data["battery_levels"]) != int(nj["battery_level"]):
                    continue
                arc_time_dict[(i, j)] = from_service + travel_min

            elif nj["kind"] == "charger_state":
                if arrival_batt < 0 or arrival_batt > target_batt:
                    continue
                # No jumps between copies of the same station
                if ni["kind"] == "charger_state" and ni["physical_station_id"] == nj["physical_station_id"]:
                    continue
                charge_min = int(round((target_batt - arrival_batt) * data["charge_rate"]))
                arc_time_dict[(i, j)] = from_service + travel_min + charge_min

            else:
                continue
            allowed_arcs.add((i, j))

    data["allowed_arcs"]  = allowed_arcs
    data["arc_time_dict"] = arc_time_dict

    # Node groups for the disjunctions
    customer_groups = {}
    charger_groups  = {}
    for idx, node in enumerate(expanded_nodes):
        if node["kind"] == "customer_state":
            customer_groups.setdefault(node["group_key"], []).append(idx)
        elif node["kind"] == "charger_state":
            charger_groups.setdefault(node["group_key"], []).append(idx)

    data["customer_groups"] = customer_groups
    data["charger_groups"]  = charger_groups

    return data


def solve_state_expanded_evrp(data, time_limit_sec, balance_coefficient=0, vehicle_fixed_cost=0):
    manager = pywrapcp.RoutingIndexManager(
        data["num_nodes"],
        data["num_vehicles"],
        data["depot"],
    )
    routing = pywrapcp.RoutingModel(manager)

    BIG_M = 10 ** 9
    DIST_SCALE = 1000

    allowed_arcs  = data["allowed_arcs"]
    arc_time_dict = data["arc_time_dict"]
    nodes         = data["nodes"]
    depot         = data["depot"]
    energy_int    = data["energy_matrix_int"]
    real_dist     = data["real_distance_matrix"]
    vehicle_speed = data["vehicle_speed"]

    def _back_to_depot_ok(from_node):
        return int(energy_int[from_node][depot]) <= int(nodes[from_node]["battery_level"])

    def distance_callback(from_index, to_index):
        if routing.IsEnd(to_index):
            if routing.IsStart(from_index):
                return 0
            fn = manager.IndexToNode(from_index)
            if not _back_to_depot_ok(fn):
                return BIG_M
            return int(round(real_dist[fn][depot] * DIST_SCALE))

        if routing.IsEnd(from_index):
            return BIG_M

        fn = manager.IndexToNode(from_index)
        tn = manager.IndexToNode(to_index)
        if (fn, tn) not in allowed_arcs:
            return BIG_M
        return int(round(real_dist[fn][tn] * DIST_SCALE))

    dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    if vehicle_fixed_cost > 0:
        fixed_cost_scaled = int(round(vehicle_fixed_cost * DIST_SCALE))
        for v in range(data["num_vehicles"]):
            routing.SetFixedCostOfVehicle(fixed_cost_scaled, v)

    # Distance dimension, used for route balancing
    max_arc_distance = max(max(row) for row in real_dist)
    max_route_distance = int(np.ceil(max_arc_distance * max(data["num_nodes"], 1) * DIST_SCALE))
    routing.AddDimension(dist_cb_idx, 0, max_route_distance, True, "Distance")
    distance_dim = routing.GetDimensionOrDie("Distance")
    if balance_coefficient > 0:
        distance_dim.SetGlobalSpanCostCoefficient(int(balance_coefficient))

    def time_callback(from_index, to_index):
        if routing.IsEnd(to_index):
            if routing.IsStart(from_index):
                return 0
            fn = manager.IndexToNode(from_index)
            if not _back_to_depot_ok(fn):
                return BIG_M
            travel_min = int(round(real_dist[fn][depot] / vehicle_speed))
            return int(nodes[fn]["service_time"]) + travel_min

        if routing.IsEnd(from_index):
            return BIG_M

        fn = manager.IndexToNode(from_index)
        tn = manager.IndexToNode(to_index)
        if (fn, tn) not in allowed_arcs:
            return BIG_M
        return arc_time_dict[(fn, tn)]

    time_cb_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_cb_idx,
        
        3600,  # waiting allowed
        200000,
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    for node_idx in range(data["num_nodes"]):
        idx = manager.NodeToIndex(node_idx)
        time_dim.CumulVar(idx).SetRange(int(nodes[node_idx]["ready_time"]),
                                        int(nodes[node_idx]["due_date"]))

    depot_ready = int(nodes[depot]["ready_time"])
    depot_due   = int(nodes[depot]["due_date"])
    for v in range(data["num_vehicles"]):
        s = routing.Start(v)
        e = routing.End(v)
        time_dim.CumulVar(s).SetRange(depot_ready, depot_due)
        time_dim.CumulVar(e).SetRange(depot_ready, 200000)
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(s))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(e))

    # Exactly one copy of each customer
    MUST_VISIT_PENALTY = int(1_000_000 * DIST_SCALE)
    for state_indices in data["customer_groups"].values():
        routing.AddDisjunction([manager.NodeToIndex(i) for i in state_indices], MUST_VISIT_PENALTY)

    # Station copies are optional; each copy can be used once
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

    solution = routing.SolveWithParameters(search_parameters)
    return manager, routing, solution


# -------------
# Solution
# -------------

def extract_routes_with_events(data, manager, routing, solution):
    nodes         = data["nodes"]
    real_dist     = data["real_distance_matrix"]
    vehicle_speed = data["vehicle_speed"]
    energy_per_km = data["energy_per_km"]
    charge_rate   = data["charge_rate"]
    target_batt   = float(data["target_charge_level"])
    depot         = data["depot"]
    depot_orig_id = int(nodes[depot]["orig_node_id"])

    routes = []
    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)

        cur_time      = float(nodes[depot]["ready_time"])
        cur_batt_real = float(data["battery_capacity"])

        events = [{
            "orig_node_id":      depot_orig_id,
            "node_type":         0,
            "arrival_time":      cur_time,
            "service_start_time": cur_time,
            "departure_time":    cur_time,
            "arrival_battery":   cur_batt_real,
            "departure_battery": cur_batt_real,
            "charge_time":       0.0,
            "travel_from_prev":  0.0,
            "travel_time_from_prev": 0.0,
            "wait_time":         0.0,
            "service_time":      0.0,
            "due_date":          float(nodes[depot]["due_date"]),
        }]

        while not routing.IsEnd(idx):
            next_idx = solution.Value(routing.NextVar(idx))

            if routing.IsStart(idx):
                i, ni, from_kind = depot, nodes[depot], "depot_start"
            else:
                i  = manager.IndexToNode(idx)
                ni = nodes[i]
                from_kind = ni["kind"]

            # Charge at the station we are leaving, then update its event
            if from_kind == "charger_state" and target_batt > cur_batt_real:
                charge_amt       = target_batt - cur_batt_real
                charge_time_here = charge_amt * charge_rate
                cur_batt_real    = target_batt
                cur_time        += charge_time_here

                assert events[-1]["orig_node_id"] == ni["orig_node_id"]
                events[-1]["charge_time"]       = charge_time_here
                events[-1]["departure_time"]    = cur_time
                events[-1]["departure_battery"] = cur_batt_real

            dep_time = cur_time
            dep_batt = cur_batt_real

            # Back to the depot
            if routing.IsEnd(next_idx):
                back_km     = float(real_dist[i][depot])
                back_travel = back_km / vehicle_speed
                arr_time = dep_time + back_travel
                arr_batt = dep_batt - back_km * energy_per_km

                events.append({
                    "orig_node_id":      depot_orig_id,
                    "node_type":         0,
                    "arrival_time":      arr_time,
                    "service_start_time": arr_time,
                    "departure_time":    arr_time,
                    "arrival_battery":   arr_batt,
                    "departure_battery": arr_batt,
                    "charge_time":       0.0,
                    "travel_from_prev":  back_km,
                    "travel_time_from_prev": back_travel,
                    "wait_time":         0.0,
                    "service_time":      0.0,
                    "due_date":          float(nodes[depot]["due_date"]),
                })
                break

            j  = manager.IndexToNode(next_idx)
            nj = nodes[j]

            travel_km   = float(real_dist[i][j])
            travel_time = travel_km / vehicle_speed
            arr_time    = dep_time + travel_time
            arr_batt    = dep_batt - travel_km * energy_per_km

            wait_time      = max(0.0, float(nj["ready_time"]) - arr_time)
            service_start  = arr_time + wait_time
            service_j      = float(nj["service_time"])
            finish_service = service_start + service_j

            events.append({
                "orig_node_id":      int(nj["orig_node_id"]),
                "node_type":         int(nj["orig_type"]),
                "arrival_time":      arr_time,
                "service_start_time": service_start,
                "departure_time":    finish_service,
                "arrival_battery":   arr_batt,
                "departure_battery": arr_batt,
                "charge_time":       0.0,
                "travel_from_prev":  travel_km,
                "travel_time_from_prev": travel_time,
                "wait_time":         wait_time,
                "service_time":      service_j,
                "due_date":          float(nj["due_date"]),
            })
            cur_time      = finish_service
            cur_batt_real = arr_batt
            idx = next_idx

        routes.append({"vehicle_id": v, "events": events})

    total_customers = len({n["orig_node_id"] for n in nodes if n["orig_type"] == 1})
    return {"routes": routes, "total_customers": total_customers}


def ortools_solution_to_plot_result(extracted: dict, result_analyzes_dir: str) -> dict:
    routes = extracted["routes"]
    total_customers = max(extracted["total_customers"], 1)

    node_sequence = [0]
    step_records  = []

    total_distance         = 0.0
    in_time_window_count   = 0
    positive_battery_count = 0
    global_step            = 0
    total_served_customers = 0

    vehicle_distances = []

    for route in routes:
        events = route["events"]
        route_distance = 0.0

        if len(events) < 2:   # vehicle not used
            continue

        for ev_idx in range(1, len(events)):
            prev_ev = events[ev_idx - 1]
            curr_ev = events[ev_idx]

            dist     = float(curr_ev["travel_from_prev"])
            arr_batt = float(curr_ev["arrival_battery"])
            in_tw    = float(curr_ev["service_start_time"]) <= float(curr_ev["due_date"])
            positive_battery = arr_batt > 0

            if int(curr_ev["node_type"]) == 1:
                total_served_customers += 1

            total_distance += dist
            route_distance += dist
            global_step    += 1
            in_time_window_count   += int(in_tw)
            positive_battery_count += int(positive_battery)

            step_records.append({
                "step":             global_step,
                "vehicle_id":       route["vehicle_id"],
                "from_node":        int(prev_ev["orig_node_id"]),
                "to_node":          int(curr_ev["orig_node_id"]),
                "dist":             round(dist, 4),
                "travel_time":      round(float(curr_ev["travel_time_from_prev"]), 4),
                "served_count":     round(total_served_customers / total_customers, 4),
                "wait_time":        round(float(curr_ev["wait_time"]), 4),
                "service_time":     round(float(curr_ev["service_time"]), 4),
                "charging_time":    round(float(curr_ev["charge_time"]), 4),
                "battery_after":    round(float(curr_ev["departure_battery"]), 4),
                "in_time_window":   in_tw,
                "positive_battery": positive_battery,
                "cumulative_distance": round(total_distance, 4),
                "cumulative_in_time_window_ratio": round(in_time_window_count / global_step, 4),
                "cumulative_positive_battery_ratio": round(positive_battery_count / global_step, 4),
            })
            node_sequence.append(int(curr_ev["orig_node_id"]))

        if route_distance > 1e-9:
            vehicle_distances.append(round(route_distance, 4))

    if vehicle_distances:
        mean_vd = sum(vehicle_distances) / len(vehicle_distances)
        std_vd  = (sum((d - mean_vd) ** 2 for d in vehicle_distances) / len(vehicle_distances)) ** 0.5
        mad_vd  = sum(abs(d - mean_vd) for d in vehicle_distances) / len(vehicle_distances)
        cv_vd   = std_vd / mean_vd if mean_vd > 1e-8 else 0.0
    else:
        mean_vd = std_vd = mad_vd = cv_vd = 0.0

    os.makedirs(result_analyzes_dir, exist_ok=True)

    if step_records:
        step_csv = os.path.join(result_analyzes_dir, f"{CRITERIA}_summary.csv")
        with open(step_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(step_records[0].keys()))
            writer.writeheader()
            writer.writerows(step_records)

    vehicle_rows = [
        {
            "vehicle_id":           vid,
            "vehicle_distance":     round(dist, 4),
            "distance_deviation":   round(dist - mean_vd, 4),
            "abs_deviation":        round(abs(dist - mean_vd), 4),
            "mad_vehicle_distance": "",
            "cv_vehicle_distance":  "",
        }
        for vid, dist in enumerate(vehicle_distances)
    ]
    vehicle_rows.append({
        "vehicle_id":           "ALL",
        "vehicle_distance":     round(mean_vd, 4),
        "distance_deviation":   "mean",
        "abs_deviation":        round(std_vd, 4),
        "mad_vehicle_distance": round(mad_vd, 4),
        "cv_vehicle_distance":  round(cv_vd, 6),
    })
    vehicle_csv = os.path.join(result_analyzes_dir, f"{CRITERIA}_vehicle_summary.csv")
    with open(vehicle_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(vehicle_rows[0].keys()))
        writer.writeheader()
        writer.writerows(vehicle_rows)

    return {"node_sequence": node_sequence, "step_records": step_records}


def select_graph_ids(df: pd.DataFrame, graph_ids=None, sample_n=None, seed=42) -> list:
    """graph_ids if given, else sample_n random ids, else all ids."""
    all_ids = sorted(df["graph_id"].unique().tolist())

    if graph_ids is not None:
        missing = [gid for gid in graph_ids if gid not in all_ids]
        if missing:
            raise ValueError(f"graph_ids not in the data: {missing}")
        return list(graph_ids)

    if sample_n is not None:
        rng = random.Random(seed)
        return sorted(rng.sample(all_ids, min(sample_n, len(all_ids))))

    return all_ids


def _solve_one_graph_worker(task):
    gid, graph_df, solver_kwargs, time_limit_sec, balance_coefficient, \
        vehicle_fixed_cost, graph_save_dir, want_result = task

    try:
        data = build_state_expanded_data_model_from_df(graph_df, **solver_kwargs)
        manager, routing, solution = solve_state_expanded_evrp(
            data,
            time_limit_sec=time_limit_sec,
            balance_coefficient=balance_coefficient,
            vehicle_fixed_cost=vehicle_fixed_cost,
        )
        if solution is None:
            print(f"[graph {gid}] No solution found, skipping.")
            return gid, None, None

        extracted = extract_routes_with_events(data, manager, routing, solution)
        result = ortools_solution_to_plot_result(extracted, result_analyzes_dir=graph_save_dir)
        row = extract_single_graph_metrics(result, graph_id=gid)
        return gid, row, (result if want_result else None)

    except Exception as e:
        print(f"[graph {gid}] Failed with error: {e}")
        return gid, None, None


def run_ortools_plots_for_graphs(
    df: pd.DataFrame,
    result_analyzes_dir: str,
    env_builder,
    solver_kwargs: dict,
    time_limit_sec: int,
    balance_coefficient: int,
    vehicle_fixed_cost: float,
    graph_ids=None,
    sample_n=None,
    sample_n_for_plot: int = 3,
    seed: int = 42,
    n_workers: int = 1,
):
    selected_ids = select_graph_ids(df, graph_ids=graph_ids, sample_n=sample_n, seed=seed)
    print(f"Selected graph_ids ({len(selected_ids)}): {selected_ids}")

    os.makedirs(result_analyzes_dir, exist_ok=True)

    plot_set = set(selected_ids[:sample_n_for_plot])

    tasks = []
    for gid in selected_ids:
        graph_df = df[df["graph_id"] == gid].copy()
        graph_save_dir = os.path.join(result_analyzes_dir, f"graph_{gid}")
        tasks.append((
            gid, graph_df, solver_kwargs, time_limit_sec,
            balance_coefficient, vehicle_fixed_cost,
            graph_save_dir, gid in plot_set,
        ))

    metrics_rows = []
    failed_graph_ids = []

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        for gid, row, result in pool.imap_unordered(_solve_one_graph_worker, tasks):
            if row is None:
                failed_graph_ids.append(gid)
                continue
            metrics_rows.append(row)

            if result is not None:
                graph_save_dir = os.path.join(result_analyzes_dir, f"graph_{gid}")
                env = env_builder(df[df["graph_id"] == gid].copy())
                plot_all_vehicle_figures(
                    env=env, result=result,
                    save_dir=os.path.join(graph_save_dir, "route_figures"),
                    depot_idx=0, show_node_labels=False,
                )
                plot_all_vehicle_battery_and_distance(
                    result=result,
                    save_dir=os.path.join(graph_save_dir, "battery_distance"),
                    depot_idx=0,
                )
                plot_all_vehicle_timeline_and_gantt(
                    result=result,
                    save_dir=os.path.join(graph_save_dir, "timeline_gantt"),
                    depot_idx=0,
                )
                plot_all_vehicle_distance_and_time_profiles(
                    env=env, result=result,
                    save_dir=os.path.join(graph_save_dir, "distance_time_profiles"),
                    depot_idx=0,
                )

            print(f"[graph {gid}] Done.")

    metrics_df = pd.DataFrame(metrics_rows, columns=METRIC_COLUMNS)
    summary_df = summarize_graph_metrics(metrics_df)
    summary_df["num_failed_graphs"] = len(failed_graph_ids)
    summary_df["failed_graph_ids"] = [failed_graph_ids]

    metrics_path = os.path.join(result_analyzes_dir, "metrics_df.csv")
    summary_path = os.path.join(result_analyzes_dir, "summary_df.csv")
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n========== OR-Tools Summary ==========")
    print(summary_df.T)
    print(f"\nSaved metrics_df to: {metrics_path}")
    print(f"Saved summary_df to: {summary_path}")

    return metrics_df, summary_df
