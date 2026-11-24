import numpy as np
import pandas as pd
from .vectorized_arcs import build_arcs_for_node_i_vectorized


def build_state_expanded_data_model_from_df_vectorized(
    graph_df: pd.DataFrame,
    battery_levels,
    num_vehicles: int = 4,
    vehicle_speed: float = 1.0,
    battery_capacity: int = 100,
    charge_rate: float = 1.0,
    energy_per_km: float = 1.0,
    forbid_same_station_jump: bool = True,
    target_charge_pct: float = 0.8
):
    """State-expanded data model of one graph (nodes, distance/energy matrices, feasible arcs
    and their travel times, disjunction groups).
    """
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

    depot_df = graph_df[graph_df["type"] == 0].copy()
    cust_df = graph_df[graph_df["type"] == 1].copy()
    station_df = graph_df[graph_df["type"] == 2].copy()

    if len(depot_df) != 1:
        raise ValueError(f"Each graph needs exactly one depot, found {len(depot_df)}")

    depot_row = depot_df.iloc[0]
    station_clone_limit = max(num_vehicles, 3)

    data = {}
    data["graph_id"] = int(graph_df["graph_id"].iloc[0])
    data["num_vehicles"] = int(num_vehicles)
    data["vehicle_speed"] = float(vehicle_speed)
    data["battery_capacity"] = int(battery_capacity)
    data["battery_levels"] = list(sorted(set(int(x) for x in battery_levels)))
    data["charge_rate"] = float(charge_rate)
    data["energy_per_km"] = float(energy_per_km)
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
        "ready_time": int(round(float(depot_row["ready_time"]))),
        "due_date": int(round(float(depot_row["due_date"]))),
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
                "ready_time": int(round(float(row["ready_time"]))),
                "due_date": int(round(float(row["due_date"]))),
                "battery_level": int(b),
                "physical_station_id": None,
                "group_key": f"CUST_{int(row['node_id'])}",
            })

    for _, row in station_df.iterrows():
        sid = int(row["node_id"])
        for c in range(station_clone_limit):
            expanded_nodes.append({
                "name": f"chg_{sid}_to80_clone{c}",
                "kind": "charger_state",
                "orig_node_id": sid,
                "orig_type": 2,
                "x_km": float(row["x_km"]),
                "y_km": float(row["y_km"]),
                "service_time": 0,
                "ready_time": int(round(float(row["ready_time"]))),
                "due_date": int(round(float(row["due_date"]))),
                "battery_level": int(data["target_charge_level"]),
                "physical_station_id": sid,
                "clone_id": c,
                "group_key": f"CHG_{sid}",
            })

    data["nodes"] = expanded_nodes
    data["num_nodes"] = len(expanded_nodes)
    data["depot"] = 0

    coords = np.array([[n["x_km"], n["y_km"]] for n in expanded_nodes], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    euclid = np.linalg.norm(diff, axis=-1)

    energy_matrix_real = euclid * data["energy_per_km"]
    energy_matrix_int = np.rint(energy_matrix_real).astype(int)


    data["real_distance_matrix"] = euclid
    data["energy_matrix_int"]    = energy_matrix_int

    num_nodes = data["num_nodes"]
    target_batt = int(data["target_charge_level"])
    levels_sorted = data["battery_levels"]

    orig_type_arr = np.array([n["orig_type"] for n in expanded_nodes], dtype=int)
    orig_node_id_arr = np.array([n["orig_node_id"] for n in expanded_nodes], dtype=int)
    kind_arr = np.array([n["kind"] for n in expanded_nodes], dtype=object)
    battery_level_arr = np.array([n["battery_level"] for n in expanded_nodes], dtype=int)
    physical_station_id_arr = np.array(
        [(-1 if n["physical_station_id"] is None else n["physical_station_id"]) for n in expanded_nodes],
        dtype=int,
    )

    allowed_arcs: set = set()
    arc_time_dict: dict = {}

    for i, ni in enumerate(expanded_nodes):
        if ni["kind"] not in {"depot_start", "customer_state", "charger_state"}:
            continue

        (
            customer_j_indices, customer_transit,
            charger_j_indices, charger_transit,
        ) = build_arcs_for_node_i_vectorized(
            i=i,
            ni=ni,
            expanded_nodes_orig_type=orig_type_arr,
            expanded_nodes_orig_node_id=orig_node_id_arr,
            expanded_nodes_kind=kind_arr,
            expanded_nodes_battery_level=battery_level_arr,
            expanded_nodes_physical_station_id=physical_station_id_arr,
            energy_matrix_int_row=energy_matrix_int[i],
            euclid_row=euclid[i],
            vehicle_speed=data["vehicle_speed"],
            levels_sorted=levels_sorted,
            target_batt=target_batt,
            num_nodes=num_nodes,
        )

        for j, t in zip(customer_j_indices.tolist(), customer_transit.tolist()):
            allowed_arcs.add((i, j))
            arc_time_dict[(i, j)] = int(t)

        for j, t in zip(charger_j_indices.tolist(), charger_transit.tolist()):
            allowed_arcs.add((i, j))
            arc_time_dict[(i, j)] = int(t)

    charger_kind_set = {
        idx for idx, n in enumerate(expanded_nodes) if n["kind"] == "charger_state"
    }

    charger_targets_by_i: dict = {}
    for (i, j) in allowed_arcs:
        if j in charger_kind_set:
            charger_targets_by_i.setdefault(i, []).append(j)

    for i, j_list in charger_targets_by_i.items():
        ni = expanded_nodes[i]
        if ni["kind"] not in {"depot_start", "customer_state", "charger_state"}:
            continue

        bi = int(ni["battery_level"])
        from_service = int(ni["service_time"])

        j_arr = np.asarray(j_list, dtype=int)
        e_ij_arr = energy_matrix_int[i, j_arr]
        travel_min_arr = np.round(euclid[i, j_arr] / data["vehicle_speed"]).astype(int)
        arrival_batt_arr = bi - e_ij_arr

        charge_min_arr = np.round(
            (target_batt - arrival_batt_arr) * data["charge_rate"]
        ).astype(int)

        transit_arr = from_service + travel_min_arr + charge_min_arr

        for j, t in zip(j_arr.tolist(), transit_arr.tolist()):
            arc_time_dict[(i, j)] = int(t)

    data["allowed_arcs"] = allowed_arcs
    data["arc_time_dict"] = arc_time_dict

    customer_groups = {}
    charger_groups = {}
    for idx, node in enumerate(expanded_nodes):
        if node["kind"] == "customer_state":
            customer_groups.setdefault(node["group_key"], []).append(idx)
        elif node["kind"] == "charger_state":
            charger_groups.setdefault(node["group_key"], []).append(idx)

    data["customer_groups"] = customer_groups
    data["charger_groups"] = charger_groups

    return data