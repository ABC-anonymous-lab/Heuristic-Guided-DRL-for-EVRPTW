import numpy as np


def round_down_level_vec(arrival_batt_array, levels_sorted):
    levels_arr = np.asarray(levels_sorted)
    idx = np.searchsorted(levels_arr, arrival_batt_array, side="right")
    mapped = np.where(idx > 0, levels_arr[np.clip(idx - 1, 0, None)], -1)
    mapped = np.where(arrival_batt_array < 0, -1, mapped)
    return mapped


def build_arcs_for_node_i_vectorized(
    i,
    ni,
    expanded_nodes_orig_type,
    expanded_nodes_orig_node_id,
    expanded_nodes_kind,
    expanded_nodes_battery_level,
    expanded_nodes_physical_station_id,
    energy_matrix_int_row,
    euclid_row,
    vehicle_speed,
    levels_sorted,
    target_batt,
    num_nodes,
):

    bi = int(ni["battery_level"])
    from_service = int(ni["service_time"])
    ni_orig_type = int(ni["orig_type"])
    ni_orig_node_id = int(ni["orig_node_id"])
    ni_kind = ni["kind"]
    ni_physical_station_id = ni["physical_station_id"]
    if ni_physical_station_id is None:
        ni_physical_station_id = -1

    j_all = np.arange(1, num_nodes)

    e_ij = energy_matrix_int_row[j_all]
    travel_min = np.round(euclid_row[j_all] / vehicle_speed).astype(int)

    kind_j = expanded_nodes_kind[j_all]
    orig_type_j = expanded_nodes_orig_type[j_all]
    orig_node_id_j = expanded_nodes_orig_node_id[j_all]
    battery_level_j = expanded_nodes_battery_level[j_all]
    physical_station_id_j = expanded_nodes_physical_station_id[j_all]

    valid_base = np.ones(len(j_all), dtype=bool)
    valid_base &= (j_all != i)

    same_customer_mask = (
        (ni_orig_type == 1)
        & (orig_type_j == 1)
        & (orig_node_id_j == ni_orig_node_id)
    )
    valid_base &= ~same_customer_mask

    valid_base &= (kind_j != "depot_start")

    is_customer_j = (kind_j == "customer_state") & valid_base
    arrival_batt_cust = bi - e_ij
    mapped_cust = round_down_level_vec(arrival_batt_cust, levels_sorted)
    customer_match = is_customer_j & (mapped_cust != -1) & (mapped_cust == battery_level_j)

    customer_j_indices = j_all[customer_match]
    customer_transit = from_service + travel_min[customer_match]

    is_charger_j = (kind_j == "charger_state") & valid_base
    arrival_batt_chg = bi - e_ij
    charger_valid = is_charger_j & (arrival_batt_chg >= 0) & (arrival_batt_chg <= target_batt)

    if ni_kind == "charger_state":
        same_station_mask = (physical_station_id_j == ni_physical_station_id)
        charger_valid &= ~same_station_mask

    charger_j_indices = j_all[charger_valid]
    charger_transit = from_service + travel_min[charger_valid]

    return (
        customer_j_indices, customer_transit,
        charger_j_indices, charger_transit,
    )