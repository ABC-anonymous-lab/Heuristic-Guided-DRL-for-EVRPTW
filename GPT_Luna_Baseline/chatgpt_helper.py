import os
import csv
import json
import numpy as np
import pandas as pd



def _recursive_find(config: dict, aliases):
    if not isinstance(config, dict):
        return None

    for key in aliases:
        if key in config and config[key] is not None:
            return config[key]

    for value in config.values():
        if isinstance(value, dict):
            found = _recursive_find(value, aliases)
            if found is not None:
                return found

    return None


def get_env_params(config: dict) -> dict:
    if "env" not in config:
        raise KeyError("Config must contain an 'env' section.")

    env = config["env"]

    required_keys = [
        "battery_max",
        "charged_battery_max",
        "n_vehicles",
        "charging_rate",
        "energy_rate",
        "speed",
        "station_allow_battery_ratio",
        "operating_time_min",
        "operating_time_max",
    ]

    missing = [
        key
        for key in required_keys
        if key not in env
    ]

    if missing:
        raise KeyError(
            "Missing required parameter(s) in config['env']: "
            + ", ".join(missing)
        )

    battery_max = float(env["battery_max"])
    charged_battery_max = float(env["charged_battery_max"])

    station_allow_battery_ratio = float(
        env["station_allow_battery_ratio"]
    )

    charging_rate = float(env["charging_rate"])
    energy_rate = float(env["energy_rate"])

    # Config uses km/hour.
    speed_km_per_hour = float(env["speed"])

    # EVRP timestamps / time windows are in minutes.
    speed_km_per_min = speed_km_per_hour / 60.0

    if speed_km_per_min <= 0:
        raise ValueError("env.speed must be > 0.")

    if battery_max <= 0:
        raise ValueError("env.battery_max must be > 0.")

    if charged_battery_max > battery_max:
        raise ValueError(
            "env.charged_battery_max cannot exceed env.battery_max."
        )

    if not (0.0 <= station_allow_battery_ratio <= 1.0):
        raise ValueError(
            "env.station_allow_battery_ratio must be between 0 and 1."
        )

    return {
        "battery_max": battery_max,
        "charged_battery_max": charged_battery_max,

        "n_vehicles": int(env["n_vehicles"]),

        "charging_rate": charging_rate,
        "energy_rate": energy_rate,

        "speed_km_per_hour": speed_km_per_hour,
        "speed_km_per_min": speed_km_per_min,

        "station_allow_battery_ratio":
            station_allow_battery_ratio,

        "station_allow_battery_threshold":
            battery_max * station_allow_battery_ratio,

        "operating_time_min":
            float(env["operating_time_min"]),

        "operating_time_max":
            float(env["operating_time_max"]),
    }


def compute_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    coords = df[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_prompts(
    df: pd.DataFrame,
    config: dict,
):
    p = get_env_params(config)

    dist = compute_distance_matrix(df)

    n_vehicles = p["n_vehicles"]

    node_lines = []

    for _, row in df.iterrows():
        nid = int(row["node_id"])
        node_type = int(row["type"])

        type_name = {
            0: "depot",
            1: "customer",
            2: "charging_station",
        }.get(node_type, "unknown")

        node_lines.append(
            f"  node_id={nid}, "
            f"type={type_name}, "
            f"x_km={row['x_km']:.3f}, "
            f"y_km={row['y_km']:.3f}, "
            f"ready_time={row['ready_time']:.1f}, "
            f"due_date={row['due_date']:.1f}, "
            f"service_time={row['service_time']:.1f}"
        )

    nodes_text = "\n".join(node_lines)

    node_ids = df["node_id"].astype(int).tolist()

    dist_lines = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dist_lines.append(
                f"  d({node_ids[i]},{node_ids[j]})="
                f"{dist[i, j]:.3f}"
            )

    dist_text = "\n".join(dist_lines)

    system_prompt = f"""
You are an expert combinatorial optimization solver for
Electric Vehicle Routing Problems with Time Windows (EVRPTW).

## Vehicle Parameters

- Maximum number of vehicles: {p["n_vehicles"]}

- battery_max: {p["battery_max"]}

- Battery must remain strictly greater than 0 at all times.

- energy_rate: {p["energy_rate"]} kilometers per battery unit
  (battery used = distance / energy_rate).

- speed: {p["speed_km_per_hour"]} km/hour,
  equivalent to {p["speed_km_per_min"]:.4f} km/minute.

- charging_rate: {p["charging_rate"]} battery units per minute.

- charged_battery_max: {p["charged_battery_max"]}

## Charging Rule

A vehicle may visit a charging station only when its battery
upon arrival is less than or equal to:

{p["station_allow_battery_ratio"] * 100:.1f}% of battery_max

Therefore the charging threshold for this instance is:

battery <= {p["station_allow_battery_threshold"]:.4f}

If battery is above this threshold, the vehicle MUST NOT visit
a charging station.

The vehicle must reach the charging station with battery > 0.

When charging is allowed, charge up to:

{p["charged_battery_max"]}

## Time

Vehicles start at:

t = {p["operating_time_min"]}

The operating interval is:

[{p["operating_time_min"]}, {p["operating_time_max"]}]

Early arrival waits until ready_time.

A customer satisfies the time window if:

service_start <= due_date

## Objectives

In priority order:

1. Maximize unique customers served.
2. Keep battery strictly greater than 0.
3. Respect charging station eligibility.
4. Respect customer time windows.
5. Minimize total distance.
6. Avoid unnecessary vehicles.
7. Every used route starts and ends at depot node 0.

Do not visit the same customer more than once.

Return ONLY valid JSON:

{{
  "routes": [
    {{
      "vehicle": 1,
      "path": [0, 3, 7, 0]
    }}
  ],
  "stats": {{
    "customers_served": 10,
    "total_distance": 100.0,
    "vehicles_used": 1
  }},
  "explanation": "brief reasoning"
}}
""".strip()

    user_prompt = f"""
Solve this EVRPTW instance.

Total nodes: {len(df)}

Customers: {int((df["type"] == 1).sum())}

Charging stations: {int((df["type"] == 2).sum())}

Maximum vehicles: {n_vehicles}

## Nodes

{nodes_text}

## Distance Matrix (km)

{dist_text}

Construct the best feasible routes under all environment rules.
""".strip()

    return system_prompt, user_prompt, dist


def estimate_tokens(system_prompt: str, user_prompt: str) -> int:
    return int(np.ceil(len(system_prompt + "\n" + user_prompt) / 3.5))


def routes_to_result(routes: list, graph_df: pd.DataFrame, config: dict) -> dict:
    """Simulate the LLM routes with the env parameters and compute step records and metrics."""
    p = get_env_params(config)

    df = graph_df.copy().reset_index(drop=True)
    node_info = df.set_index("node_id")
    dist_matrix = compute_distance_matrix(df)
    node_id_to_idx = {int(row["node_id"]): i for i, row in df.iterrows()}

    total_customers = int((df["type"] == 1).sum())
    customer_ids = set(df.loc[df["type"] == 1, "node_id"].astype(int))

    node_sequence = [0]
    step_records = []
    vehicle_distances = []

    total_distance = 0.0
    global_step = 0
    positive_battery_count = 0
    customer_visit_count = 0
    in_time_window_count = 0
    served_customers = set()
    battery_violations = 0
    charging_rule_violations = 0
    time_window_violations = 0
    duplicate_customer_visits = 0

    battery_max = p["battery_max"]
    charged_battery_max = p["charged_battery_max"]
    charging_rate = p["charging_rate"]
    energy_rate = p["energy_rate"]
    speed_km_per_min = p["speed_km_per_min"]
    operating_time_min = p["operating_time_min"]
    charging_threshold = p["station_allow_battery_threshold"]

    for route_info in routes:
        path = [int(x) for x in route_info.get("path", [])]
        vehicle_id = int(route_info.get("vehicle", 1)) - 1

        if len(path) < 2:
            continue
        if path[0] != 0:
            raise ValueError(f"Vehicle {vehicle_id + 1} route does not start at depot: {path}")
        if path[-1] != 0:
            raise ValueError(f"Vehicle {vehicle_id + 1} route does not end at depot: {path}")

        battery = battery_max
        time = operating_time_min
        route_distance = 0.0

        for k in range(1, len(path)):
            from_nid = path[k - 1]
            to_nid = path[k]
            for nid in (from_nid, to_nid):
                if nid not in node_id_to_idx:
                    raise ValueError(f"Unknown node_id: {nid}")

            dist = float(dist_matrix[node_id_to_idx[from_nid], node_id_to_idx[to_nid]])
            travel_time = dist / speed_km_per_min
            energy_used = dist / energy_rate

            battery_before = battery
            battery -= energy_used
            battery_on_arrival = battery
            time += travel_time
            route_distance += dist
            total_distance += dist
            global_step += 1

            positive_battery = battery > 0.0
            if positive_battery:
                positive_battery_count += 1
            else:
                battery_violations += 1

            # Destination node
            to_row = node_info.loc[to_nid]
            node_type = int(to_row["type"])
            ready_time = float(to_row["ready_time"])
            due_date = float(to_row["due_date"])
            service_time = float(to_row["service_time"])

            # Time window
            wait_time = max(0.0, ready_time - time)
            time += wait_time
            service_start = time
            in_time_window = service_start <= due_date

            if node_type == 1:
                customer_visit_count += 1
                if in_time_window:
                    in_time_window_count += 1
                else:
                    time_window_violations += 1

            time += service_time
            if node_type == 1:
                if to_nid in served_customers:
                    duplicate_customer_visits += 1
                else:
                    served_customers.add(to_nid)

            # Charging: allowed only with 0 < battery <= threshold
            charging_time = 0.0
            charging_allowed = None
            if node_type == 2:
                charging_allowed = 0.0 < battery <= charging_threshold
                if charging_allowed:
                    charge_amount = max(0.0, charged_battery_max - battery)
                    charging_time = charge_amount / charging_rate
                    battery += charge_amount
                    time += charging_time
                else:
                    charging_rule_violations += 1

            served_ratio = len(served_customers) / max(total_customers, 1)
            tw_ratio = in_time_window_count / customer_visit_count if customer_visit_count > 0 else 0.0
            positive_battery_ratio = positive_battery_count / global_step

            step_records.append({
                "step": global_step,
                "vehicle_id": vehicle_id,
                "from_node": from_nid,
                "to_node": to_nid,
                "dist": round(dist, 4),
                "travel_time": round(travel_time, 4),
                "battery_before": round(battery_before, 4),
                "battery_on_arrival": round(battery_on_arrival, 4),
                "energy_used": round(energy_used, 4),
                "served_count": round(served_ratio, 4),
                "wait_time": round(wait_time, 4),
                "service_time": round(service_time, 4),
                "service_start": round(service_start, 4),
                "charging_time": round(charging_time, 4),
                "charging_allowed": charging_allowed,
                "battery_after": round(battery, 4),
                "in_time_window": bool(in_time_window),
                "positive_battery": bool(positive_battery),
                "cumulative_distance": round(total_distance, 4),
                "cumulative_in_time_window_ratio": round(tw_ratio, 4),
                "cumulative_positive_battery_ratio": round(positive_battery_ratio, 4),
            })
            node_sequence.append(to_nid)

        if route_distance > 1e-9:
            vehicle_distances.append(round(route_distance, 4))

    # Final metrics
    served_count = len(served_customers) / max(total_customers, 1)
    in_time_window_ratio = in_time_window_count / customer_visit_count if customer_visit_count > 0 else 0.0
    positive_battery_ratio = positive_battery_count / global_step if global_step > 0 else 0.0
    feasible = (battery_violations == 0
                and charging_rule_violations == 0
                and duplicate_customer_visits == 0)

    return {
        "node_sequence": node_sequence,
        "step_records": step_records,
        "vehicle_distances": vehicle_distances,
        "metrics": {
            "total_distance": total_distance,
            "served_count": served_count,
            "in_time_window_ratio": in_time_window_ratio,
            "positive_battery_ratio": positive_battery_ratio,
            "num_steps": global_step,
            "num_vehicles_used": len(vehicle_distances),
            "battery_violations": battery_violations,
            "charging_rule_violations": charging_rule_violations,
            "time_window_violations": time_window_violations,
            "duplicate_customer_visits": duplicate_customer_visits,
            "all_customers_served": served_customers == customer_ids,
            "feasible": feasible,
        },
    }


def extract_metrics(result, graph_id, estimated_input_tokens, usage):
    m = result["metrics"]
    return {
        "graph_id": graph_id,
        "total_distance": round(m["total_distance"], 4),
        "served_count": round(m["served_count"], 6),
        "in_time_window_ratio": round(m["in_time_window_ratio"], 6),
        "positive_battery_ratio": round(m["positive_battery_ratio"], 6),
        "num_steps": m["num_steps"],
        "num_vehicles_used": m["num_vehicles_used"],
        "battery_violations": m["battery_violations"],
        "charging_rule_violations": m["charging_rule_violations"],
        "time_window_violations": m["time_window_violations"],
        "duplicate_customer_visits": m["duplicate_customer_visits"],
        "all_customers_served": m["all_customers_served"],
        "feasible": m["feasible"],
        "estimated_input_tokens": estimated_input_tokens,
        "api_input_tokens": usage["input_tokens"],
        "api_output_tokens": usage["output_tokens"],
        "api_total_tokens": usage["total_tokens"],
    }


def summarize_metrics(metrics_df: pd.DataFrame) -> dict:
    summary = {"num_graphs": len(metrics_df)}
    cols = [
        "total_distance", "served_count", "in_time_window_ratio",
        "positive_battery_ratio", "num_steps", "num_vehicles_used",
        "battery_violations", "charging_rule_violations",
        "time_window_violations", "duplicate_customer_visits",
        "estimated_input_tokens", "api_input_tokens",
        "api_output_tokens", "api_total_tokens",
    ]

    for col in cols:
        if col not in metrics_df.columns:
            continue
        values = pd.to_numeric(metrics_df[col], errors="coerce").dropna()
        if not len(values):
            continue

        summary[f"{col}_mean"] = round(float(values.mean()), 6)
        summary[f"{col}_std"] = round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0
        summary[f"{col}_variance"] = round(float(values.var(ddof=1)), 6) if len(values) > 1 else 0.0
        summary[f"{col}_min"] = round(float(values.min()), 6)
        summary[f"{col}_max"] = round(float(values.max()), 6)

    for col in ["estimated_input_tokens", "api_input_tokens", "api_output_tokens", "api_total_tokens"]:
        if col in metrics_df.columns:
            summary[f"{col}_sum"] = int(metrics_df[col].sum())

    if "feasible" in metrics_df.columns:
        summary["feasible_ratio"] = round(float(metrics_df["feasible"].astype(float).mean()), 6)
    if "all_customers_served" in metrics_df.columns:
        summary["all_customers_served_ratio"] = round(
            float(metrics_df["all_customers_served"].astype(float).mean()), 6
        )

    return summary


def save_example_summaries(result: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    records = result["step_records"]

    if records:
        with open(os.path.join(save_dir, "chatgpt_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    distances = result["vehicle_distances"]
    if distances:
        mean_v = float(np.mean(distances))
        std_v = float(np.std(distances, ddof=0))
        mad_v = float(np.mean(np.abs(np.asarray(distances) - mean_v)))
        cv_v = std_v / mean_v if mean_v > 1e-8 else 0.0
    else:
        mean_v = std_v = mad_v = cv_v = 0.0

    rows = []
    for vid, d in enumerate(distances):
        rows.append({
            "vehicle_id": vid,
            "vehicle_distance": round(d, 4),
            "distance_deviation": round(d - mean_v, 4),
            "abs_deviation": round(abs(d - mean_v), 4),
            "mad_vehicle_distance": "",
            "cv_vehicle_distance": "",
        })

    rows.append({
        "vehicle_id": "ALL",
        "vehicle_distance": round(mean_v, 4),
        "distance_deviation": "mean",
        "abs_deviation": round(std_v, 4),
        "mad_vehicle_distance": round(mad_v, 4),
        "cv_vehicle_distance": round(cv_v, 6),
    })

    with open(os.path.join(save_dir, "chatgpt_vehicle_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_prompt_example(
    save_dir, graph_id, model,
    system_prompt, user_prompt,
    raw_response, parsed_response,
    usage, estimated_input_tokens
):
    prompt_dir = os.path.join(save_dir, "prompt_example")
    os.makedirs(prompt_dir, exist_ok=True)

    with open(os.path.join(prompt_dir, "system_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(system_prompt)
    with open(os.path.join(prompt_dir, "user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(user_prompt)
    with open(os.path.join(prompt_dir, "raw_response.txt"), "w", encoding="utf-8") as f:
        f.write(raw_response)
    with open(os.path.join(prompt_dir, "response.json"), "w", encoding="utf-8") as f:
        json.dump(parsed_response, f, indent=2, ensure_ascii=False)

    metadata = {
        "graph_id": int(graph_id),
        "model": model,
        "estimated_input_tokens": int(estimated_input_tokens),
        "api_input_tokens": int(usage["input_tokens"]),
        "api_output_tokens": int(usage["output_tokens"]),
        "api_total_tokens": int(usage["total_tokens"]),
    }
    with open(os.path.join(prompt_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
