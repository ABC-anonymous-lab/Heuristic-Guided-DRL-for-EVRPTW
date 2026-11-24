import numpy as np
import pandas as pd


# Per-graph metric columns (same names as the DRL test summaries)
METRIC_COLUMNS = [
    "graph_id", "reward", "len", "served_count", "in_time_window_ratio",
    "positive_battery_ratio", "total_distance", "num_vehicles",
    "vehicle_distance_std", "vehicle_distance_mean",
]


def extract_single_graph_metrics(result, graph_id=None):
    step_records = result["step_records"]
    last_step = step_records[-1]

    vehicle_dist = {}
    for step in step_records:
        vid = step["vehicle_id"]
        vehicle_dist[vid] = vehicle_dist.get(vid, 0.0) + step["dist"]
    vehicle_distances = np.asarray(list(vehicle_dist.values()), dtype=float)

    return {
        "graph_id": graph_id,
        "reward": np.nan,
        "len": len(step_records),
        "served_count": last_step["served_count"],
        "in_time_window_ratio": last_step["cumulative_in_time_window_ratio"],
        "positive_battery_ratio": last_step["cumulative_positive_battery_ratio"],
        "total_distance": last_step["cumulative_distance"],
        "num_vehicles": len(vehicle_distances),
        "vehicle_distance_std": float(np.std(vehicle_distances)),
        "vehicle_distance_mean": float(np.mean(vehicle_distances)),
    }


def summarize_graph_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "num_graphs": len(metrics_df),

        "reward": metrics_df["reward"].mean(),
        "len": metrics_df["len"].mean(),
        "served_count": metrics_df["served_count"].mean(),
        "in_time_window_ratio": metrics_df["in_time_window_ratio"].mean(),
        "positive_battery_ratio": metrics_df["positive_battery_ratio"].mean(),

        "total_distance": metrics_df["total_distance"].sum(),
        "total_distance_min": metrics_df["total_distance"].min(),
        "total_distance_max": metrics_df["total_distance"].max(),
        "total_distance_mean": metrics_df["total_distance"].mean(),

        "num_vehicles_min": metrics_df["num_vehicles"].min(),
        "num_vehicles_max": metrics_df["num_vehicles"].max(),
        "num_vehicles_mean": metrics_df["num_vehicles"].mean(),

        "vehicle_distance_std_min": metrics_df["vehicle_distance_std"].min(),
        "vehicle_distance_std_max": metrics_df["vehicle_distance_std"].max(),
        "vehicle_distance_std_mean": metrics_df["vehicle_distance_std"].mean(),

        "vehicle_distance_mean_min": metrics_df["vehicle_distance_mean"].min(),
        "vehicle_distance_mean_max": metrics_df["vehicle_distance_mean"].max(),
        "vehicle_distance_mean_mean": metrics_df["vehicle_distance_mean"].mean(),
    }
    return pd.DataFrame([summary])