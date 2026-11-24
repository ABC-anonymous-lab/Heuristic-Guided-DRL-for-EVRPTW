from typing import List, Optional, Tuple

import numpy as np


def extract_vehicle_distance_stats(vehicle_distances: Optional[List[float]]) -> Tuple[int, float, float, float, float]:
    if vehicle_distances is None or len(vehicle_distances) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    vd = np.asarray(vehicle_distances, dtype=np.float32)
    return (
        int(len(vd)),
        float(vd.mean()),
        float(vd.std(ddof=0)),
        float(vd.min()),
        float(vd.max()),
    )


def summarize_eval_stats(all_stats: List[dict], return_details: bool = False):
    if not all_stats:
        empty = {"num_graphs": 0}
        return (empty, []) if return_details else empty

    def values_of(key):
        return [x[key] for x in all_stats if key in x]

    def mean_of(key):
        v = values_of(key)
        return float(sum(v) / len(v)) if v else 0.0

    def min_of(key):
        v = values_of(key)
        return float(min(v)) if v else 0.0

    def max_of(key):
        v = values_of(key)
        return float(max(v)) if v else 0.0

    summary = {
        "num_graphs": int(len(all_stats)),
        "reward": mean_of("reward"),
        "len": mean_of("len"),
        "served_count": mean_of("served_count"),
        "in_time_window_ratio": mean_of("in_time_window_ratio"),
        "positive_battery_ratio": mean_of("positive_battery_ratio"),
        "total_distance": mean_of("total_distance"),
        "total_distance_min": min_of("total_distance"),
        "total_distance_max": max_of("total_distance"),
        "total_distance_mean": mean_of("total_distance"),
        "num_vehicles_min": min_of("num_vehicles"),
        "num_vehicles_max": max_of("num_vehicles"),
        "num_vehicles_mean": mean_of("num_vehicles"),
        "vehicle_distance_std_min": min_of("vehicle_distance_std"),
        "vehicle_distance_std_max": max_of("vehicle_distance_std"),
        "vehicle_distance_std_mean": mean_of("vehicle_distance_std"),
        "vehicle_distance_mean_min": min_of("vehicle_distance_mean"),
        "vehicle_distance_mean_max": max_of("vehicle_distance_mean"),
        "vehicle_distance_mean_mean": mean_of("vehicle_distance_mean"),
    }

    return (summary, all_stats) if return_details else summary