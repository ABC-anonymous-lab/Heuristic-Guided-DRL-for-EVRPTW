import os
import sys
import argparse
from datetime import datetime

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import load_config
from utils.init_model import build_env_builder
from ortools_to_plot_result import run_ortools_plots_for_graphs


def build_solver_kwargs(env: dict, ort: dict) -> dict:
    battery_max = int(env["battery_max"])
    return {
        "num_vehicles": int(env["n_vehicles"]),
        "battery_capacity": battery_max,
        "battery_levels": list(range(0, battery_max, int(ort["battery_level_step"]))),
        "target_charge_pct": round(float(env["charged_battery_max"]) / battery_max, 2),
        "vehicle_speed": float(env["speed"]) / 60.0,       # km/h -> km/min
        "energy_per_km": 1.0 / float(env["energy_rate"]),  # env: km per battery unit
        "charge_rate": 1.0 / float(env["charging_rate"]),  # env: battery units per minute
    }


def load_graph_ids(raw):
    if raw is None or isinstance(raw, list):
        return raw
    ids = []
    with open(_from_repo_root(raw), "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0]
            ids.extend(int(tok) for tok in line.replace(",", " ").split())
    return ids


def _from_repo_root(path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def load_benchmark_df(data_path: str) -> pd.DataFrame:
    if data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    raise ValueError(f"Unsupported data format (use .csv or .parquet): {data_path}")


def main():
    parser = argparse.ArgumentParser(description="OR-Tools baseline for the EVRPTW")
    parser.add_argument("--config", required=True, help="YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg, ort = cfg["env"], cfg["ortools"]
    solver_kwargs = build_solver_kwargs(env_cfg, ort)

    data_path = _from_repo_root(ort["data_path"])
    df = load_benchmark_df(data_path)

    run_dir = os.path.join(_from_repo_root(ort["results_dir"]),
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    print(f"Config        : {args.config}")
    print(f"Data          : {data_path}")
    print(f"Output        : {run_dir}")
    print(f"Solver        : {solver_kwargs}")
    print(f"time_limit_sec: {ort['time_limit_sec']}, n_workers: {ort['n_workers']}")

    run_ortools_plots_for_graphs(
        df=df,
        result_analyzes_dir=run_dir,
        env_builder=build_env_builder(cfg),
        solver_kwargs=solver_kwargs,
        time_limit_sec=int(ort["time_limit_sec"]),
        balance_coefficient=int(ort["balance_coefficient"]),
        vehicle_fixed_cost=float(ort["vehicle_fixed_cost"]),
        graph_ids=load_graph_ids(ort["graph_ids"]),
        sample_n=ort["sample_n"],
        sample_n_for_plot=int(ort["sample_n_for_plot"]),
        seed=int(ort["seed"]),
        n_workers=int(ort["n_workers"]),
    )


if __name__ == "__main__":
    main()