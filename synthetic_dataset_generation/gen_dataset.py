"""Generate EVRPTW instances from raw customer locations.

Pipeline: (optional) raw locations from R -> depot -> charging stations ->
coordinate scaling -> time windows -> time-window normalization -> train/val/test split.

Usage:
    python synthetic_dataset_generation/gen_dataset.py synthetic_dataset_generation/dataset_config_example.yaml
"""

import sys
import time
import random
import subprocess
from pathlib import Path
from typing import List, Tuple

import yaml
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# File name written by gen_raw_data.R
RAW_NAME_TEMPLATE = "tsp_n{n}_{n_graphs}graphs.csv"

# Column order of the output files
FINAL_COLS = [
    "graph_id", "node_id", "type", "x", "y",
    "time_slot", "service_time", "ready_time", "due_date",
    "x_km", "y_km", "ready_time_norm", "due_date_norm",
]


# Mark all raw points as customers (type 1) and add the depot (type 0, node 0)
# at the center of the customers of each graph
def add_depot_by_kmeans(df: pd.DataFrame, kmeans_seed: int = 0) -> pd.DataFrame:
    required = {"graph_id", "node_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["type"] = 1

    depot_rows = []
    for graph_id, group in df.groupby("graph_id", sort=True):
        coords = group[["x", "y"]].to_numpy()
        km = KMeans(n_clusters=1, random_state=kmeans_seed, n_init=10)
        km.fit(coords)
        cx, cy = km.cluster_centers_[0]
        depot_rows.append({"graph_id": graph_id, "node_id": 0, "type": 0,
                           "x": float(cx), "y": float(cy)})

    depot_df = pd.DataFrame(depot_rows)
    customer_df = df[["graph_id", "node_id", "type", "x", "y"]]
    out = pd.concat([depot_df, customer_df], ignore_index=True)
    out = out.sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    return out


# Add num_stations charging stations (type 2) at the k-means centers of the customers.
def add_charging_stations(df: pd.DataFrame, num_stations: int,
                          kmeans_seed: int = 0) -> pd.DataFrame:
    required = {"graph_id", "node_id", "type", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    station_rows = []
    for graph_id, group in df.groupby("graph_id", sort=True):
        customers = group[group["type"] == 1]
        if customers.empty:
            raise ValueError(f"Graph {graph_id} has no customer nodes (type == 1).")

        coords = customers[["x", "y"]].to_numpy()
        km = KMeans(n_clusters=num_stations, random_state=kmeans_seed, n_init=10)
        km.fit(coords)

        next_node_id = int(group["node_id"].max()) + 1
        for i, (cx, cy) in enumerate(km.cluster_centers_):
            station_rows.append({"graph_id": graph_id, "node_id": next_node_id + i,
                                 "type": 2, "x": float(cx), "y": float(cy)})

    station_df = pd.DataFrame(station_rows)
    out = pd.concat([df, station_df], ignore_index=True)
    out = out.sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    return out


# Unit-square coordinates -> km.
def scale_coordinates(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    df = df.copy()
    df["x_km"] = df["x"] * scale
    df["y_km"] = df["y"] * scale
    return df


# Give each customer a random time window from periods.
def assign_time_windows(df: pd.DataFrame, periods: List[Tuple[int, int]],
                        service_time: int = 3, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    df = df.copy()

    global_start = min(s for s, _ in periods)
    global_end = max(e for _, e in periods)

    df["time_slot"] = 0
    df["service_time"] = 0
    df.loc[df["type"] == 1, "service_time"] = service_time
    df["ready_time"] = global_start
    df["due_date"] = global_end

    for idx in df.index[df["type"] == 1]:
        slot_idx = random.randint(0, len(periods) - 1)
        start, end = periods[slot_idx]
        df.at[idx, "time_slot"] = slot_idx + 1      # 1-based
        df.at[idx, "ready_time"] = start
        df.at[idx, "due_date"] = end

    return df.sort_values(["graph_id", "node_id"]).reset_index(drop=True)


def norm_timewindow(df: pd.DataFrame, t_min: float, t_max: float) -> pd.DataFrame:
    df = df.copy()
    span = (t_max - t_min) if t_max > t_min else 1.0
    df["ready_time_norm"] = (df["ready_time"] - t_min) / span
    df["due_date_norm"] = (df["due_date"] - t_min) / span
    return df


def split_data_by_graph(df: pd.DataFrame, val_ratio: float, test_ratio: float,
                        seed: int = 42, graph_col: str = "graph_id"):
    graph_ids = df[graph_col].unique()
    temp_size = val_ratio + test_ratio
    train_gids, temp_gids = train_test_split(
        graph_ids, test_size=temp_size, random_state=seed, shuffle=True)
    rel_test = test_ratio / temp_size if temp_size > 0 else 0.5
    val_gids, test_gids = train_test_split(
        temp_gids, test_size=rel_test, random_state=seed, shuffle=True)

    train_df = df[df[graph_col].isin(train_gids)].copy()
    val_df = df[df[graph_col].isin(val_gids)].copy()
    test_df = df[df[graph_col].isin(test_gids)].copy()
    return train_df, val_df, test_df


def _save_df(df: pd.DataFrame, folder: Path, base_name: str, fmt: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        path = folder / f"{base_name}.csv"
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        path = folder / f"{base_name}.parquet"
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output_format: {fmt} (use 'csv' or 'parquet')")
    return path


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = yaml.safe_load(p.read_text())
    cfg["_config_dir"] = str(p.resolve().parent)
    return cfg


def _resolve(cfg: dict, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else Path(cfg["_config_dir"]) / p


def _ask_regenerate(raw_path: Path) -> bool:
    if not sys.stdin.isatty():
        print(f"[R] {raw_path.name} exists; keeping it (non-interactive run).")
        return False
    ans = input(f"[R] {raw_path.name} exists. Regenerate? [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def _maybe_run_r(cfg: dict, sizes: List[int], raw_dir: Path) -> None:
    if not cfg["run_r"]:
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    mode = cfg["on_raw_exists"]
    if mode not in ("skip", "regenerate", "ask"):
        raise ValueError(f"on_raw_exists must be skip / regenerate / ask, got: {mode}")

    to_generate = []
    for n in sizes:
        raw_path = raw_dir / RAW_NAME_TEMPLATE.format(n=n, n_graphs=cfg["n_graphs"])
        if not raw_path.exists():
            to_generate.append(n)
        elif mode == "skip":
            print(f"[R] Exists, skipping: {raw_path.name}")
        elif mode == "regenerate":
            print(f"[R] Exists, regenerating: {raw_path.name}")
            to_generate.append(n)
        elif _ask_regenerate(raw_path):
            to_generate.append(n)

    if not to_generate:
        print("[R] No raw files to generate.")
        return

    r_script = _resolve(cfg, cfg["r_script"])
    cmd = [
        cfg["rscript_path"], str(r_script),
        f"--sizes={','.join(str(n) for n in to_generate)}",
        f"--n_graphs={cfg['n_graphs']}",
        f"--iters={cfg['r_iters']}",
        f"--seed={cfg['r_seed']}",
        f"--out_dir={raw_dir}",
    ]
    print(f"[R] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"Rscript not found ('{cfg['rscript_path']}'). Install R and put Rscript on "
            f"the PATH, or set rscript_path to its absolute path."
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Rscript failed (exit code {e.returncode}). Check that the R package tspgen "
            f"is installed; run the command above by hand to see the R error."
        ) from None


def _build_geometry(raw_df: pd.DataFrame, size_cfg: dict, cfg: dict) -> pd.DataFrame:
    g = add_depot_by_kmeans(raw_df, kmeans_seed=cfg["kmeans_seed"])
    g = add_charging_stations(g, num_stations=size_cfg["num_stations"],
                              kmeans_seed=cfg["kmeans_seed"])
    g = scale_coordinates(g, scale=size_cfg["scale_km"])
    return g


def run(cfg: dict) -> None:
    output_dir = _resolve(cfg, cfg["output_dir"])
    raw_dir = _resolve(cfg, cfg["raw_dir"])
    sizes = [int(n) for n in cfg["customer_counts"]]
    size_params = {int(k): v for k, v in cfg["size_params"].items()}
    variants = cfg["time_window_selection"]

    output_dir.mkdir(parents=True, exist_ok=True)
    _maybe_run_r(cfg, sizes, raw_dir)

    print(f"\nSizes: {sizes}, time-window sets: {variants}")
    print(f"raw_dir={raw_dir}  ->  output_dir={output_dir}\n")

    for n in sizes:
        if n not in size_params:
            raise ValueError(f"customer_counts contains {n}, but size_params has no entry for it.")
        size_cfg = size_params[n]
        raw_path = raw_dir / RAW_NAME_TEMPLATE.format(n=n, n_graphs=cfg["n_graphs"])
        if not raw_path.exists():
            print(f"[SKIP] Raw file not found: {raw_path} (set run_r: true to generate it)")
            continue

        raw_stem = raw_path.stem
        print(f"=== n={n}  <-  {raw_path} ===")
        raw_df = pd.read_csv(raw_path)

        geom = _build_geometry(raw_df, size_cfg, cfg)
        print(f"    geometry: {geom['graph_id'].nunique()} graphs, {len(geom)} rows "
              f"(depot + {size_cfg['num_stations']} stations per graph, scale={size_cfg['scale_km']} km)")

        for variant in variants:
            periods = [tuple(p) for p in cfg["time_window_variants"][variant]]
            t_min = cfg["time_norm_min"] if cfg["time_norm_min"] is not None \
                else min(s for s, _ in periods)
            t_max = cfg["time_norm_max"] if cfg["time_norm_max"] is not None \
                else max(e for _, e in periods)

            df = assign_time_windows(geom, periods,
                                     service_time=cfg["customer_service_time"],
                                     seed=cfg["tw_seed"])
            df = norm_timewindow(df, t_min=t_min, t_max=t_max)
            df = df.reindex(columns=FINAL_COLS)

            out_name = cfg["out_name_template"].format(
                raw_stem=raw_stem, n=n, n_graphs=cfg["n_graphs"],
                variant=variant, p=len(periods))
            fmt = cfg["output_format"]
            out_path = _save_df(df, output_dir, out_name, fmt)
            msg = f"    [{variant}] {len(periods)} periods -> {out_path}"

            if cfg["make_split"]:
                tr, va, te = split_data_by_graph(
                    df, val_ratio=cfg["split"]["val_ratio"],
                    test_ratio=cfg["split"]["test_ratio"], seed=cfg["split"]["seed"])
                _save_df(tr, output_dir / "train_data", out_name, fmt)
                _save_df(va, output_dir / "val_data", out_name, fmt)
                _save_df(te, output_dir / "test_data", out_name, fmt)
                msg += (f"  | split graphs train/val/test = "
                        f"{tr['graph_id'].nunique()}/{va['graph_id'].nunique()}/"
                        f"{te['graph_id'].nunique()}")
            print(msg)
        print()


def _parse_cli(argv):
    config_path, overrides = None, {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            overrides[k] = yaml.safe_load(v)
        elif config_path is None:
            config_path = a
    if config_path is None:
        sys.exit(__doc__)
    return config_path, overrides


def main():
    config_path, overrides = _parse_cli(sys.argv[1:])
    cfg = _load_config(config_path)
    if overrides:
        print(f"[cli] Overrides: {overrides}")
        cfg.update(overrides)
    start = time.time()
    run(cfg)
    print(f"Done in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()