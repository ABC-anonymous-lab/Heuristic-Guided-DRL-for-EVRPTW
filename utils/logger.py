import os
import csv
import json
import time
from contextlib import contextmanager
from tqdm import tqdm
from datetime import datetime

import torch


class TrainingLogger:
    def __init__(self, cfg, run_name, writer=None):
        self.cfg = cfg
        self.run_name = run_name
        self.writer = writer

        self.csv_fieldnames = [
            "time",
            "epoch",
            "global_step",
            "batch_idx",
            "graph_id",
            "reward",
            "len",
            "served_count",
            "in_time_window_ratio",
            "positive_battery_ratio",
            "total_distance",
        ]

        log_dir = cfg["logger"]["result_dir"]
        os.makedirs(log_dir, exist_ok=True)

        self.csv_paths = {
            "train": os.path.join(log_dir, f"{run_name}_train_metrics.csv"),
            "val": os.path.join(log_dir, f"{run_name}_val_metrics.csv"),
            "test": os.path.join(log_dir, f"{run_name}_test_metrics.csv"),
        }

        self._init_csv_files()

        self.timing = {}
        self.timing_counts = {}
        self._run_start_wall = time.perf_counter()
        self._run_start_process = time.process_time()
        self.timing_path = os.path.join(log_dir, f"{run_name}_timing.json")
        self._is_cuda = str(cfg.get("device", "cpu")).startswith("cuda")

    def _init_csv_files(self):
        for path in self.csv_paths.values():
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                    writer.writeheader()

    @contextmanager
    def timer(self, phase: str, cuda_sync: bool = False):
        if cuda_sync and self._is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda_sync and self._is_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.timing[phase] = self.timing.get(phase, 0.0) + dt
            self.timing_counts[phase] = self.timing_counts.get(phase, 0) + 1

    def timing_summary(self):
        total_wall = time.perf_counter() - self._run_start_wall
        phases = {}
        for phase, total in self.timing.items():
            n = self.timing_counts.get(phase, 0)
            phases[phase] = {
                "total_s": round(total, 3),
                "count": n,
                "mean_s": round(total / n, 6) if n else None,
                "share_of_wall": round(total / total_wall, 4) if total_wall > 0 else None,
            }
        return {
            "run_name": self.run_name,
            "device": self.cfg.get("device", "cpu"),
            "wall_time_total_s": round(total_wall, 3),
            "process_time_main_s": round(time.process_time() - self._run_start_process, 3),
            "phases": phases,
        }

    def save_timing(self, extra: dict = None):
        summary = self.timing_summary()
        if extra:
            summary.update(extra)
        with open(self.timing_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def append_csv_log(
        self,
        mode,
        stats,
        epoch=None,
        global_step=None,
        batch_idx=None,
        graph_id=None,
    ):
        required_keys = [
            "reward",
            "len",
            "served_count",
            "in_time_window_ratio",
            "positive_battery_ratio",
            "total_distance",
        ]

        for k in required_keys:
            if k not in stats:
                raise ValueError(f"[CSV LOG ERROR] Missing required key: {k}")

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch if epoch is not None else "",
            "global_step": global_step if global_step is not None else "",
            "batch_idx": batch_idx if batch_idx is not None else "",
            "graph_id": graph_id if graph_id is not None else "",
            "reward": stats["reward"],
            "len": stats["len"],
            "in_time_window_ratio": stats["in_time_window_ratio"],
            "positive_battery_ratio": stats["positive_battery_ratio"],
            "total_distance": stats["total_distance"],
            "served_count": stats["served_count"],
        }

        with open(self.csv_paths[mode], "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)
            f.flush()

    def log_metrics(self, prefix, metrics, step):
        if self.writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def print_metrics(self, prefix, metrics, extra=None):
        parts = [f"[{prefix}]"]
        if extra:
            for k, v in extra.items():
                parts.append(f"{k}={v}")
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        tqdm.write(" ".join(parts))