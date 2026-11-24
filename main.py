import os
import gc
import argparse
import yaml
import torch
from copy import deepcopy
from datetime import datetime

from utils.seed import seed_everything
from utils.init_model import build_env_builder, build_model_and_optimizer, build_algorithm, init_output
from utils.dataloader import build_dataloaders
from utils.plotter import ResultPlotter
from utils.logger import TrainingLogger
from utils.checkpoint_manager import CheckpointManager


class CudaConfigOOM(RuntimeError):
    pass


def _is_cuda_oom(exc):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    if oom_cls and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="EVRP DRL training and testing. Runs one or more configs in order; each config is run once per seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  One config:        python main.py DRL_results/ID08_n50_HG-PPO_w1.0/seed42/config.yaml\n"
            "  Several configs:   python main.py DRL_results/ID07_n50_PPO_w1.0/seed42/config.yaml DRL_results/ID08_n50_HG-PPO_w1.0/seed42/config.yaml\n"
            "  Wildcard:          python main.py DRL_results/ID08_*/seed*/config.yaml   (expanded by the shell)\n"
            "  Override seeds:    python main.py DRL_results/ID08_n50_HG-PPO_w1.0/seed42/config.yaml --seeds 1 2 3\n"
        ),
    )
    parser.add_argument(
        "configs", nargs="+",
        help="One or more config YAML files, run in the given order",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Override the seeds/seed setting of every config (the YAML files are not modified)",
    )
    return parser.parse_args()


def load_config(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                config_data = yaml.safe_load(file)
                return config_data
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
    else:
        print(f"Error: The file '{file_path}' was not found.")


def get_seed_list(cfg, override=None):
    if override:
        return [int(s) for s in override]
    if "seeds" in cfg and cfg["seeds"]:
        seeds = cfg["seeds"]
        if not isinstance(seeds, (list, tuple)):
            seeds = [seeds]
        return [int(s) for s in seeds]
    return [int(cfg["seed"])]


def run_one_seed(base_cfg, seed, cfg_idx=1, total_configs=1, seed_idx=1, total_seeds=1):
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg.pop("seeds", None)
    cfg["seed_start_timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    cfg["run_tag"] = (
        f"c{cfg_idx}/{total_configs}|"
        f"{cfg['train']['RL_algo_name']}|"
        f"s{seed}({seed_idx}/{total_seeds})"
    )

    run_name = None
    writer = None
    model = optimizer = None
    loaders = None
    logger = plotter = checkpoint_manager = trainer = None
    oom = False

    try:
        seed_everything(int(seed))

        # run_name ends with the seed
        run_name, writer = init_output(cfg)

        loaders = build_dataloaders(cfg)
        num_batches_per_epoch = len(loaders["train"])
        rollout_steps_per_epoch = (
            num_batches_per_epoch
            * cfg["train"].get("repeat_each_graph", 1)
        )
        optimizer_steps_per_epoch = (
            rollout_steps_per_epoch
            * cfg["train"].get("ppo_epochs", 1)
        )

        total_rollout_steps = (
            rollout_steps_per_epoch
            * cfg["train"]["num_epochs"]
        )
        total_optimizer_steps = (
            optimizer_steps_per_epoch
            * cfg["train"]["num_epochs"]
        )
        cfg["train"]["total_rollout_steps"] = total_rollout_steps
        cfg["train"]["total_optimizer_steps"] = total_optimizer_steps

        model, optimizer = build_model_and_optimizer(cfg)
        checkpoint_manager = CheckpointManager(
            cfg=cfg, run_name=run_name, model=model,
            optimizer=optimizer,
            device=device,
        )

        logger = TrainingLogger(cfg=cfg, run_name=run_name, writer=writer)
        plotter = ResultPlotter(cfg=cfg, run_name=run_name, device=device)

        env_builder = build_env_builder(cfg)
        trainer = build_algorithm(
            algo_name=cfg["train"]["RL_algo_name"],
            cfg=cfg,
            env_builder=env_builder,
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            run_name=run_name,
            device=device,
            logger=logger,
            plotter=plotter,
            checkpoint_manager=checkpoint_manager,
            total_optimizer_steps=total_optimizer_steps,
            total_rollout_steps=total_rollout_steps,
            need_val=True
        )

        try:
            trainer.run()
        finally:
            if logger is not None:
                try:
                    timing = logger.save_timing(extra={"seed": int(seed)})
                    print(
                        f"[seed {seed}] wall time: "
                        f"{timing['wall_time_total_s']:.1f}s "
                        f"-> {logger.timing_path}"
                    )
                except Exception as timing_exc:
                    print(
                        f"[warning] seed {seed}: failed to save timing: "
                        f"{type(timing_exc).__name__}: {timing_exc}"
                    )

            # HG trainers keep a persistent OR-Tools worker pool; close it after each seed.
            repair_fn = getattr(trainer, "ortools_repair_fn", None)
            if repair_fn is not None and hasattr(repair_fn, "close"):
                try:
                    repair_fn.close()
                except Exception as close_exc:
                    print(
                        f"[warning] seed {seed}: failed to close OR-Tools pool: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )

    except Exception as e:
        if _is_cuda_oom(e):
            oom = True
            print(
                f"\n[OOM] config {cfg_idx}/{total_configs} | seed {seed}: "
                f"CUDA out of memory, aborting this run.\n  {type(e).__name__}: {e}"
            )
        else:
            raise
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        del trainer, model, optimizer, loaders, logger, plotter, checkpoint_manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if oom:
        raise CudaConfigOOM(f"seed {seed}: CUDA out of memory")

    return run_name


def main():
    args = parse_args()

    total_configs = len(args.configs)
    finished_runs = []
    skipped_configs = []
    for cfg_idx, config_path in enumerate(args.configs, start=1):
        base_cfg = load_config(config_path)
        if base_cfg is None:
            print(f"Skipping config that could not be loaded [{cfg_idx}/{total_configs}]: {config_path}")
            continue

        seeds = get_seed_list(base_cfg, override=args.seeds)
        total_seeds = len(seeds)

        base_cfg["config_path"] = config_path

        # All seeds of one config share a run-group timestamp, so init_output puts
        # them in the same outputs/<size>/<run_group>/ folder (one subfolder per seed).
        base_cfg["run_group_timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print(f"\nConfig [{cfg_idx}/{total_configs}]: {config_path}")
        print(f"  algo = {base_cfg['train']['RL_algo_name']}, seeds = {seeds}")

        for seed_idx, seed in enumerate(seeds, start=1):
            try:
                run_name = run_one_seed(
                    base_cfg, seed,
                    cfg_idx=cfg_idx, total_configs=total_configs,
                    seed_idx=seed_idx, total_seeds=total_seeds,
                )
            except CudaConfigOOM as e:
                print(
                    f"\n[skip] config [{cfg_idx}/{total_configs}] {config_path} "
                    f"ran out of CUDA memory at seed {seed}; skipping its remaining seeds "
                    f"and moving to the next config.\n  {e}"
                )
                skipped_configs.append((config_path, seed))
                break

            finished_runs.append((config_path, seed, run_name))

    print("\nAll runs finished:")
    for config_path, seed, run_name in finished_runs:
        print(f"  {config_path} | seed {seed:>6d} -> {run_name}")

    if skipped_configs:
        print("\nSkipped due to CUDA OOM:")
        for config_path, seed in skipped_configs:
            print(f"  {config_path} | first OOM at seed {seed}")


if __name__ == "__main__":
    main()