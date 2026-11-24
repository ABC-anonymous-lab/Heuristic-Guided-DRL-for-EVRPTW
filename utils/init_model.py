import os
import re
import glob
import yaml
import torch
import shutil
import importlib
from copy import deepcopy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# Classes are chosen in the config by dotted path and imported on demand:
#
#     train:
#         trainer_class_path: algorithm.PPO.PPO.PPOTrainer
#     model:
#         class_path: models.model_PPO.PPOGATPointerPolicyEnergy
#     env:
#         class_path: env.EVRP.EVRPEnv
#         mask_function_path: env.mask_function.mask_function
#         reward_function_path: env.reward_function.reward_function

def import_from_string(dotted_path: str):
    """Import and return an object from a dotted path such as 'package.module.ClassName'."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Invalid class path {dotted_path!r}: expected 'package.module.ClassName'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module {module_path!r} (from {dotted_path!r}): {e}") from e
    try:
        return getattr(module, attr_name)
    except AttributeError:
        raise ImportError(f"Module {module_path!r} has no attribute {attr_name!r} (from {dotted_path!r})")


def _require(section: dict, key: str, section_name: str):
    """Return section[key], or raise a clear error if the config does not set it."""
    value = section.get(key)
    if not value:
        raise KeyError(f"Config must set {section_name}.{key}")
    return value


def build_output_dirs(cfg, base_dir, run_subpath):
    """Create the output folders of one run and store their paths in cfg["logger"]."""
    logger_cfg = cfg["logger"]
    paths = {
        "base_dir": base_dir,
        "save_dir": os.path.join(base_dir, "checkpoints"),
        "result_dir": os.path.join(base_dir, "results"),
        "result_analyzes_dir": os.path.join(base_dir, "result_analyzes"),
        "run_dir": os.path.join(logger_cfg["run_root"], run_subpath),   # TensorBoard logs
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    logger_cfg.update(paths)
    return paths


def backup_source_files(cfg, dest_dir):
    paths = list(cfg["logger"].get("backup_paths", []))

    if not paths:
        print("[backup] logger.backup_paths is not set; source files are not backed up for this run.")
        return

    seen = set()
    for pattern in paths:
        matches = glob.glob(pattern)
        if not matches:
            print(f"[backup] Path to back up not found: {pattern}")
            continue

        for src in matches:
            src = os.path.normpath(src)
            if src in seen:
                continue
            seen.add(src)

            try:
                if os.path.isdir(src):
                    shutil.copytree(
                        src,
                        os.path.join(dest_dir, os.path.basename(src)),
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                else:
                    shutil.copy2(src, dest_dir)
            except Exception as e:
                print(f"[backup] Failed to copy {src}: {e}")


def _infer_problem_size_tag(cfg):
    # Problem size from a path component of train_csv such as "/n50/" or "_n50_"
    path = str(cfg["dataset"]["train_csv"]).replace("\\", "/")
    m = re.search(r"(?:^|[/_])n(\d+)(?:[/_.])", path)
    if m:
        return f"n{m.group(1)}"
    return None


def init_output(cfg):
    algo_name = cfg["train"]["RL_algo_name"]
    timestamp = cfg.get("run_group_timestamp") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    size_tag = _infer_problem_size_tag(cfg)          # "n50" / "n100" / None
    size_dir = size_tag if size_tag else "n_unknown"

    group_parts = ["evrp", algo_name]
    if size_tag:
        group_parts.append(size_tag)
    group_parts.append(timestamp)
    run_group = "_".join(group_parts)

    seed = cfg.get("seed", None)
    seed_dir = f"seed{int(seed)}" if seed is not None else "seed_na"
    run_name = f"{run_group}_{seed_dir}" if seed is not None else run_group
    cfg["run_name"] = run_name

    output_root = cfg["logger"]["output_root"]
    base_dir = os.path.join(output_root, size_dir, run_group, seed_dir)
    run_subpath = os.path.join(size_dir, run_group, seed_dir)

    paths = build_output_dirs(cfg, base_dir=base_dir, run_subpath=run_subpath)

    config_to_save = deepcopy(cfg)
    config_path = os.path.join(
        paths["base_dir"],
        f"{run_name}_config.yaml",
    )

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_to_save,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    backup_source_files(cfg, cfg["logger"]["result_dir"])

    writer = SummaryWriter(
        log_dir=cfg["logger"]["run_dir"],
    )

    return run_name, writer


def build_env_builder(cfg):
    env_cls = import_from_string(_require(cfg["env"], "class_path", "env"))

    # Resolve the functions once, not for every graph
    mask_fn_path = cfg["env"].get("mask_function_path")
    reward_fn_path = cfg["env"].get("reward_function_path")
    mask_fn = import_from_string(mask_fn_path) if mask_fn_path else None
    reward_fn = import_from_string(reward_fn_path) if reward_fn_path else None

    def make_env(graph_df):
        env_kwargs = dict(cfg["env"])
        env_kwargs.pop("class_path", None)
        env_kwargs.pop("mask_function_path", None)
        env_kwargs.pop("reward_function_path", None)

        if mask_fn is not None:
            env_kwargs["mask_function"] = mask_fn
        if reward_fn is not None:
            env_kwargs["reward_function"] = reward_fn

        env_kwargs.pop("time_windows")

        return env_cls(df=graph_df, **env_kwargs)

    return make_env


def build_algorithm(algo_name, **kwargs):
    """Build the trainer given by train.trainer_class_path.

    algo_name (train.RL_algo_name) is only used to name the output folders.
    """
    cfg = kwargs.get("cfg", {})
    trainer_cls = import_from_string(_require(cfg.get("train", {}), "trainer_class_path", "train"))
    return trainer_cls(**kwargs)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return total_params, trainable_params
    

def build_model_and_optimizer(cfg):
    device = torch.device(cfg["device"])

    model_cfg = cfg["model"]

    model_cls = import_from_string(_require(model_cfg, "class_path", "model"))
    model_kwargs = dict(model_cfg.get("kwargs", {}))

    model = model_cls(**model_kwargs).to(device)

    count_parameters(model)

    opt_name = cfg["model"]["optimizer"]["name"].lower()

    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["model"]["optimizer"]["lr"]),
            weight_decay=cfg["model"]["optimizer"]["weight_decay"],
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["model"]["optimizer"]["lr"]),
            weight_decay=cfg["model"]["optimizer"]["weight_decay"],
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg["model"]["optimizer"]["lr"]),
            weight_decay=cfg["model"]["optimizer"]["weight_decay"],
            momentum=cfg["model"]["optimizer"]["momentum"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg['model']['optimizer']['name']}")

    return model, optimizer
