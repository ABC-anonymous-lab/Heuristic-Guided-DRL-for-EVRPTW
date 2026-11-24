import os
from abc import ABC, abstractmethod

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class BaseTrainer(ABC):
    """Base class of all trainers. Subclasses implement train_step(), evaluate() and rollout_greedy()."""

    def __init__(
        self,
        cfg,
        env_builder,
        model,
        optimizer,
        loaders,
        run_name,
        device,
        logger=None,
        plotter=None,
        checkpoint_manager=None,
        total_optimizer_steps=None,
        total_rollout_steps=None,
        need_val=True,
    ):
        self.cfg = cfg
        self.env_builder = env_builder
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.run_name = run_name
        self.device = device

        self.logger = logger
        self.plotter = plotter
        self.checkpoint_manager = checkpoint_manager

        self.global_step = 0
        self.total_optimizer_steps = total_optimizer_steps
        self.total_rollout_steps = total_rollout_steps
        self.best_reward = float("-inf")
        self.best_total_distance = float("inf")
        self.need_val = need_val


    @abstractmethod
    def train_step(self, batch):
        """Run one training step on a batch and return (stats, viz_payload)."""


    def _get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


    def get_batch_size(self, batch):
        if "batch_size" in batch:
            return int(batch["batch_size"])

        if "graphs" in batch:
            return len(batch["graphs"])

        raise KeyError(
            "Cannot determine batch size: "
            "batch must contain 'batch_size' or 'graphs'."
        )


    def _get_graph_id(self, batch, graph_index=None):
        batch_size = self.get_batch_size(batch)

        if graph_index is None:
            if batch_size == 1:
                graph_index = 0
            else:
                return [
                    int(graph_id)
                    for graph_id in batch["graph_ids"]
                ]

        if not 0 <= graph_index < batch_size:
            raise IndexError(
                f"graph_index={graph_index} is outside batch size "
                f"{batch_size}"
            )

        if "graph_ids" in batch:
            graph_id = batch["graph_ids"][graph_index]
        else:
            graph_df = batch["graphs"][graph_index]

            if "graph_id" not in graph_df.columns:
                return "unknown"

            graph_id = graph_df["graph_id"].iloc[0]

        if torch.is_tensor(graph_id):
            graph_id = graph_id.item()

        return int(graph_id)


    def extract_graph_from_batch(self, batch, graph_index=0):
        batch_size = self.get_batch_size(batch)

        if batch_size <= 0:
            raise ValueError("Cannot extract graph from an empty batch.")

        if not 0 <= graph_index < batch_size:
            raise IndexError(
                f"graph_index={graph_index} is outside batch size "
                f"{batch_size}"
            )

        graph_df = batch["graphs"][graph_index]
        graph_id = self._get_graph_id(
            batch,
            graph_index=graph_index,
        )

        return graph_df, graph_id


    def _print_metrics_safely(self, *args, **kwargs):
        if self.logger is None:
            return

        external_write_mode = getattr(tqdm, "external_write_mode", None)
        if external_write_mode is None:
            self.logger.print_metrics(*args, **kwargs)
            return

        with external_write_mode():
            self.logger.print_metrics(*args, **kwargs)


    def _iter_train_batches(self, train_loader):
        repeat_k = self.cfg["train"].get("repeat_each_graph", 1)

        for batch in train_loader:
            for _ in range(repeat_k):
                yield batch


    def run(self):
        """Full run: train for num_epochs, validate every eval_interval steps (saving the best
        checkpoints by reward and by distance), then test both best checkpoints and plot routes.
        """
        train_loader = self.loaders["train"]
        val_loader = self.loaders["val"]
        test_loader = self.loaders["test"]

        num_epochs = self.cfg["train"]["num_epochs"]
        repeat_k = self.cfg["train"].get("repeat_each_graph", 1)
        num_graphs = len(train_loader)
        steps_per_epoch = num_graphs * repeat_k

        eval_interval = self.cfg["train"].get("eval_interval", 1000)
        num_validations = 0
        checkpoint_interval = self.cfg["logger"].get(
            "checkpoint_interval",
            1000,
        )
        log_interval = self.cfg["logger"].get("log_interval", 100)

        if eval_interval <= 0:
            raise ValueError("train.eval_interval must be > 0.")
        if checkpoint_interval <= 0:
            raise ValueError("logger.checkpoint_interval must be > 0.")
        if log_interval <= 0:
            raise ValueError("logger.log_interval must be > 0.")

        run_tag = self.cfg.get(
            "run_tag",
            self.cfg["train"].get("RL_algo_name", self.run_name),
        )
        device_name = str(self.device)

        banner = None
        epoch_pbar = None
        train_pbar = None

        def set_banner(status, epoch=None):
            if banner is None:
                return

            parts = [run_tag, f"device={device_name}"]
            if epoch is not None:
                parts.append(f"epoch={epoch + 1}/{num_epochs}")
            parts.append(f"step={self.global_step}")
            parts.append(status)

            banner.set_description_str(
                "[ " + " | ".join(parts) + " ]",
                refresh=True,
            )

        try:
            banner = tqdm(
                total=1,
                position=0,
                leave=False,
                dynamic_ncols=True,
                bar_format="{desc}",
                desc=(
                    f"[ {run_tag} | device={device_name} | "
                    f"step={self.global_step} | INITIALIZING ]"
                ),
            )
            banner.refresh()

            epoch_pbar = tqdm(
                range(num_epochs),
                desc="Epoch",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )

            for epoch in epoch_pbar:
                set_banner("TRAINING", epoch=epoch)

                train_pbar = tqdm(
                    self._iter_train_batches(train_loader),
                    total=steps_per_epoch,
                    desc=f"Train E{epoch + 1}",
                    position=2,
                    leave=False,
                    dynamic_ncols=True,
                )

                for batch_idx, batch in enumerate(train_pbar):
                    train_stats, viz_payload = self.train_step(batch)
                    self.global_step += 1
                    batch_graph_ids = self._get_graph_id(batch)

                    self.logger.log_metrics(
                        "train",
                        train_stats,
                        self.global_step,
                    )
                    self.logger.append_csv_log(
                        mode="train",
                        stats=train_stats,
                        epoch=epoch,
                        global_step=self.global_step,
                        batch_idx=batch_idx,
                        graph_id=batch_graph_ids,
                    )

                    train_pbar.set_postfix(
                        {
                            "R": f"{train_stats['reward']:.2f}",
                            "L": f"{int(train_stats['len'])}",
                            "step": self.global_step,
                        },
                        refresh=False,
                    )

                    if self.global_step % log_interval == 0:
                        self._print_metrics_safely(
                            "Train",
                            train_stats,
                            {
                                "epoch": epoch,
                                "step": self.global_step,
                                "batch": batch_idx,
                            },
                        )

                    # With fewer steps than eval_interval, validate once at the last step
                    is_last_step = (
                        epoch == num_epochs - 1
                        and batch_idx == steps_per_epoch - 1
                    )
                    if self.need_val and (
                        self.global_step % eval_interval == 0
                        or (is_last_step and num_validations == 0)
                    ):
                        num_validations += 1
                        set_banner("VALIDATING", epoch=epoch)

                        val_stats, val_details = self.evaluate(
                            val_loader,
                            return_details=True,
                        )

                        self.logger.log_metrics(
                            "val",
                            val_stats,
                            self.global_step,
                        )
                        self.logger.append_csv_log(
                            mode="val",
                            stats=val_stats,
                            epoch=epoch,
                            global_step=self.global_step,
                            batch_idx="",
                            graph_id="",
                        )
                        self._print_metrics_safely(
                            "Val",
                            val_stats,
                            {
                                "epoch": epoch,
                                "step": self.global_step,
                            },
                        )

                        (
                            self.best_reward,
                            self.best_total_distance,
                        ) = self.checkpoint_manager.update_best(
                            reward=val_stats["reward"],
                            total_distance=val_stats.get(
                                "total_distance",
                                None,
                            ),
                            epoch=epoch,
                            global_step=self.global_step,
                            best_reward=self.best_reward,
                            best_total_distance=self.best_total_distance,
                            served_count=val_stats.get(
                                "served_count",
                                None,
                            ),
                            cfg=self.cfg,
                        )

                        # REINFORCE trainers
                        # update the rollout baseline from the validation results
                        if hasattr(self, "maybe_update_baseline_from_eval"):
                            baseline_update_interval = self.cfg["train"].get(
                                "baseline_update_interval",
                                10,
                            )
                            if baseline_update_interval <= 0:
                                raise ValueError(
                                    "train.baseline_update_interval must be > 0."
                                )

                            if (
                                self.global_step
                                % baseline_update_interval
                                == 0
                            ):
                                set_banner("BASELINE UPDATE", epoch=epoch)

                                baseline_stats = (
                                    self.maybe_update_baseline_from_eval(
                                        eval_details=val_details,
                                        loader=val_loader,
                                    )
                                )

                                self.logger.log_metrics(
                                    "baseline",
                                    baseline_stats,
                                    self.global_step,
                                )
                                self._print_metrics_safely(
                                    "Baseline",
                                    baseline_stats,
                                    {
                                        "epoch": epoch,
                                        "step": self.global_step,
                                    },
                                )

                        set_banner("TRAINING", epoch=epoch)

                        epoch_pbar.refresh()
                        train_pbar.refresh()

                    if self.global_step % checkpoint_interval == 0:
                        self.checkpoint_manager.save_epoch_checkpoint(
                            epoch=epoch,
                            global_step=self.global_step,
                            best_reward=self.best_reward,
                        )

                train_pbar.close()
                train_pbar = None

                epoch_pbar.set_postfix(
                    {
                        "step": self.global_step,
                        "best_R": f"{self.best_reward:.2f}",
                        "best_D": (
                            f"{self.best_total_distance:.2f}"
                            if np.isfinite(self.best_total_distance)
                            else "inf"
                        ),
                    },
                    refresh=False,
                )

            set_banner("TEST: BEST REWARD")
            self.evaluate_best_checkpoint(
                test_loader,
                criteria="reward",
            )

            set_banner("TEST: BEST DISTANCE")
            self.evaluate_best_checkpoint(
                test_loader,
                criteria="total_distance",
            )

            set_banner("PLOTTING")
            try:
                self.plotter.analyze_best_model_and_plot(
                    self,
                    criteria="reward",
                )
                self.plotter.analyze_best_model_and_plot(
                    self,
                    criteria="total_distance",
                )
            except FileNotFoundError as e:
                tqdm.write(f"Model not found: {e}")

            set_banner("FINISHED")
            tqdm.write(f"[{run_tag}] Training finished.")

        finally:
            if train_pbar is not None:
                try:
                    train_pbar.close()
                except Exception:
                    pass

            if epoch_pbar is not None:
                try:
                    epoch_pbar.close()
                except Exception:
                    pass

            if banner is not None:
                try:
                    banner.close()
                except Exception:
                    pass


    def evaluate_best_checkpoint(
        self,
        loader,
        criteria="reward",
    ):
        """Load the best checkpoint for 'criteria' and evaluate it on the test set;
        writes test_details.csv and test_summary.csv under result_analyzes/test_results/<criteria>/.
        """
        if criteria not in ("reward", "total_distance"):
            raise ValueError(
                "criteria must be either 'reward' or 'total_distance', "
                f"got {criteria!r}."
            )

        tqdm.write(f"\nLoading best model by criteria: {criteria}")

        checkpoint_info = self.checkpoint_manager.load_best_model(
            criteria=criteria,
            strict=True,
            return_model_info=True,
        )

        if checkpoint_info is None:
            raise FileNotFoundError(
                f"Best checkpoint for criteria='{criteria}' was not found."
            )

        tqdm.write(
            "Loaded best checkpoint: "
            f"epoch={checkpoint_info.get('epoch', 'unknown')}, "
            f"global_step={checkpoint_info.get('global_step', 'unknown')}"
        )

        test_stats, test_details = self.evaluate(
            loader,
            return_details=True,
        )

        metric_mode = f"test/{criteria}"
        self.logger.log_metrics(
            metric_mode,
            test_stats,
            self.global_step,
        )
        self._print_metrics_safely(
            f"Test ({criteria})",
            test_stats,
        )

        test_result_dir = os.path.join(
            self.cfg["logger"]["result_analyzes_dir"],
            "test_results",
            criteria,
        )
        os.makedirs(test_result_dir, exist_ok=True)

        test_details_path = os.path.join(
            test_result_dir,
            "test_details.csv",
        )
        test_summary_path = os.path.join(
            test_result_dir,
            "test_summary.csv",
        )

        pd.DataFrame(test_details).to_csv(
            test_details_path,
            index=False,
        )
        pd.DataFrame([test_stats]).to_csv(
            test_summary_path,
            index=False,
        )

        tqdm.write(
            f"Detailed test results saved to: {test_details_path}"
        )
        tqdm.write(
            f"Test summary saved to: {test_summary_path}"
        )

        return test_stats, test_details


    def select_action(self, probs, mode="sample", return_entropy=False):
        dist = torch.distributions.Categorical(probs=probs)

        if mode == "sample":
            action = dist.sample()
        elif mode == "greedy":
            action = torch.argmax(probs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if return_entropy:
            return action, log_prob, entropy
        return action, log_prob