import os
import torch
from tqdm import tqdm


class CheckpointManager:
    def __init__(self, cfg, run_name, model, optimizer=None, device="cpu"):
        self.cfg = cfg
        self.run_name = run_name
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self._fallback_best_key = (float("-inf"), float("-inf"))


    @property
    def save_dir(self):
        return self.cfg["logger"]["save_dir"]


    def save(self, save_path, epoch, global_step, best_reward):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        opt_state = self.optimizer.state_dict() if self.optimizer is not None else None

        ckpt = {
            "epoch":              epoch,
            "global_step":        global_step,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": opt_state,
            "best_reward":        best_reward,
            "config":             self.cfg,
        }
        torch.save(ckpt, save_path)



    def save_epoch_checkpoint(self, epoch, global_step, best_reward):
        save_path = os.path.join(
            self.cfg["logger"]["save_dir"],
            f"{self.run_name}_epoch_{epoch + 1}.pt"
        )
        self.save(
            save_path=save_path,
            epoch=epoch,
            global_step=global_step,
            best_reward=best_reward,
        )
        return save_path


    def save_best(self, epoch, global_step, best_reward, criteria, seperate=False):
        if not seperate:
            best_path = os.path.join(
                self.save_dir,
                criteria,
                f"{self.run_name}_best.pt"
            )
        else:
            best_path = os.path.join(
                self.save_dir,
                criteria,
                f"{self.run_name}_step_{global_step}_{best_reward:.2f}.pt"
            )

        self.save(
            save_path=best_path,
            epoch=epoch,
            global_step=global_step,
            best_reward=best_reward,
        )
        return best_path


    def load(self, load_path, strict=False):
        print(f"Loading checkpoint from {load_path}")
        ckpt = torch.load(load_path, map_location=self.device)

        missing_keys, unexpected_keys = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=strict)
        if not strict:
            if missing_keys:   print(f"[Warning] Missing keys: {missing_keys}")
            if unexpected_keys: print(f"[Warning] Unexpected keys: {unexpected_keys}")

        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(opt_state)
            except Exception as e:
                print(f"[Warning] Failed to load optimizer state: {e}")

        epoch       = ckpt.get("epoch",       0)
        global_step = ckpt.get("global_step", 0)
        best_reward = ckpt.get("best_reward", float("-inf"))

        print(f"Checkpoint loaded: epoch={epoch}, global_step={global_step}, best_reward={best_reward:.2f}")
        return {"epoch": epoch, "global_step": global_step,
                "best_reward": best_reward, "config": ckpt.get("config")}


    def update_best(
        self,
        reward,
        total_distance,
        epoch,
        global_step,
        best_reward,
        best_total_distance,
        served_count,
        graph_df=None,
        greedy_result=None,
        trainer=None,
        cfg=None,
    ):
        updated_total_distance = best_total_distance
        updated_reward = best_reward

        min_served_threshold = float(cfg["logger"]["min_served_threshold"])

        # Best total distance model
        if total_distance < best_total_distance and served_count >= min_served_threshold:
            self.save_best(
                epoch=epoch,
                global_step=global_step,
                best_reward=total_distance,
                criteria="total_distance"
            )
            self.save_best(
                epoch=epoch,
                global_step=global_step,
                best_reward=total_distance,
                criteria="total_distance",
                seperate=True
            )

            tqdm.write(
                f"Saved best total distance model at global step {global_step} "
                f"total distance {total_distance:.2f} and served rate {served_count:.3f}"
            )

            updated_total_distance = total_distance

            if trainer is not None and graph_df is not None and greedy_result is not None:
                trainer.save_best_route_visualizations(
                    graph_df=graph_df,
                    greedy_result=greedy_result,
                    epoch=epoch,
                    global_step=global_step,
                    criteria="total_distance",
                )

        # Best reward model
        if reward > best_reward and served_count >= min_served_threshold:
            self.save_best(
                epoch=epoch,
                global_step=global_step,
                best_reward=reward,
                criteria="reward"
            )
            self.save_best(
                epoch=epoch,
                global_step=global_step,
                best_reward=reward,
                criteria="reward",
                seperate=True
            )

            tqdm.write(
                f"Saved best reward model at global step {global_step} "
                f"with reward {reward:.2f} and served rate {served_count:.3f}"
            )

            updated_reward = reward

            if trainer is not None and graph_df is not None and greedy_result is not None:
                trainer.save_best_route_visualizations(
                    graph_df=graph_df,
                    greedy_result=greedy_result,
                    epoch=epoch,
                    global_step=global_step,
                    criteria="reward",
                )

        # Fallback best: ignores min_served_threshold
        fallback_key = (float(served_count), float(reward))
        if fallback_key > self._fallback_best_key:
            self._fallback_best_key = fallback_key
            self.save_best(
                epoch=epoch,
                global_step=global_step,
                best_reward=reward,
                criteria="fallback",
            )

        return updated_reward, updated_total_distance


    def load_best_model(self, criteria="reward", strict=True, return_model_info=False,
                        allow_fallback=True):
        best_model_path = os.path.join(
            self.save_dir,
            criteria,
            f"{self.run_name}_best.pt"
        )

        if not os.path.exists(best_model_path):
            fallback_path = os.path.join(
                self.save_dir,
                "fallback",
                f"{self.run_name}_best.pt"
            )
            if allow_fallback and os.path.exists(fallback_path):
                print(
                    f"[Warning] No best '{criteria}' model (min_served_threshold was never "
                    f"reached during training). Using the fallback model instead: {fallback_path}\n"
                    f"          !! Results come from a model below the threshold; mark them "
                    f"as such when reporting."
                )
                best_model_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"best model not found: {best_model_path} "
                    f"(no fallback model available; allow_fallback={allow_fallback})"
                )

        if return_model_info:
            return self.load(best_model_path, strict=strict)

        self.load(best_model_path, strict=strict)
        self.model.to(self.device)
        self.model.eval()
        return best_model_path