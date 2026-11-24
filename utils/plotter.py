import os
import csv
import torch
import random


from utils.plot_functions import (
    plot_all_vehicle_figures,
    plot_all_vehicle_timeline_and_gantt,
    plot_all_vehicle_battery_and_distance,
    plot_all_vehicle_distance_and_time_profiles
)


class ResultPlotter:
    def __init__(self, cfg, run_name, device):
        self.cfg = cfg
        self.run_name = run_name
        self.device = device

    def rollout_result_to_plot_result(self, rollout_result, graph_id_dir, depot_idx=0, criteria="reward"):
        csv_filename = os.path.join(
            self.cfg["logger"]["result_analyzes_dir"],
            criteria,
            graph_id_dir,
            f"{criteria}_summary.csv"
        )

        step_records = []
        node_sequence = [depot_idx]

        total_distance = 0.0
        in_time_window_count = 0
        positive_battery_count = 0

        vehicle_distances = []
        current_vehicle_distance = 0.0
        vehicle_id = 0

        for i, info in enumerate(rollout_result.infos):
            step_info = info["step_info"]

            dist = float(step_info.dist)
            in_tw = bool(step_info.in_time_window)
            pos_bat = bool(step_info.positive_battery)

            total_distance += dist
            current_vehicle_distance += dist

            if in_tw:
                in_time_window_count += 1
            if pos_bat:
                positive_battery_count += 1

            current_step = i + 1
            to_node = int(step_info.to_node)

            record = {
                "step": current_step,
                "vehicle_id": vehicle_id,
                "from_node": int(step_info.from_node),
                "to_node": to_node,
                "dist": dist,
                "travel_time": float(getattr(step_info, "travel_time", 0.0)),
                "served_count": float(getattr(step_info, "served_count", 0.0)),
                "wait_time": float(getattr(step_info, "wait_time", 0.0)),
                "service_time": float(getattr(step_info, "service_time", 0.0)),
                "charging_time": float(getattr(step_info, "charging_time", 0.0)),
                "battery_after": float(getattr(step_info, "battery_after", 0.0)),
                "in_time_window": in_tw,
                "positive_battery": pos_bat,
                "cumulative_distance": round(total_distance, 4),
                "cumulative_in_time_window_ratio": round(in_time_window_count / current_step, 4),
                "cumulative_positive_battery_ratio": round(positive_battery_count / current_step, 4),
            }

            step_records.append(record)
            node_sequence.append(to_node)

            # Back at the depot: this vehicle's route ends
            if to_node == depot_idx:
                if current_vehicle_distance > 0:
                    vehicle_distances.append(round(current_vehicle_distance, 4))
                current_vehicle_distance = 0.0
                vehicle_id += 1

        # A final route that did not return to the depot still counts as one vehicle
        if current_vehicle_distance > 0:
            vehicle_distances.append(round(current_vehicle_distance, 4))

        if len(vehicle_distances) > 0:
            mean_vehicle_distance = sum(vehicle_distances) / len(vehicle_distances)

            std_vehicle_distance = (
                sum((d - mean_vehicle_distance) ** 2 for d in vehicle_distances) / len(vehicle_distances)
            ) ** 0.5

            mad_vehicle_distance = (
                sum(abs(d - mean_vehicle_distance) for d in vehicle_distances) / len(vehicle_distances)
            )

            cv_vehicle_distance = (
                std_vehicle_distance / mean_vehicle_distance
                if mean_vehicle_distance > 1e-8 else 0.0
            )
        else:
            mean_vehicle_distance = 0.0
            std_vehicle_distance = 0.0
            mad_vehicle_distance = 0.0
            cv_vehicle_distance = 0.0

        if step_records:
            with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
                fieldnames = list(step_records[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(step_records)

        vehicle_csv_filename = os.path.join(
            self.cfg["logger"]["result_analyzes_dir"],
            criteria,
            graph_id_dir,
            f"{criteria}_vehicle_summary.csv"
        )

        vehicle_rows = []
        for vid, dist in enumerate(vehicle_distances):
            vehicle_rows.append({
                "vehicle_id": vid,
                "vehicle_distance": round(dist, 4),
                "distance_deviation": round(dist - mean_vehicle_distance, 4),
                "abs_deviation": round(abs(dist - mean_vehicle_distance), 4),
                "mad_vehicle_distance": "",
                "cv_vehicle_distance": "",
            })

        vehicle_rows.append({
            "vehicle_id": "ALL",
            "vehicle_distance": round(mean_vehicle_distance, 4),
            "distance_deviation": "mean",
            "abs_deviation": round(std_vehicle_distance, 4),
            "mad_vehicle_distance": round(mad_vehicle_distance, 4),
            "cv_vehicle_distance": round(cv_vehicle_distance, 6),
        })

        if vehicle_rows:
            with open(vehicle_csv_filename, mode="w", newline="", encoding="utf-8") as f:
                fieldnames = list(vehicle_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(vehicle_rows)

        return {
            "node_sequence": node_sequence,
            "step_records": step_records,
        }


    def analyze_current_model_and_plot(
        self,
        trainer,
        save_name="current_model",
    ):
        trainer.model.eval()

        result_analyzes_dir = os.path.join(
            self.cfg["logger"]["result_analyzes_dir"],
            save_name,
        )
        os.makedirs(result_analyzes_dir, exist_ok=True)

        last_env = None
        last_rollout = None
        last_plot_result = None
        last_graph_id = None

        max_plot_graphs = int(
            self.cfg["logger"].get("max_plot_graphs", 1)
        )

        plot_random_seed = self.cfg["logger"].get(
            "plot_random_seed",
            None,
        )
        rng = random.Random(plot_random_seed)

        plotted_count = 0

        for batch_idx, batch in enumerate(
            trainer.loaders["test"]
        ):
            if plotted_count >= max_plot_graphs:
                break

            batch_size = trainer.get_batch_size(batch)

            if batch_size <= 0:
                continue

            if batch_size == 1:
                selected_index = 0
            else:
                selected_index = rng.randrange(batch_size)

            graph_df, graph_id = trainer.extract_graph_from_batch(
                batch=batch,
                graph_index=selected_index,
            )

            print(
                f"[Plot] save_name={save_name}, "
                f"batch_idx={batch_idx}, "
                f"batch_size={batch_size}, "
                f"selected_index={selected_index}, "
                f"graph_id={graph_id}"
            )

            env = trainer.env_builder(graph_df)

            graph_id_dir = f"graph_{graph_id}"

            graph_save_dir = os.path.join(
                result_analyzes_dir,
                graph_id_dir,
            )
            os.makedirs(graph_save_dir, exist_ok=True)

            with torch.no_grad():
                greedy_result = trainer.rollout_greedy(env)

            result = self.rollout_result_to_plot_result(
                rollout_result=greedy_result,
                graph_id_dir=graph_id_dir,
                depot_idx=0,
                criteria=save_name,
            )

            plot_all_vehicle_figures(
                env=env,
                result=result,
                save_dir=os.path.join(
                    graph_save_dir,
                    "route_figures",
                ),
                depot_idx=0,
                show_node_labels=False,
            )

            plot_all_vehicle_battery_and_distance(
                result=result,
                save_dir=os.path.join(
                    graph_save_dir,
                    "battery_distance",
                ),
                depot_idx=0,
            )

            plot_all_vehicle_timeline_and_gantt(
                result=result,
                save_dir=os.path.join(
                    graph_save_dir,
                    "timeline_gantt",
                ),
                depot_idx=0,
            )

            plot_all_vehicle_distance_and_time_profiles(
                env=env,
                result=result,
                save_dir=os.path.join(
                    graph_save_dir,
                    "distance_time_profiles",
                ),
                depot_idx=0,
            )

            last_env = env
            last_rollout = greedy_result
            last_plot_result = result
            last_graph_id = graph_id

            plotted_count += 1

        if plotted_count == 0:
            print(
                f"[Warning] No graph was plotted for "
                f"save_name={save_name}"
            )
        else:
            print(
                f"Current-model plots saved to: "
                f"{result_analyzes_dir}. "
                f"Plotted {plotted_count} graph(s)."
            )

        return {
            "save_name": save_name,
            "graph_id": last_graph_id,
            "env": last_env,
            "rollout_result": last_rollout,
            "plot_result": last_plot_result,
            "save_dir": result_analyzes_dir,
            "num_plotted_graphs": plotted_count,
        }


    def analyze_best_model_and_plot(
        self,
        trainer,
        criteria,
        strict=False,
    ):
        if trainer.checkpoint_manager is None:
            raise ValueError(
                "trainer.checkpoint_manager is required "
                "for plotting best model"
            )

        trainer.checkpoint_manager.load_best_model(
            criteria=criteria,
            strict=strict,
        )

        return self.analyze_current_model_and_plot(
            trainer=trainer,
            save_name=criteria,
        )

