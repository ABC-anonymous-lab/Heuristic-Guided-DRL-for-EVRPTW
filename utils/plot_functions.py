import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib


def _get_xy_from_env(env):
    if not hasattr(env, "xy_km"):
        raise AttributeError("env has no xy_km (node coordinates in km)")
    xy = np.asarray(env.xy_km)

    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Expected coordinates of shape [num_nodes, 2], got {xy.shape}")

    return xy[:, :2]


def _get_node_type_from_env(env):
    """
    0 = depot
    1 = customer
    2 = charging station
    """
    if not hasattr(env, "node_type"):
        raise AttributeError("env has no node_type (0 = depot, 1 = customer, 2 = charging station)")
    return np.asarray(env.node_type)


def plot_raw_map(env, save_path=None, show_node_labels=False):
    """
    - depot
    - customer
    - charging station
    """
    xy = _get_xy_from_env(env)
    node_type = _get_node_type_from_env(env)

    depot_idx = np.where(node_type == 0)[0]
    customer_idx = np.where(node_type == 1)[0]
    charging_idx = np.where(node_type == 2)[0]

    fig, ax = plt.subplots(figsize=(8, 8))

    if len(charging_idx) > 0:
        ax.scatter(
            xy[charging_idx, 0],
            xy[charging_idx, 1],
            s=70,
            marker="^",
            color="blue",
            label="Charging station"
        )

    if len(depot_idx) > 0:
        ax.scatter(
            xy[depot_idx, 0],
            xy[depot_idx, 1],
            s=150,
            marker="*",
            color="red",
            label="Depot"
        )

    legend_elements = []

    if len(customer_idx) > 0:
        tw_mid = env.time_window_mid[customer_idx]
        tw_mid = tw_mid.astype(int)

        unique_tw = np.unique(tw_mid)
        n_bins = len(unique_tw)

        tw_to_id = {v: i for i, v in enumerate(unique_tw)}
        bin_ids = np.array([tw_to_id[v] for v in tw_mid])

        cmap = matplotlib.colormaps["viridis"].resampled(n_bins)
        colors = cmap(bin_ids)

        ax.scatter(
            xy[customer_idx, 0],
            xy[customer_idx, 1],
            s=45,
            marker="o",
            c=colors,
            edgecolors="black",
            linewidths=0.5
        )

        for i, tw in enumerate(unique_tw):
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=f"TW {tw}",
                    markerfacecolor=cmap(i),
                    markeredgecolor='black',
                    markersize=8
                )
            )

    if len(charging_idx) > 0:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='^',
                color='w',
                label='Charging station',
                markerfacecolor='blue',
                markersize=8
            )
        )

    if len(depot_idx) > 0:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='*',
                color='w',
                label='Depot',
                markerfacecolor='red',
                markersize=10
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=7  # seven in a row
        )

    if show_node_labels:
        for node_id, (x, y) in enumerate(xy):
            ax.annotate(
                str(node_id),
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8
            )

    ax.set_title("Raw Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis("equal")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_vehicle_node_order(env, vehicle_route, save_path=None, show_node_labels=False):
    xy = _get_xy_from_env(env)
    seq = vehicle_route["node_sequence"]
    vid = vehicle_route["vehicle_id"]

    route_xy = xy[seq]

    plt.figure(figsize=(8, 8))

    # Background: all nodes
    plt.scatter(xy[:, 0], xy[:, 1], s=30, alpha=0.4, color="gray", label="All nodes")
    plt.plot(
        route_xy[:, 0],
        route_xy[:, 1],
        linewidth=1.5,
        color="black",
        label=f"Vehicle {vid} path"
    )

    if len(seq) >= 2 and seq[0] == 0 and seq[-1] == 0:
        inner_seq = seq[1:-1]
    else:
        inner_seq = seq

    if len(inner_seq) > 0:
        inner_xy = xy[inner_seq]
        plt.scatter(
            inner_xy[:, 0],
            inner_xy[:, 1],
            s=35,
            marker="o",
            color="black"
        )

    # depot
    plt.scatter(xy[0, 0], xy[0, 1], s=120, marker="*", color="red", label="Depot")

    for order, node in enumerate(inner_seq, start=1):
        x, y = xy[node]
        plt.annotate(
            str(order),
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8
        )

    if show_node_labels:
        for node_id, (x, y) in enumerate(xy):
            plt.annotate(
                f"n{node_id}",
                (x, y),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=7,
                alpha=0.7
            )

    plt.title(f"Vehicle {vid} Node Order")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_vehicle_timewindow(env, vehicle_route, save_path=None, show_node_labels=False):
    xy = _get_xy_from_env(env)
    seq = vehicle_route["node_sequence"]
    step_records = vehicle_route["step_records"]
    vid = vehicle_route["vehicle_id"]

    plt.figure(figsize=(8, 8))
    plt.scatter(xy[:, 0], xy[:, 1], s=25, alpha=0.15, color="gray", label="All nodes")
    plt.scatter(xy[0, 0], xy[0, 1], s=120, marker="s", color="blue", label="Depot")

    green_added = False
    red_added = False

    for i, step in enumerate(step_records):
        node = seq[i + 1]
        x, y = xy[node]

        if node == 0:
            continue

        if bool(step["in_time_window"]):
            label = "In time window" if not green_added else None
            plt.scatter(x, y, s=90, color="green", label=label)
            green_added = True
        else:
            label = "Out of time window" if not red_added else None
            plt.scatter(x, y, s=90, color="red", label=label)
            red_added = True

    if show_node_labels: # node id
        visited_nodes = set(seq)
        for node in visited_nodes:
            x, y = xy[node]
            plt.annotate(
                str(node),
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8
            )

    plt.title(f"Vehicle {vid} Time Window Status")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_vehicle_battery(vehicle_route, save_path=None):
    vid = vehicle_route["vehicle_id"]
    seq = vehicle_route["node_sequence"]
    step_records = vehicle_route["step_records"]

    if len(step_records) == 0:
        print(f"Vehicle {vid} has no step_records, skipping battery plot")
        return

    battery_after = [step["battery_after"] for step in step_records]
    x_labels = seq[1:]

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(battery_after)), battery_after, marker="o", color="black")

    for i, node in enumerate(x_labels):
        plt.annotate(
            str(node),
            (i, battery_after[i]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8
        )

    plt.title(f"Vehicle {vid} Battery After Each Move")
    plt.xlabel("Step index")
    plt.ylabel("Battery")
    plt.grid(True)
    plt.xticks(range(len(x_labels)), [str(n) for n in x_labels], rotation=45)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_all_vehicle_figures(
    env,
    result,
    save_dir,
    depot_idx=0,
    show_node_labels=False
):
    node_sequence = result["node_sequence"]
    step_records = result["step_records"]

    vehicle_routes = split_routes_by_depot_zero(
        node_sequence=node_sequence,
        step_records=step_records,
        depot_idx=depot_idx
    )

    os.makedirs(save_dir, exist_ok=True)
    raw_map_path = os.path.join(save_dir, "raw_map.png")
    plot_raw_map(
        env=env,
        save_path=raw_map_path,
        show_node_labels=show_node_labels
    )

    for vr in vehicle_routes:
        vid = vr["vehicle_id"]

        node_order_path = os.path.join(save_dir, f"vehicle_{vid}_node_order.png")
        tw_path = os.path.join(save_dir, f"vehicle_{vid}_timewindow.png")

        plot_vehicle_node_order(
            env=env,
            vehicle_route=vr,
            save_path=node_order_path,
            show_node_labels=show_node_labels
        )

        plot_vehicle_timewindow(
            env=env,
            vehicle_route=vr,
            save_path=tw_path,
            show_node_labels=show_node_labels
        )

    return vehicle_routes


def split_routes_by_depot_zero(node_sequence, step_records, depot_idx=0):
    if len(node_sequence) == 0:
        return []

    routes = []
    curr_nodes = [node_sequence[0]]
    curr_steps = []

    for i, step in enumerate(step_records):
        next_node = node_sequence[i + 1]
        curr_nodes.append(next_node)
        curr_steps.append(step)

        if next_node == depot_idx:
            routes.append({
                "vehicle_id": len(routes) + 1,
                "node_sequence": curr_nodes.copy(),
                "step_records": curr_steps.copy(),
            })

            if i + 1 < len(step_records):
                curr_nodes = [depot_idx]
                curr_steps = []
            else:
                curr_nodes = []
                curr_steps = []

    if len(curr_nodes) > 0:
        routes.append({
            "vehicle_id": len(routes) + 1,
            "node_sequence": curr_nodes.copy(),
            "step_records": curr_steps.copy(),
        })

    return routes


def plot_vehicle_node_timeline(vehicle_route, save_path=None, annotate_wait=True):
    vid = vehicle_route["vehicle_id"]
    seq = vehicle_route["node_sequence"]
    step_records = vehicle_route["step_records"]

    if len(seq) <= 1:
        print(f"Vehicle {vid} has too few visits, skipping node timeline")
        return

    x = np.arange(len(seq))
    y = np.array(seq)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, marker="o", linewidth=1.5)

    for i, node in enumerate(seq):
        plt.annotate(
            str(node),
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8
        )

    if annotate_wait and len(step_records) > 0:
        for i, step in enumerate(step_records, start=1):
            wait_t = float(step.get("wait_time", 0.0))
            charge_t = float(step.get("charging_time", 0.0))

            extra = []
            if wait_t > 1e-8:
                extra.append(f"W={wait_t:.1f}")
            if charge_t > 1e-8:
                extra.append(f"C={charge_t:.1f}")

            if extra:
                plt.annotate(
                    "\n".join(extra),
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, -28),
                    ha="center",
                    fontsize=7
                )

    plt.title(f"Vehicle {vid} Node Visit Timeline")
    plt.xlabel("Visit order / step")
    plt.ylabel("Node ID")
    plt.grid(True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_vehicle_event_gantt(vehicle_route, save_path=None):
    vid = vehicle_route["vehicle_id"]
    step_records = vehicle_route["step_records"]

    if len(step_records) == 0:
        print(f"Vehicle {vid} has no step_records, skipping Gantt chart")
        return

    plt.figure(figsize=(14, max(4, 0.45 * len(step_records) + 1)))

    y_positions = np.arange(len(step_records))
    bar_h = 0.6

    current_time = 0.0

    for i, step in enumerate(step_records):
        travel_t = float(step.get("travel_time", 0.0))
        wait_t = float(step.get("wait_time", 0.0))
        charging_t = float(step.get("charging_time", 0.0))

        start_t = current_time

        # travel
        if travel_t > 1e-8:
            plt.barh(i, travel_t, left=start_t, height=bar_h, label="travel" if i == 0 else None)
            start_t += travel_t

        # wait
        if wait_t > 1e-8:
            plt.barh(i, wait_t, left=start_t, height=bar_h, hatch="//", label="wait" if i == 0 else None)
            start_t += wait_t

        # charging
        if charging_t > 1e-8:
            plt.barh(i, charging_t, left=start_t, height=bar_h, hatch="xx", label="charging" if i == 0 else None)
            start_t += charging_t

        from_node = step.get("from_node", None)
        to_node = step.get("to_node", None)
        text_x = start_t + max(0.5, 0.01 * max(1.0, start_t))
        plt.text(
            text_x,
            i,
            f"{from_node}->{to_node}",
            va="center",
            fontsize=8
        )

        current_time = start_t

    plt.yticks(y_positions, [f"step {i}" for i in range(len(step_records))])
    plt.xlabel("Cumulative time")
    plt.ylabel("Step")
    plt.title(f"Vehicle {vid} Event Gantt / Time Decomposition")
    plt.grid(True, axis="x")
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_all_vehicle_timeline_and_gantt(result, save_dir, depot_idx=0):
    node_sequence = result["node_sequence"]
    step_records = result["step_records"]

    vehicle_routes = split_routes_by_depot_zero(
        node_sequence=node_sequence,
        step_records=step_records,
        depot_idx=depot_idx
    )

    os.makedirs(save_dir, exist_ok=True)

    for vr in vehicle_routes:
        vid = vr["vehicle_id"]

        timeline_path = os.path.join(save_dir, f"vehicle_{vid}_node_timeline.png")
        gantt_path = os.path.join(save_dir, f"vehicle_{vid}_event_gantt.png")

        plot_vehicle_node_timeline(
            vehicle_route=vr,
            save_path=timeline_path,
            annotate_wait=True
        )

        plot_vehicle_event_gantt(
            vehicle_route=vr,
            save_path=gantt_path
        )

    return vehicle_routes


def plot_vehicle_cumulative_distance(vehicle_route, save_path=None):
    vid = vehicle_route["vehicle_id"]
    seq = vehicle_route["node_sequence"]
    step_records = vehicle_route["step_records"]

    if len(step_records) == 0:
        print(f"Vehicle {vid} has no step_records, skipping cumulative distance plot")
        return

    step_dist = [float(step.get("dist", 0.0)) for step in step_records]
    cumulative_dist = np.cumsum(step_dist)
    x_labels = seq[1:]

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(cumulative_dist)), cumulative_dist, marker="o")

    for i, node in enumerate(x_labels):
        plt.annotate(
            str(node),
            (i, cumulative_dist[i]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8
        )

    plt.title(f"Vehicle {vid} Cumulative Distance")
    plt.xlabel("Step index")
    plt.ylabel("Cumulative distance")
    plt.grid(True)
    plt.xticks(range(len(x_labels)), [str(n) for n in x_labels], rotation=45)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_all_vehicle_battery_and_distance(result, save_dir="sim_plots_distance", depot_idx=0):
    node_sequence = result["node_sequence"]
    step_records = result["step_records"]

    vehicle_routes = split_routes_by_depot_zero(
        node_sequence=node_sequence,
        step_records=step_records,
        depot_idx=depot_idx
    )

    os.makedirs(save_dir, exist_ok=True)

    for vr in vehicle_routes:
        vid = vr["vehicle_id"]

        battery_path = os.path.join(save_dir, f"vehicle_{vid}_battery.png")
        distance_path = os.path.join(save_dir, f"vehicle_{vid}_cumulative_distance.png")

        plot_vehicle_battery(
            vehicle_route=vr,
            save_path=battery_path
        )

        plot_vehicle_cumulative_distance(
            vehicle_route=vr,
            save_path=distance_path
        )

        # print(f"Saved Vehicle {vid} battery plot to: {battery_path}")
        # print(f"Saved Vehicle {vid} cumulative distance plot to: {distance_path}")

    return vehicle_routes


def plot_all_vehicle_distance_and_time_profiles(env, result, save_dir, depot_idx=0):
    node_sequence = result["node_sequence"]
    step_records = result["step_records"]

    vehicle_routes = split_routes_by_depot_zero(
        node_sequence=node_sequence,
        step_records=step_records,
        depot_idx=depot_idx
    )

    os.makedirs(save_dir, exist_ok=True)

    for vr in vehicle_routes:
        vid = vr["vehicle_id"]

        distance_path = os.path.join(save_dir, f"vehicle_{vid}_distance_profile.png")
        time_path = os.path.join(save_dir, f"vehicle_{vid}_time_profile.png")

        plot_vehicle_distance_profile(
            env=env,
            vehicle_route=vr,
            save_path=distance_path
        )

        plot_vehicle_time_profile(
            env=env,
            vehicle_route=vr,
            save_path=time_path
        )

    return vehicle_routes


def plot_vehicle_distance_profile(env, vehicle_route, save_path=None):
    seq = vehicle_route["node_sequence"]
    vid = vehicle_route["vehicle_id"]

    if len(seq) < 2:
        return

    step_ids = []
    distances = []
    step_labels = []

    for i in range(len(seq) - 1):
        from_node = seq[i]
        to_node = seq[i + 1]

        d = float(env.dist[from_node][to_node])

        step_ids.append(i + 1)
        distances.append(d)
        step_labels.append(f"{from_node}->{to_node}")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        step_ids,
        distances,
        marker="o",
        linewidth=1.8,
        label="Distance (km)",
        color="blue"
    )

    ax.set_title(f"Vehicle {vid} Step Distance Profile")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (km)")
    ax.set_xticks(step_ids)
    # ax.set_xticklabels(step_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_vehicle_time_profile(env, vehicle_route, save_path=None):
    seq = vehicle_route["node_sequence"]
    vid = vehicle_route["vehicle_id"]

    if len(seq) < 2:
        return

    step_ids = []
    travel_times = []
    step_labels = []

    for i in range(len(seq) - 1):
        from_node = seq[i]
        to_node = seq[i + 1]

        d = float(env.dist[from_node][to_node])
        t = float(d / env.speed)

        step_ids.append(i + 1)
        travel_times.append(t)
        step_labels.append(f"{from_node}->{to_node}")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        step_ids,
        travel_times,
        marker="s",
        linewidth=1.8,
        label="Travel time (min)",
        color="orange",
    )

    ax.set_title(f"Vehicle {vid} Step Travel Time Profile")
    ax.set_xlabel("Step")
    ax.set_ylabel("Travel time (min)")
    ax.set_xticks(step_ids)
    # ax.set_xticklabels(step_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)