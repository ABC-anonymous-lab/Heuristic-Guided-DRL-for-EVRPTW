import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from env.mask_function import mask_function
from env.reward_function import reward_function


@dataclass
class StepInfo:
    """What happened in one step (used for metrics and plots)."""
    dist: float
    from_node: int
    to_node: int
    travel_time: float
    wait_time: float
    service_time: float
    charging_time: float
    battery_after: float
    in_time_window: bool
    positive_battery: bool
    served_count: float          # fraction of customers served so far


class EVRPEnv:

    DEPOT = 0
    CUSTOMER = 1
    STATION = 2

    def __init__(
        self,
        *,
        df,
        n_vehicles: int = 4,
        max_steps: int = 200,
        operating_time_min: int = 240,
        operating_time_max: int = 1320,
        min_battery: float = 10.0,            # reserve kept when going to a customer (see mask_function)
        battery_max: float = 100.0,
        charged_battery_max: float = 80.0,    # charge-to level at a station
        energy_rate: float = 1,               # km per battery unit
        charging_rate: float = 1,             # battery units per minute
        speed: float = 60,                    # km per hour
        distance_weight=1.0,                  # weight of the distance term in the reward
        station_allow_battery_ratio=0.5,      # stations allowed only below this battery fraction
        reward_function=reward_function,
        mask_function=mask_function,
    ):
        self.battery_max = float(battery_max)
        self.charged_battery_max = float(charged_battery_max)
        self.min_battery = float(min_battery)
        self.energy_rate = float(energy_rate)
        self.charging_rate = float(charging_rate)
        self.speed = float(speed / 60)        # km per minute
        self.n_vehicles = int(n_vehicles)
        self.station_allow_battery_ratio = station_allow_battery_ratio
        self.max_steps = int(max_steps)
        self.operating_time_min = operating_time_min
        self.operating_time_max = operating_time_max
        self.distance_weight = distance_weight

        self.mask_function = mask_function
        self.reward_function = reward_function

        self.vehicle_used = 1
        self.is_in_timewindow = True

        required = ["node_id", "type", "x", "y", "x_km", "y_km", "ready_time", "due_date",
                    "ready_time_norm", "due_date_norm", "service_time"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing}. Found columns: {list(df.columns)}")

        self.depot_idx = 0
        self.xy_km = np.stack([df["x_km"].to_numpy(dtype=np.float32),
                               df["y_km"].to_numpy(dtype=np.float32)], axis=1)
        self.xy = np.stack([df["x"].to_numpy(dtype=np.float32),
                            df["y"].to_numpy(dtype=np.float32)], axis=1)
        self.ready = df["ready_time"].to_numpy(dtype=np.float32)
        self.due = df["due_date"].to_numpy(dtype=np.float32)
        self.ready_norm = df["ready_time_norm"].to_numpy(dtype=np.float32)
        self.due_norm = df["due_date_norm"].to_numpy(dtype=np.float32)
        self.time_window_mid = (self.ready + self.due) / 2.0
        self.service = df["service_time"].to_numpy(dtype=np.float32)
        self.node_type = df["type"].to_numpy(dtype=np.int32)
        self.N = self.xy.shape[0]

        if len(np.where(self.node_type == self.DEPOT)[0]) != 1:
            raise ValueError(f"Expected exactly 1 depot (type 0), found {int((self.node_type == self.DEPOT).sum())}.")

        # Euclidean distance matrix (km)
        diff = self.xy_km[:, None, :] - self.xy_km[None, :, :]
        self.dist = np.linalg.norm(diff, axis=-1).astype(np.float32)

        self.customer_idx = np.where(self.node_type == self.CUSTOMER)[0]
        if len(self.customer_idx) == 0:
            raise ValueError("No customers found (type == 1).")

        self.reset()


    def norm_time(self, time):
        return (time - self.operating_time_min) / (self.operating_time_max - self.operating_time_min)


    def _update_lookahead(self):
        batt_arrival = (self.battery - self.dist[self.current_node] / self.energy_rate).astype(np.float32)
        batt_arrival = np.clip(batt_arrival, self.min_battery, self.battery_max)
        self.batt_arrival_at_all_other_node = (batt_arrival - self.min_battery) / (self.battery_max - self.min_battery)

        arrival_time = self.time + self.dist[self.current_node] / self.speed
        self.time_arrival_at_all_other_node = self.norm_time(arrival_time).astype(np.float32)


    def reset(self) -> Dict[str, Any]:
        self.timestamp = 0
        self.current_node = 0
        self.prev_node = 0
        self.time = self.operating_time_min
        self.battery = self.battery_max
        self.served_rate = 0.0

        # Counters used in the PN-LSTM reward
        self.station_visits = 0
        self.negative_battery_sum = 0.0

        self._update_lookahead()

        # The first vehicle is already in use
        self.vehicles_left = self.n_vehicles - 1

        self.visited_customer = np.zeros(self.N, dtype=bool)
        self.total_distance = 0.0
        self.done = False

        self.current_vehicle_distance = 0.0
        self.vehicle_distances = []

        return self._get_obs()


    def _finish_current_vehicle(self):
        if self.current_vehicle_distance > 0:
            self.vehicle_distances.append(float(self.current_vehicle_distance))
        self.current_vehicle_distance = 0.0


    def reset_vehicles(self):
        self._finish_current_vehicle()
        self.current_node = 0
        self.battery = self.battery_max
        self.time = self.operating_time_min


    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        a = int(action)
        if a < 0 or a >= self.N:
            raise ValueError(f"Invalid action {a}, must be in [0, {self.N-1}]")

        d = float(self.dist[self.current_node, a])
        self.total_distance += d
        self.current_vehicle_distance += d

        self.prev_node = self.current_node
        self.last_valid_mask = self.mask_function(self)   # used by the reward
        self.current_node = a

        # Travel
        travel_time = d / self.speed
        energy_used = d / self.energy_rate
        batt_after = float(self.battery) - energy_used
        self.battery = batt_after
        is_positive_battery = batt_after > 0
        self.negative_battery_sum += max(-float(batt_after), 0.0)

        # Time window: customers must be reached by their due date
        arrival_time = float(self.time + travel_time)
        if self.node_type[a] == self.CUSTOMER:
            self.is_in_timewindow = arrival_time <= self.due[a]
        else:
            self.is_in_timewindow = True
        self.time = arrival_time
        wait_time = 0.0
        if arrival_time < float(self.ready[a]):
            wait_time = float(self.ready[a]) - arrival_time
            arrival_time = float(self.ready[a])
            self.time = arrival_time

        service_time = float(self.service[a])
        depart_time = arrival_time + service_time
        self.time = depart_time
        self.timestamp += 1

        # Station: charge to charged_battery_max; charging time delays the departure
        charging_time_cost = 0.0
        if self.node_type[a] == self.STATION:
            charging_energy = max(0.0, float(self.charged_battery_max - batt_after))
            charging_time_cost = charging_energy / float(self.charging_rate)
            depart_time += charging_time_cost
            self.battery = self.battery + charging_energy
            self.time = depart_time
            self.station_visits += 1

        if self.node_type[a] == self.CUSTOMER:
            self.visited_customer[a] = True

        # Depot: the current vehicle's route ends, the next vehicle starts
        if self.node_type[a] == self.DEPOT:
            if self.vehicles_left > 0:
                self.vehicle_used += 1
                self.reset_vehicles()
            self.vehicles_left -= 1

        self._update_lookahead()

        all_served = bool(np.all(self.visited_customer[self.customer_idx]))
        self.served_rate = np.sum(self.visited_customer[self.customer_idx]) / len(self.customer_idx)

        # Termination
        reached_max_steps = self.timestamp > self.max_steps
        finished_at_depot = all_served and self.current_node == self.depot_idx
        out_of_vehicles = self.vehicles_left < 0
        returned_with_no_vehicle_left = out_of_vehicles and self.current_node == self.depot_idx

        if finished_at_depot:
            self.done = True
            termination_reason = "all_served"
        elif returned_with_no_vehicle_left:
            self.done = True
            termination_reason = "out_of_vehicles"
        elif reached_max_steps:
            self.done = True
            termination_reason = "max_steps"
        elif not np.any(self.get_action_mask()):
            self.done = True
            termination_reason = "no_feasible_action"
        else:
            termination_reason = None

        reward = self.get_reward(
            step_distance=d,
            all_served=all_served,
            use_new_vehicle=self.node_type[a] == self.DEPOT and self.done == False
        )

        info = StepInfo(
            dist=d,
            from_node=self.prev_node,
            to_node=self.current_node,
            travel_time=travel_time,
            wait_time=wait_time,
            service_time=service_time,
            charging_time=charging_time_cost,
            battery_after=float(self.battery),
            in_time_window=self.is_in_timewindow,
            positive_battery=is_positive_battery,
            served_count=self.served_rate,
        )

        if self.done:
            self._finish_current_vehicle()

        return self._get_obs(), float(reward), bool(self.done), {
            "step_info": info, "all_served": all_served, "termination_reason": termination_reason,
        }

    def get_action_mask(self):
        return self.mask_function(self)

    def get_reward(self, **kwargs):
        return self.reward_function(self, **kwargs)

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "dist": self.dist.copy(),                                   # [N, N]
            "xy": self.xy.copy(),                                       # [N, 2]
            "ready_norm": self.ready_norm.copy(),                       # [N]
            "due_norm": self.due_norm.copy(),                           # [N]
            "node_type": self.node_type.copy(),                         # [N]
            "visited_customer": self.visited_customer.copy(),           # [N]
            "current_node": int(self.current_node),
            "time_norm": float(self.norm_time(self.time)),
            "battery_norm": float((self.battery - self.min_battery) / (self.battery_max - self.min_battery)),
            "vehicles_left": int(self.vehicles_left),
            "batt_arrival": self.batt_arrival_at_all_other_node,         # [N]
            "time_arrival_at_all_other_node": self.time_arrival_at_all_other_node.copy(),   # [N]
        }