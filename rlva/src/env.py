from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_gym_backend():
    try:
        import gymnasium as gym  # type: ignore

        return gym
    except ImportError:
        try:
            import gym  # type: ignore

            return gym
        except ImportError:
            return None


def _make_cartpole_backend():
    gym_backend = _load_gym_backend()
    if gym_backend is None:
        raise ImportError("CartPole requires gymnasium or gym. Install `gymnasium[classic-control]` to enable this environment.")
    return gym_backend.make("CartPole-v1")


def _make_lunarlander_backend():
    gym_backend = _load_gym_backend()
    if gym_backend is None:
        raise ImportError("LunarLander requires gymnasium or gym. Install `gymnasium[box2d]` to enable this environment.")
    for env_id in ["LunarLander-v3", "LunarLander-v2"]:
        try:
            return gym_backend.make(env_id)
        except Exception:
            continue
    raise ImportError("LunarLander backend is unavailable. Install `gymnasium[box2d]` for this environment.")


def _gym_reset(backend: Any, seed: Optional[int]) -> Tuple[List[float], Dict[str, Any]]:
    reset_out = backend.reset(seed=seed)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    return list(obs), dict(info)


def _gym_step(backend: Any, action: int) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
    step_out = backend.step(action)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
    elif isinstance(step_out, tuple) and len(step_out) == 4:
        obs, reward, done, info = step_out
        terminated, truncated = bool(done), False
    else:
        raise ValueError("Unexpected CartPole step output: {0}".format(step_out))
    return list(obs), float(reward), bool(terminated), bool(truncated), dict(info)


def is_env_available(env_name: str) -> bool:
    if env_name == "cartpole":
        return _load_gym_backend() is not None
    if env_name == "lunarlander":
        try:
            backend = _make_lunarlander_backend()
            close_fn = getattr(backend, "close", None)
            if callable(close_fn):
                close_fn()
            return True
        except Exception:
            return False
    if env_name != "cartpole":
        return True
    return _load_gym_backend() is not None


class BaseDecisionEnv(object):
    ENV_NAME = "base"
    ACTION_LABELS = ("a0", "a1", "a2")

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.current_step = 0

    def reset(self, seed: Optional[int] = None) -> List[float]:
        raise NotImplementedError

    def step(self, action_id: int) -> Tuple[List[float], float, Dict[str, float]]:
        raise NotImplementedError

    def get_state(self) -> List[float]:
        raise NotImplementedError

    def sample_action(self) -> int:
        return self.rng.randint(0, 2)

    @property
    def state_dim(self) -> int:
        return 4

    def action_label(self, action_id: int) -> str:
        if action_id < 0 or action_id >= len(self.ACTION_LABELS):
            raise ValueError("Unknown action_id={0}".format(action_id))
        return self.ACTION_LABELS[action_id]


@dataclass
class InterventionSpec(object):
    name: str
    start_step: int
    end_step: int
    params: Dict[str, float]

    def is_active(self, step: int) -> bool:
        return self.start_step <= step <= self.end_step

    def starts_at(self, step: int) -> bool:
        return step == self.start_step


@dataclass
class Task(object):
    task_id: int
    arrival_step: int
    processing_time: int


class QueueSchedulingEnv(BaseDecisionEnv):
    ENV_NAME = "queue"
    ACTION_LABELS = ("shortest", "longest", "fcfs")

    def __init__(
        self,
        capacity: int = 50,
        arrival_rate: float = 1.4,
        processing_time_range: Tuple[int, int] = (1, 12),
        seed: Optional[int] = None,
        initial_queue_size: int = 6,
    ) -> None:
        super(QueueSchedulingEnv, self).__init__(seed=seed)
        self.capacity = capacity
        self.arrival_rate = arrival_rate
        self.processing_time_range = processing_time_range
        self.initial_queue_size = initial_queue_size
        self._next_task_id = 0
        self.queue: List[Task] = []

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            self.rng.seed(seed)
        self.current_step = 0
        self._next_task_id = 0
        self.queue = []
        for _ in range(self.initial_queue_size):
            processing_time = self.rng.randint(*self.processing_time_range)
            waited_steps = self.rng.randint(1, 5)
            self.queue.append(
                Task(
                    task_id=self._alloc_task_id(),
                    arrival_step=-waited_steps,
                    processing_time=processing_time,
                )
            )
        return self.get_state()

    def get_state(self) -> List[float]:
        q_t = len(self.queue)
        if q_t == 0:
            mu_t = 0.0
            sigma_t = 0.0
        else:
            processing_times = [task.processing_time for task in self.queue]
            mu_t = sum(processing_times) / q_t
            variance = sum((x - mu_t) ** 2 for x in processing_times) / q_t
            sigma_t = math.sqrt(variance)
        rho_t = min(1.0, q_t / float(self.capacity))
        return [float(q_t), float(mu_t), float(sigma_t), float(rho_t)]

    def step(self, action_id: int) -> Tuple[List[float], float, Dict[str, float]]:
        arrivals = self._arrive_tasks()
        avg_wait = self._queue_average_wait()
        selected_task = self._select_task(action_id)
        if selected_task is not None:
            self.queue.remove(selected_task)

        reward = -avg_wait
        self.current_step += 1
        next_state = self.get_state()
        info = {
            "arrivals": float(arrivals),
            "avg_wait": float(avg_wait),
            "queue_len": float(len(self.queue)),
            "action_label": self.action_label(action_id),
        }
        if selected_task is not None:
            info["scheduled_processing_time"] = float(selected_task.processing_time)
        return next_state, reward, info

    def _arrive_tasks(self) -> int:
        num_arrivals = self._sample_poisson(self.arrival_rate)
        for _ in range(num_arrivals):
            self.queue.append(
                Task(
                    task_id=self._alloc_task_id(),
                    arrival_step=self.current_step,
                    processing_time=self.rng.randint(*self.processing_time_range),
                )
            )
        return num_arrivals

    def _queue_average_wait(self) -> float:
        if not self.queue:
            return 0.0
        waits = [self.current_step - task.arrival_step for task in self.queue]
        return sum(waits) / float(len(waits))

    def _select_task(self, action_id: int) -> Optional[Task]:
        if not self.queue:
            return None
        if action_id == 0:
            return min(self.queue, key=lambda task: (task.processing_time, task.arrival_step, task.task_id))
        if action_id == 1:
            return max(self.queue, key=lambda task: (task.processing_time, -task.arrival_step, -task.task_id))
        if action_id == 2:
            return min(self.queue, key=lambda task: (task.arrival_step, task.task_id))
        raise ValueError("Unknown action_id={0}".format(action_id))

    def _alloc_task_id(self) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _sample_poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while p > limit:
            k += 1
            p *= self.rng.random()
        return k - 1


class InventoryControlEnv(BaseDecisionEnv):
    ENV_NAME = "inventory"
    ACTION_LABELS = ("low_restock", "balanced_restock", "high_restock")

    def __init__(
        self,
        max_inventory: int = 70,
        base_demand: float = 12.0,
        seed: Optional[int] = None,
    ) -> None:
        super(InventoryControlEnv, self).__init__(seed=seed)
        self.max_inventory = max_inventory
        self.base_demand = base_demand
        self.inventory = 0.0
        self.backlog = 0.0
        self.last_demand = base_demand
        self.recent_demands: List[float] = []

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            self.rng.seed(seed)
        self.current_step = 0
        self.inventory = float(self.max_inventory * 0.55)
        self.backlog = 0.0
        self.recent_demands = [self.base_demand + self.rng.uniform(-2.0, 2.0) for _ in range(5)]
        self.last_demand = self.recent_demands[-1]
        return self.get_state()

    def get_state(self) -> List[float]:
        demand_mean = sum(self.recent_demands) / float(len(self.recent_demands))
        demand_var = sum((value - demand_mean) ** 2 for value in self.recent_demands) / float(len(self.recent_demands))
        demand_std = math.sqrt(demand_var)
        stock_ratio = min(1.0, self.inventory / float(self.max_inventory))
        return [float(self.backlog), float(demand_mean), float(demand_std), float(stock_ratio)]

    def step(self, action_id: int) -> Tuple[List[float], float, Dict[str, float]]:
        reorder_levels = [6.0, 12.0, 18.0]
        if action_id < 0 or action_id >= len(reorder_levels):
            raise ValueError("Unknown action_id={0}".format(action_id))

        order_amount = reorder_levels[action_id]
        seasonal = 3.0 * math.sin(self.current_step / 6.0)
        demand = max(0.0, self.base_demand + seasonal + self.rng.uniform(-4.0, 4.0))

        available = self.inventory + order_amount
        served = min(available, demand + self.backlog)
        unmet = max(0.0, demand + self.backlog - served)
        next_inventory = max(0.0, available - served)

        holding_cost = 0.08 * next_inventory
        stockout_cost = 0.35 * unmet
        order_cost = 0.03 * order_amount
        reward = -(holding_cost + stockout_cost + order_cost)

        self.inventory = min(float(self.max_inventory), next_inventory)
        self.backlog = unmet
        self.last_demand = demand
        self.recent_demands = (self.recent_demands + [demand])[-5:]
        self.current_step += 1

        info = {
            "demand": float(demand),
            "order_amount": float(order_amount),
            "inventory": float(self.inventory),
            "backlog": float(self.backlog),
            "action_label": self.action_label(action_id),
        }
        return self.get_state(), reward, info


class TrafficSignalEnv(BaseDecisionEnv):
    ENV_NAME = "traffic"
    ACTION_LABELS = ("prioritize_ns", "prioritize_ew", "balanced")

    def __init__(
        self,
        seed: Optional[int] = None,
        max_queue: float = 80.0,
    ) -> None:
        super(TrafficSignalEnv, self).__init__(seed=seed)
        self.max_queue = max_queue
        self.queue_ns = 0.0
        self.queue_ew = 0.0
        self.arrival_history: List[float] = []

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            self.rng.seed(seed)
        self.current_step = 0
        self.queue_ns = 12.0
        self.queue_ew = 10.0
        self.arrival_history = [18.0 + self.rng.uniform(-3.0, 3.0) for _ in range(5)]
        return self.get_state()

    def get_state(self) -> List[float]:
        total_queue = self.queue_ns + self.queue_ew
        mean_arrival = sum(self.arrival_history) / float(len(self.arrival_history))
        variance = sum((value - mean_arrival) ** 2 for value in self.arrival_history) / float(len(self.arrival_history))
        imbalance = abs(self.queue_ns - self.queue_ew)
        congestion = min(1.0, total_queue / self.max_queue)
        return [float(total_queue), float(mean_arrival), float(math.sqrt(variance) + 0.1 * imbalance), float(congestion)]

    def step(self, action_id: int) -> Tuple[List[float], float, Dict[str, float]]:
        base_arrival = 18.0 + 5.0 * math.sin(self.current_step / 5.0)
        arrivals_ns = max(0.0, base_arrival + self.rng.uniform(-4.0, 4.0))
        arrivals_ew = max(0.0, 16.0 + 4.0 * math.cos(self.current_step / 7.0) + self.rng.uniform(-4.0, 4.0))

        if action_id == 0:
            service_ns, service_ew = 17.0, 9.0
        elif action_id == 1:
            service_ns, service_ew = 9.0, 17.0
        elif action_id == 2:
            service_ns, service_ew = 13.0, 13.0
        else:
            raise ValueError("Unknown action_id={0}".format(action_id))

        self.queue_ns = max(0.0, self.queue_ns + arrivals_ns - service_ns)
        self.queue_ew = max(0.0, self.queue_ew + arrivals_ew - service_ew)
        total_queue = self.queue_ns + self.queue_ew
        imbalance = abs(self.queue_ns - self.queue_ew)
        reward = -(0.07 * total_queue + 0.02 * imbalance)

        total_arrivals = arrivals_ns + arrivals_ew
        self.arrival_history = (self.arrival_history + [total_arrivals])[-5:]
        self.current_step += 1

        info = {
            "arrivals_ns": float(arrivals_ns),
            "arrivals_ew": float(arrivals_ew),
            "queue_ns": float(self.queue_ns),
            "queue_ew": float(self.queue_ew),
            "action_label": self.action_label(action_id),
        }
        return self.get_state(), reward, info


class CartPoleEnv(BaseDecisionEnv):
    ENV_NAME = "cartpole"
    ACTION_LABELS = ("push_left", "push_right", "stabilize")

    def __init__(self, seed: Optional[int] = None) -> None:
        super(CartPoleEnv, self).__init__(seed=seed)
        self.backend = _make_cartpole_backend()
        self._last_obs = [0.0, 0.0, 0.0, 0.0]
        self.gravity = float(getattr(self.backend.unwrapped, "gravity", 9.8))
        self.force_mag = float(getattr(self.backend.unwrapped, "force_mag", 10.0))

    @property
    def state_dim(self) -> int:
        return 4

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            self.rng.seed(seed)
        self.current_step = 0
        obs, _info = _gym_reset(self.backend, seed=seed)
        self._last_obs = obs
        self._sync_backend_params()
        return self.get_state()

    def get_state(self) -> List[float]:
        x, x_dot, theta, theta_dot = self._last_obs
        return [
            _clip(float(x) / 2.4, -1.0, 1.0),
            _clip(float(x_dot) / 3.0, -1.0, 1.0),
            _clip(float(theta) / 0.2095, -1.0, 1.0),
            _clip(float(theta_dot) / 3.5, -1.0, 1.0),
        ]

    def step(self, action_id: int) -> Tuple[List[float], float, Dict[str, float]]:
        backend_action = self._map_action(action_id)
        obs, reward, terminated, truncated, info = _gym_step(self.backend, backend_action)
        self._last_obs = obs
        done = bool(terminated or truncated)
        shaped_reward = float(reward) - (1.0 if done else 0.0)
        self.current_step += 1

        out_info: Dict[str, float] = dict(info)
        out_info["action_label"] = self.action_label(action_id)
        out_info["backend_action"] = float(backend_action)
        out_info["episode_done"] = float(done)
        out_info["gravity"] = float(getattr(self.backend.unwrapped, "gravity", self.gravity))
        out_info["force_mag"] = float(getattr(self.backend.unwrapped, "force_mag", self.force_mag))

        if done:
            reset_obs, _reset_info = _gym_reset(self.backend, seed=self.rng.randint(0, 10 ** 6))
            self._last_obs = reset_obs

        return self.get_state(), shaped_reward, out_info

    def _map_action(self, action_id: int) -> int:
        if action_id == 0:
            return 0
        if action_id == 1:
            return 1
        if action_id == 2:
            theta = float(self._last_obs[2])
            theta_dot = float(self._last_obs[3])
            return 0 if theta + 0.25 * theta_dot < 0.0 else 1
        raise ValueError("Unknown action_id={0}".format(action_id))

    def _sync_backend_params(self) -> None:
        if hasattr(self.backend.unwrapped, "gravity"):
            self.backend.unwrapped.gravity = float(self.gravity)
        if hasattr(self.backend.unwrapped, "force_mag"):
            self.backend.unwrapped.force_mag = float(self.force_mag)


class LunarLanderEnv(BaseDecisionEnv):
    ENV_NAME = "lunarlander"
    ACTION_LABELS = ("left_thruster", "main_engine", "right_thruster")

    def __init__(self, seed: Optional[int] = None) -> None:
        super(LunarLanderEnv, self).__init__(seed=seed)
        self.backend = _make_lunarlander_backend()
        self._last_obs = [0.0] * 8
        self.main_engine_failure_prob = 0.0
        self.lateral_engine_swap_prob = 0.0

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            self.rng.seed(seed)
        self.current_step = 0
        obs, _info = _gym_reset(self.backend, seed=seed)
        self._last_obs = obs
        return self.get_state()

    def get_state(self) -> List[float]:
        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = self._last_obs
        lateral_offset = _clip(float(x) / 1.2, -1.0, 1.0)
        altitude = _clip(float(y) / 1.5, -1.0, 1.0)
        descent_rate = _clip(float(-vy) / 2.0, -1.0, 1.0)
        attitude = _clip(float(angle + 0.35 * ang_vel), -1.0, 1.0)
        leg_contact_bonus = 0.15 * float(left_leg + right_leg)
        return [
            lateral_offset,
            altitude,
            descent_rate,
            _clip(attitude + leg_contact_bonus, -1.0, 1.0),
        ]

    def step(self, action_id: int) -> Tuple[List[float], float, Dict[str, float]]:
        backend_action = self._map_action(action_id)
        obs, reward, terminated, truncated, info = _gym_step(self.backend, backend_action)
        self._last_obs = obs
        done = bool(terminated or truncated)
        self.current_step += 1

        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = self._last_obs
        stability_bonus = 0.1 * float(left_leg + right_leg) - 0.05 * abs(float(angle)) - 0.03 * abs(float(ang_vel))
        shaped_reward = float(reward) + stability_bonus

        out_info: Dict[str, float] = dict(info)
        out_info["action_label"] = self.action_label(action_id)
        out_info["backend_action"] = float(backend_action)
        out_info["episode_done"] = float(done)
        out_info["x"] = float(x)
        out_info["y"] = float(y)
        out_info["vx"] = float(vx)
        out_info["vy"] = float(vy)
        out_info["angle"] = float(angle)
        out_info["ang_vel"] = float(ang_vel)
        out_info["left_leg_contact"] = float(left_leg)
        out_info["right_leg_contact"] = float(right_leg)
        out_info["main_engine_failure_prob"] = float(self.main_engine_failure_prob)
        out_info["lateral_engine_swap_prob"] = float(self.lateral_engine_swap_prob)

        if done:
            reset_obs, _reset_info = _gym_reset(self.backend, seed=self.rng.randint(0, 10 ** 6))
            self._last_obs = reset_obs

        return self.get_state(), shaped_reward, out_info

    def _map_action(self, action_id: int) -> int:
        if action_id == 0:
            backend_action = 1
        elif action_id == 1:
            backend_action = 2
        elif action_id == 2:
            backend_action = 3
        else:
            raise ValueError("Unknown action_id={0}".format(action_id))

        if backend_action == 2 and self.main_engine_failure_prob > 0.0 and self.rng.random() < self.main_engine_failure_prob:
            return 0
        if backend_action in (1, 3) and self.lateral_engine_swap_prob > 0.0 and self.rng.random() < self.lateral_engine_swap_prob:
            return 3 if backend_action == 1 else 1
        return backend_action


class InterventionEnv(BaseDecisionEnv):
    ENV_NAME = "intervention"

    def __init__(self, base_env: BaseDecisionEnv, interventions: Sequence[InterventionSpec]) -> None:
        super(InterventionEnv, self).__init__(seed=None)
        self.base_env = base_env
        self.ENV_NAME = base_env.ENV_NAME
        self.interventions = list(interventions)
        self._base_params = self._capture_base_params()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def _capture_base_params(self) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for attr in [
            "arrival_rate",
            "base_demand",
            "queue_ns",
            "queue_ew",
            "gravity",
            "force_mag",
            "main_engine_failure_prob",
            "lateral_engine_swap_prob",
        ]:
            if hasattr(self.base_env, attr):
                params[attr] = float(getattr(self.base_env, attr))
        return params

    @property
    def state_dim(self) -> int:
        return self.base_env.state_dim

    def action_label(self, action_id: int) -> str:
        return self.base_env.action_label(action_id)

    def get_state(self) -> List[float]:
        return self.base_env.get_state()

    def reset(self, seed: Optional[int] = None) -> List[float]:
        if seed is not None:
            self.rng.seed(seed)
        state = self.base_env.reset(seed=seed)
        self.current_step = self.base_env.current_step
        self._base_params = self._capture_base_params()
        return self._apply_observation_interventions(state, self._active_interventions())

    def _active_interventions(self) -> List[InterventionSpec]:
        return [spec for spec in self.interventions if spec.is_active(self.base_env.current_step)]

    def _starting_interventions(self) -> List[InterventionSpec]:
        return [spec for spec in self.interventions if spec.starts_at(self.base_env.current_step)]

    def _apply_baseline(self) -> None:
        for key, value in self._base_params.items():
            if key in ("queue_ns", "queue_ew"):
                continue
            setattr(self.base_env, key, value)

    def _apply_pre_step_interventions(self, active_specs: Sequence[InterventionSpec]) -> None:
        self._apply_baseline()
        for spec in active_specs:
            if isinstance(self.base_env, QueueSchedulingEnv) and "arrival_rate_scale" in spec.params:
                self.base_env.arrival_rate = self._base_params.get("arrival_rate", self.base_env.arrival_rate) * float(spec.params["arrival_rate_scale"])
            elif isinstance(self.base_env, InventoryControlEnv) and "demand_delta" in spec.params:
                self.base_env.base_demand = self._base_params.get("base_demand", self.base_env.base_demand) + float(spec.params["demand_delta"])
            elif isinstance(self.base_env, TrafficSignalEnv):
                self.base_env.queue_ns += float(spec.params.get("queue_ns_delta", 0.0))
                self.base_env.queue_ew += float(spec.params.get("queue_ew_delta", 0.0))
            elif isinstance(self.base_env, CartPoleEnv):
                if "gravity_scale" in spec.params:
                    self.base_env.gravity = self._base_params.get("gravity", self.base_env.gravity) * float(spec.params["gravity_scale"])
                if "force_mag_scale" in spec.params:
                    self.base_env.force_mag = self._base_params.get("force_mag", self.base_env.force_mag) * float(spec.params["force_mag_scale"])
                self.base_env._sync_backend_params()
            elif isinstance(self.base_env, LunarLanderEnv):
                if "main_engine_failure_prob" in spec.params:
                    self.base_env.main_engine_failure_prob = float(spec.params["main_engine_failure_prob"])
                if "lateral_engine_swap_prob" in spec.params:
                    self.base_env.lateral_engine_swap_prob = float(spec.params["lateral_engine_swap_prob"])

    def _apply_action_interventions(self, action_id: int, active_specs: Sequence[InterventionSpec]) -> Tuple[int, bool]:
        executed_action = int(action_id)
        action_corrupted = False
        for spec in active_specs:
            swap_prob = float(spec.params.get("action_flip_prob", 0.0))
            if swap_prob > 0.0 and self.rng.random() < swap_prob:
                candidates = [candidate for candidate in range(3) if candidate != executed_action]
                executed_action = candidates[self.rng.randint(0, len(candidates) - 1)]
                action_corrupted = True
        return executed_action, action_corrupted

    def _apply_observation_interventions(self, state: Sequence[float], active_specs: Sequence[InterventionSpec]) -> List[float]:
        observed = [float(value) for value in state]
        for spec in active_specs:
            noise_std = float(spec.params.get("obs_noise_std", 0.0))
            scale = float(spec.params.get("obs_scale", 1.0))
            bias = float(spec.params.get("obs_bias", 0.0))
            if scale != 1.0 or bias != 0.0:
                observed = [_clip(scale * value + bias, -1.0, 1.0) for value in observed]
            if noise_std > 0.0:
                observed = [_clip(value + self.rng.gauss(0.0, noise_std), -1.0, 1.0) for value in observed]
        return observed

    def _apply_reward_interventions(self, reward: float, active_specs: Sequence[InterventionSpec]) -> float:
        adjusted_reward = float(reward)
        for spec in active_specs:
            adjusted_reward *= float(spec.params.get("reward_scale", 1.0))
            adjusted_reward += float(spec.params.get("reward_bias", 0.0))
        return adjusted_reward

    def step(self, action_id: int) -> Tuple[List[float], float, Dict[str, float]]:
        active_specs = self._active_interventions()
        starting_specs = self._starting_interventions()
        self._apply_pre_step_interventions(active_specs)
        executed_action, action_corrupted = self._apply_action_interventions(action_id, active_specs)
        next_state, reward, info = self.base_env.step(executed_action)
        self.current_step = self.base_env.current_step
        self._apply_baseline()

        active_names = [spec.name for spec in active_specs]
        info = dict(info)
        info["policy_action"] = float(action_id)
        info["executed_action"] = float(executed_action)
        info["action_corrupted"] = float(action_corrupted)
        info["intervention_active"] = float(bool(active_specs))
        info["intervention_start"] = float(bool(starting_specs))
        info["intervention_name"] = "|".join(active_names) if active_names else "none"
        observed_state = self._apply_observation_interventions(next_state, active_specs)
        adjusted_reward = self._apply_reward_interventions(reward, active_specs)
        return observed_state, adjusted_reward, info


def build_default_interventions(env_name: str, horizon: int, profile: str = "standard") -> List[InterventionSpec]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if profile not in ("standard", "mild", "severe"):
        raise ValueError("Unknown intervention profile: {0}".format(profile))

    start = max(5, horizon // 3)
    mid = max(start + 5, horizon // 2)
    end = max(mid + 5, (2 * horizon) // 3)
    profile_scale = {"mild": 0.65, "standard": 1.0, "severe": 1.35}[profile]

    if env_name == QueueSchedulingEnv.ENV_NAME:
        return [
            InterventionSpec(
                name="queue_load_spike",
                start_step=start,
                end_step=mid,
                params={"arrival_rate_scale": 1.0 + 0.9 * profile_scale},
            ),
            InterventionSpec(
                name="queue_action_corruption",
                start_step=mid,
                end_step=end,
                params={"action_flip_prob": 0.18 * profile_scale, "reward_scale": 1.0 - 0.08 * profile_scale},
            ),
        ]
    if env_name == InventoryControlEnv.ENV_NAME:
        return [
            InterventionSpec(
                name="inventory_demand_spike",
                start_step=start,
                end_step=mid,
                params={"demand_delta": 7.5 * profile_scale},
            ),
            InterventionSpec(
                name="inventory_sensor_noise",
                start_step=mid,
                end_step=end,
                params={"obs_noise_std": 0.08 * profile_scale, "obs_scale": 1.0 + 0.06 * profile_scale},
            ),
        ]
    if env_name == TrafficSignalEnv.ENV_NAME:
        return [
            InterventionSpec(
                name="traffic_wave_shift",
                start_step=start,
                end_step=mid,
                params={"queue_ns_delta": 6.0 * profile_scale, "queue_ew_delta": 2.5 * profile_scale},
            ),
            InterventionSpec(
                name="traffic_signal_latency",
                start_step=mid,
                end_step=end,
                params={"action_flip_prob": 0.15 * profile_scale, "obs_noise_std": 0.04 * profile_scale},
            ),
        ]
    if env_name == CartPoleEnv.ENV_NAME:
        return [
            InterventionSpec(
                name="cartpole_gravity_shift",
                start_step=start,
                end_step=mid,
                params={"gravity_scale": 1.0 + 0.35 * profile_scale, "force_mag_scale": 1.0 - 0.12 * profile_scale},
            ),
            InterventionSpec(
                name="cartpole_sensor_noise",
                start_step=mid,
                end_step=end,
                params={"obs_noise_std": 0.06 * profile_scale, "action_flip_prob": 0.08 * profile_scale},
            ),
        ]
    if env_name == LunarLanderEnv.ENV_NAME:
        return [
            InterventionSpec(
                name="lunarlander_engine_fault",
                start_step=start,
                end_step=mid,
                params={
                    "main_engine_failure_prob": min(0.95, 0.7 * profile_scale),
                    "lateral_engine_swap_prob": min(0.9, 0.35 * profile_scale),
                },
            ),
            InterventionSpec(
                name="lunarlander_navigation_noise",
                start_step=mid,
                end_step=end,
                params={"obs_noise_std": 0.05 * profile_scale, "reward_bias": -0.08 * profile_scale},
            )
        ]
    raise ValueError("Unknown environment: {0}".format(env_name))


def make_env(
    env_name: str,
    seed: Optional[int] = None,
    with_interventions: bool = False,
    horizon: Optional[int] = None,
    interventions: Optional[Sequence[InterventionSpec]] = None,
    intervention_profile: str = "standard",
) -> BaseDecisionEnv:
    if env_name == QueueSchedulingEnv.ENV_NAME:
        env = QueueSchedulingEnv(seed=seed)
    elif env_name == InventoryControlEnv.ENV_NAME:
        env = InventoryControlEnv(seed=seed)
    elif env_name == TrafficSignalEnv.ENV_NAME:
        env = TrafficSignalEnv(seed=seed)
    elif env_name == CartPoleEnv.ENV_NAME:
        env = CartPoleEnv(seed=seed)
    elif env_name == LunarLanderEnv.ENV_NAME:
        env = LunarLanderEnv(seed=seed)
    else:
        raise ValueError("Unknown environment: {0}".format(env_name))

    if with_interventions:
        specs = list(interventions) if interventions is not None else build_default_interventions(
            env_name=env_name,
            horizon=horizon or 600,
            profile=intervention_profile,
        )
        return InterventionEnv(base_env=env, interventions=specs)
    return env
