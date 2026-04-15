from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class PricingEnvConfig:
    inventory: int = 50
    horizon: int = 30
    prices: Tuple[float, ...] = tuple(np.round(np.linspace(5.0, 20.0, 31), 2))

    # Traffic model
    base_arrival_rate: float = 12.0
    time_pattern: Tuple[float, ...] = (
        0.8, 0.9, 1.0, 1.1, 1.2,
        1.2, 1.1, 1.0, 0.9, 0.8,
    )

    # Conversion model
    conversion_ref_price: float = 10.0
    conversion_slope: float = 0.35

    seed: int = 42


class DynamicPricingEnv:
    """
    Semi-realistic dynamic pricing environment.

    State:
        [normalized_inventory, normalized_time_left]

    Action:
        index of chosen price from config.prices

    Reward:
        revenue = price * units_sold
    """

    def __init__(self, config: PricingEnvConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.inventory = config.inventory
        self.t = 0

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return len(self.config.prices)

    def reset(self) -> np.ndarray:
        self.inventory = self.config.inventory
        self.t = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        inventory_ratio = self.inventory / self.config.inventory
        time_left_ratio = (self.config.horizon - self.t) / self.config.horizon
        return np.array([inventory_ratio, time_left_ratio], dtype=np.float32)

    def _time_multiplier(self, t: int) -> float:
        pattern = self.config.time_pattern
        return pattern[t % len(pattern)]

    def _arrival_rate(self, t: int) -> float:
        return self.config.base_arrival_rate * self._time_multiplier(t)

    def _conversion_probability(self, price: float) -> float:
        x = -(price - self.config.conversion_ref_price) * self.config.conversion_slope
        prob = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(prob, 0.01, 0.99))

    def step(self, action: int):
        if not (0 <= action < self.action_dim):
            raise ValueError(f"Invalid action index: {action}")

        if self.t >= self.config.horizon or self.inventory <= 0:
            raise RuntimeError("Episode is done. Call reset() before step().")

        price = self.config.prices[action]

        arrival_rate = self._arrival_rate(self.t)
        arrivals = self.rng.poisson(arrival_rate)

        conversion_prob = self._conversion_probability(price)
        unconstrained_sales = self.rng.binomial(arrivals, conversion_prob)

        units_sold = min(self.inventory, unconstrained_sales)
        reward = float(price * units_sold)

        self.inventory -= units_sold
        self.t += 1

        done = (self.inventory == 0) or (self.t >= self.config.horizon)
        next_state = self._get_state()

        info: Dict = {
            "t": self.t,
            "price": price,
            "arrival_rate": float(arrival_rate),
            "arrivals": int(arrivals),
            "conversion_prob": float(conversion_prob),
            "unconstrained_sales": int(unconstrained_sales),
            "units_sold": int(units_sold),
            "inventory": int(self.inventory),
            "revenue": float(reward),
        }

        return next_state, reward, done, info