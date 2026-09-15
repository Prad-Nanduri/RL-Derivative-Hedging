"""Gymnasium environment: delta-hedging a short European call option.

Reward formulation follows Buehler, Gonon, Teichmann & Wood (2019), "Deep
Hedging", Quantitative Finance 19(8): reward each step is the negative
incremental P&L variance contribution minus proportional transaction
costs. See also Kolm & Ritter (2019), "Dynamic Replication and Hedging:
A Reinforcement Learning Approach", Journal of Financial Data Science.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pricing.black_scholes import call_price


class HedgingEnv(gym.Env):
    """Short one European call at t=0 (premium received at Black-Scholes
    price); each step the agent adjusts its hedge position in the underlying.

    Observation: [S/S0, tau, hedge position, realized vol estimate].
    Action: continuous adjustment to hedge position in [-max_adj, max_adj];
    position is clipped to [0, 1] (one share of underlying per option).
    Reward: -(step P&L)^2 - cost_rate * |trade notional|.
    Episode: one option lifetime; terminates at expiry with payoff settled.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        days_to_expiry=30,
        s0=100.0,
        strike=100.0,
        sigma=0.2,
        mu=0.0,
        r=0.0,
        cost_rate=0.001,
        max_adj=0.5,
        regime="gbm",
        garch_params=None,
        seed=None,
    ):
        super().__init__()
        self.n_steps = days_to_expiry
        self.dt = 1.0 / 252.0
        self.s0 = s0
        self.strike = strike
        self.sigma = sigma
        self.mu = mu
        self.r = r
        self.cost_rate = cost_rate
        self.max_adj = max_adj
        self.regime = regime
        self.garch_params = garch_params or {
            "mu": mu * self.dt,
            "omega": 1e-6,
            "alpha": 0.08,
            "beta": 0.90,
        }
        self._rng = np.random.default_rng(seed)
        # GARCH state (per-step variance units)
        self._garch_var = self.garch_params["omega"] / (
            1.0 - self.garch_params["alpha"] - self.garch_params["beta"]
        )
        self._eps2 = self._garch_var

        high = np.array([np.inf, 1.0, 1.0, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-max_adj, high=max_adj, shape=(1,), dtype=np.float32
        )

        self.reset(seed=seed)

    def _vol_estimate(self):
        if len(self._ret_hist) < 2:
            return self.sigma
        return float(np.std(self._ret_hist) / np.sqrt(self.dt))

    def _obs(self):
        tau = (self.n_steps - self.t) * self.dt * 252.0 / self.n_steps
        # express tau as fraction of original life for a bounded obs
        tau_frac = (self.n_steps - self.t) / self.n_steps
        return np.array(
            [self.S / self.s0, tau_frac, self.position, self._vol_estimate()],
            dtype=np.float32,
        )

    def _step_price(self):
        if self._path is not None:
            prev = self.S
            self.S = float(self._path[self.t + 1])
            self._ret_hist.append(np.log(self.S / prev))
            return
        if self.regime == "garch":
            p = self.garch_params
            z = self._rng.standard_normal()
            eps = np.sqrt(self._garch_var) * z
            ret = p["mu"] + eps
            self._garch_var = (
                p["omega"] + p["alpha"] * self._eps2 + p["beta"] * self._garch_var
            )
            self._eps2 = eps**2
        else:
            ret = (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(
                self.dt
            ) * self._rng.standard_normal()
        self.S *= float(np.exp(ret))
        self._ret_hist.append(ret)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._path = None
        if options and "path" in options:
            self._path = np.asarray(options["path"], dtype=np.float64)
        p = self.garch_params
        self._garch_var = p["omega"] / max(1.0 - p["alpha"] - p["beta"], 1e-6)
        self._eps2 = self._garch_var
        self.S = float(self._path[0]) if self._path is not None else self.s0
        self.t = 0
        self.position = 0.0
        self._ret_hist = []
        # short the call; receive Black-Scholes premium
        self.option_value = call_price(self.S, self.strike, self.n_steps * self.dt, self.r, self.sigma)
        self.cash = self.option_value
        self.prev_option_value = self.option_value
        return self._obs(), {}

    def step(self, action):
        adj = float(np.clip(action[0], -self.max_adj, self.max_adj))
        trade = adj  # shares bought (positive) or sold (negative)
        cost = self.cost_rate * abs(trade) * self.S
        # hold position over the step, earning/paying the underlying's move
        prev_S = self.S
        prev_position = self.position
        self._step_price()
        self.t += 1
        self.position = float(np.clip(prev_position + trade, 0.0, 1.0))
        self.cash -= trade * prev_S + cost

        ttm = max((self.n_steps - self.t) * self.dt, 0.0)
        option_value = (
            call_price(self.S, self.strike, ttm, self.r, self.sigma)
            if ttm > 0
            else max(self.S - self.strike, 0.0)
        )
        pnl_step = prev_position * (self.S - prev_S) - (option_value - self.prev_option_value)
        self.prev_option_value = option_value

        reward = -(pnl_step**2) - cost
        terminated = self.t >= self.n_steps
        if terminated:
            # settle: short call payoff is -(S-K)+; position shares & cash mark out
            self.option_value = option_value
        return self._obs(), reward, terminated, False, {"pnl_step": pnl_step, "cost": cost}

    def portfolio_value(self):
        """Mark-to-market: cash + shares - call liability."""
        return self.cash + self.position * self.S - self.prev_option_value
