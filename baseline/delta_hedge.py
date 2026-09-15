"""Black-Scholes delta-hedge baseline agent (Kolm & Ritter 2019 benchmark)."""
import numpy as np

from pricing.black_scholes import call_price, call_delta


def run_delta_hedge(path, strike, sigma, r, cost_rate, dt=1.0 / 252.0):
    """Rebalance to the Black-Scholes call delta at each step.

    path: 1-D array of underlying prices (length n_steps+1).
    Returns dict with final pnl, total cost, and positions.
    """
    n_steps = len(path) - 1
    S = float(path[0])
    cash = call_price(S, strike, n_steps * dt, r, sigma)  # premium received
    position = 0.0
    total_cost = 0.0
    positions = np.zeros(n_steps + 1)

    for t in range(n_steps):
        ttm = (n_steps - t) * dt
        target = call_delta(S, strike, ttm, r, sigma)
        trade = target - position
        cost = cost_rate * abs(trade) * S
        position = target
        cash -= trade * S + cost
        total_cost += cost
        S = float(path[t + 1])
        positions[t + 1] = position

    # liquidate the final hedge position and settle the short call payoff
    cash += position * S
    payoff = max(S - strike, 0.0)
    pnl = cash - payoff
    return {
        "pnl": pnl,
        "total_cost": total_cost,
        "positions": positions,
    }


class DeltaHedgePolicy:
    """Policy wrapper usable inside HedgingEnv rollouts: returns the
    adjustment needed to reach BS delta."""

    def __init__(self, strike, sigma, r, n_steps, dt=1.0 / 252.0):
        self.strike, self.sigma, self.r = strike, sigma, r
        self.n_steps, self.dt = n_steps, dt

    def action(self, env):
        ttm = max((self.n_steps - env.t) * self.dt, 0.0)
        target = call_delta(env.S, self.strike, ttm, self.r, self.sigma)
        return np.array([target - env.position], dtype=np.float32)
