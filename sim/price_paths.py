"""Underlying price-path simulators: GBM and GARCH(1,1)."""
import numpy as np


def simulate_gbm(n_paths, n_steps, s0, mu, sigma, dt, seed=None):
    """Geometric Brownian motion. Returns (n_paths, n_steps+1) array."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_ret, axis=1))
    return paths


def simulate_garch(n_paths, n_steps, s0, garch_params, dt, seed=None):
    """GARCH(1,1) simulated with the `arch` package.

    garch_params: dict with keys
      - mu: per-step mean return (decimal, e.g. 0.0003)
      - omega, alpha, beta: GARCH(1,1) variance params in per-step units
        (alpha + beta < 1 required for stationarity)
    Returns (n_paths, n_steps+1) array of prices.
    """
    from arch.univariate import arch_model

    params = np.array([
        garch_params.get("mu", 0.0),
        garch_params["omega"],
        garch_params["alpha"],
        garch_params["beta"],
    ])
    # arch 8.x simulate() draws from numpy's global RNG
    rng = np.random.default_rng(seed)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    am = arch_model(None, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
    rets = np.empty((n_paths, n_steps))
    for i in range(n_paths):
        sim = am.simulate(params, nobs=n_steps, burn=250)
        rets[i] = np.asarray(sim.data).ravel()[-n_steps:]
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(np.log1p(rets), axis=1))
    return paths


def fit_garch(returns, dt=1.0 / 252.0):
    """Fit a GARCH(1,1) to an array of per-step returns; returns param dict
    suitable for `simulate_garch` (converted to the same return units)."""
    from arch.univariate import arch_model

    am = arch_model(returns * 100.0, mean="Constant", vol="GARCH", p=1, q=1)
    fit = am.fit(disp="off")
    p = fit.params
    return {
        "mu": p["mu"] / 100.0,
        "omega": p["omega"] / 1e4,
        "alpha": p["alpha[1]"],
        "beta": p["beta[1]"],
    }
