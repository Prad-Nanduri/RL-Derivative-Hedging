import numpy as np
import pytest
from scipy import stats

from sim.price_paths import simulate_garch, simulate_gbm


def test_gbm_shape_and_positive():
    p = simulate_gbm(50, 30, 100.0, 0.05, 0.2, 1 / 252, seed=1)
    assert p.shape == (50, 31)
    assert (p > 0).all()
    assert (p[:, 0] == 100.0).all()


def test_gbm_terminal_moments():
    n, steps, s0, mu, sigma, dt = 4000, 30, 100.0, 0.05, 0.2, 1 / 252
    p = simulate_gbm(n, steps, s0, mu, sigma, dt, seed=2)
    log_ret = np.log(p[:, -1] / s0)
    exp_mean = (mu - 0.5 * sigma**2) * steps * dt
    exp_std = sigma * np.sqrt(steps * dt)
    assert log_ret.mean() == pytest.approx(exp_mean, abs=3 * exp_std / np.sqrt(n))
    assert log_ret.std() == pytest.approx(exp_std, rel=0.1)


def test_gbm_lognormality():
    p = simulate_gbm(3000, 30, 100.0, 0.0, 0.25, 1 / 252, seed=3)
    log_ret = np.log(p[:, -1] / 100.0)
    _, pval = stats.normaltest(log_ret)
    assert pval > 0.01


def test_garch_shape_and_positive():
    params = {"mu": 0.0003, "omega": 2e-6, "alpha": 0.08, "beta": 0.90}
    p = simulate_garch(40, 30, 100.0, params, 1 / 252, seed=4)
    assert p.shape == (40, 31)
    assert (p > 0).all()
