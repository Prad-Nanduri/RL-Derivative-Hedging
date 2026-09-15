import numpy as np
import pytest

from env.hedging_env import HedgingEnv


def make_env(**kw):
    defaults = dict(days_to_expiry=30, s0=100.0, strike=100.0, sigma=0.2,
                    cost_rate=0.001, regime="gbm", seed=0)
    defaults.update(kw)
    return HedgingEnv(**defaults)


def test_reward_no_trade_flat_path():
    # Hand-computed: flat path at 100, no trade -> reward = -pnl_step^2 where
    # pnl_step = -(option_value_t1 - option_value_t0) (position 0)
    env = make_env()
    path = np.full(31, 100.0)
    env.reset(options={"path": path})
    from pricing.black_scholes import call_price
    v0 = call_price(100, 100, 30 / 252, 0.0, 0.2)
    v1 = call_price(100, 100, 29 / 252, 0.0, 0.2)
    _, reward, *_ = env.step(np.array([0.0], dtype=np.float32))
    assert reward == pytest.approx(-((v0 - v1) ** 2), abs=1e-6)


def test_transaction_cost_subtracted():
    env = make_env(cost_rate=0.01)
    path = np.full(31, 100.0)
    env.reset(options={"path": path})
    from pricing.black_scholes import call_price
    v0 = call_price(100, 100, 30 / 252, 0.0, 0.2)
    v1 = call_price(100, 100, 29 / 252, 0.0, 0.2)
    _, reward, _, _, info = env.step(np.array([0.1], dtype=np.float32))
    expected_cost = 0.01 * 0.1 * 100.0
    assert info["cost"] == pytest.approx(expected_cost)
    assert reward == pytest.approx(-((v0 - v1) ** 2) - expected_cost, abs=1e-6)


def test_position_clipped_and_obs_bounds():
    env = make_env()
    env.reset(options={"path": np.linspace(100, 110, 31)})
    obs, *_ = env.step(np.array([0.5], dtype=np.float32))
    assert env.position == pytest.approx(0.5)
    obs, *_ = env.step(np.array([0.5], dtype=np.float32))
    assert env.position == pytest.approx(1.0)
    assert env.observation_space.contains(obs)


def test_episode_terminates_and_settles():
    env = make_env(days_to_expiry=5)
    path = np.linspace(100, 120, 6)
    env.reset(options={"path": path})
    done = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        done = terminated or truncated
    assert done
    assert env.prev_option_value == pytest.approx(20.0)  # S-K at expiry
    pv = env.portfolio_value()
    # no trades, short call: pv = premium - payoff
    assert pv == pytest.approx(
        env.cash - 20.0, abs=1e-6
    )


def test_deterministic_with_seed():
    a, b = make_env(seed=42), make_env(seed=42)
    seq_a = [a.step(np.array([0.1], dtype=np.float32))[1] for _ in range(3)]
    a.reset(seed=42)
    seq_b = [b.step(np.array([0.1], dtype=np.float32))[1] for _ in range(3)]
    # reseed both for fairness: sample fresh sequences
    a.reset(seed=7); b.reset(seed=7)
    seq_a = [a.step(np.array([0.1], dtype=np.float32))[1] for _ in range(3)]
    seq_b = [b.step(np.array([0.1], dtype=np.float32))[1] for _ in range(3)]
    assert seq_a == seq_b
