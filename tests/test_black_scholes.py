import math

import pytest

from pricing.black_scholes import call_delta, call_price, put_price


def test_put_call_parity():
    s, k, t, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.25
    c = call_price(s, k, t, r, sigma)
    p = put_price(s, k, t, r, sigma)
    assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-10)


def test_call_price_known_value():
    # textbook value: S=K=100, T=1, r=0.05, sigma=0.2 -> ~10.4506
    assert call_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-3)


def test_delta_bounds():
    assert call_delta(200, 100, 1.0, 0.0, 0.2) == pytest.approx(1.0, abs=1e-3)
    assert call_delta(1, 100, 1.0, 0.0, 0.2) == pytest.approx(0.0, abs=1e-6)
    assert call_delta(100, 100, 1.0, 0.0, 0.2) == pytest.approx(0.5398, abs=1e-3)


def test_expiry_payoff():
    assert call_price(110, 100, 0.0, 0.0, 0.2) == 10.0
    assert put_price(110, 100, 0.0, 0.0, 0.2) == 0.0
