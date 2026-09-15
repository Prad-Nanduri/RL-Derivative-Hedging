"""Hand-rolled Black-Scholes pricing and Greeks (no QuantLib)."""
import math
from statistics import NormalDist

_NORM = NormalDist()


def _d1(s, k, t, r, sigma):
    if t <= 0 or sigma <= 0:
        return 0.0
    return (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))


def call_price(s, k, t, r, sigma):
    if t <= 0:
        return max(s - k, 0.0)
    d1 = _d1(s, k, t, r, sigma)
    d2 = d1 - sigma * math.sqrt(t)
    return s * _NORM.cdf(d1) - k * math.exp(-r * t) * _NORM.cdf(d2)


def put_price(s, k, t, r, sigma):
    if t <= 0:
        return max(k - s, 0.0)
    d1 = _d1(s, k, t, r, sigma)
    d2 = d1 - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * _NORM.cdf(-d2) - s * _NORM.cdf(-d1)


def call_delta(s, k, t, r, sigma):
    if t <= 0:
        return 1.0 if s > k else 0.0
    return _NORM.cdf(_d1(s, k, t, r, sigma))
