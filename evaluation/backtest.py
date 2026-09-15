"""Backtest: PPO agent vs Black-Scholes delta-hedge baseline.

Runs both agents on identical seeded price paths in each regime (GBM,
GARCH), computes per-path terminal P&L, transaction costs, and a paired
significance test on |P&L| (t-test; Wilcoxon when the t-test is unreliable).
Writes results/backtest_results.json and results/backtest_table.md.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from stable_baselines3 import PPO

from baseline.delta_hedge import DeltaHedgePolicy, run_delta_hedge
from env.hedging_env import HedgingEnv
from sim.price_paths import simulate_garch, simulate_gbm

DEFAULT_GARCH = {"mu": 0.0, "omega": 2e-6, "alpha": 0.08, "beta": 0.90}


def gen_paths(regime, n_paths, n_steps, s0, sigma, dt, seed):
    if regime == "garch":
        return simulate_garch(n_paths, n_steps, s0, DEFAULT_GARCH, dt, seed=seed)
    return simulate_gbm(n_paths, n_steps, s0, mu=0.0, sigma=sigma, dt=dt, seed=seed)


def rollout_rl(model, env_cfg, path):
    env = HedgingEnv(**env_cfg)
    obs, _ = env.reset(options={"path": path})
    done = False
    total_cost = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        total_cost += info["cost"]
        done = terminated or truncated
    return env.portfolio_value(), total_cost


def run_backtest(model_path, n_paths=200, seed=123, out_dir="results",
                 env_overrides=None):
    model = PPO.load(model_path)
    cfg_path = os.path.join(os.path.dirname(model_path), "ppo_hedging_config.json")
    env_cfg = {
        "days_to_expiry": 30,
        "s0": 100.0,
        "strike": 100.0,
        "sigma": 0.2,
        "mu": 0.0,
        "r": 0.0,
        "cost_rate": 0.001,
    }
    if os.path.exists(cfg_path):
        env_cfg.update(json.load(open(cfg_path)))
    env_cfg.update(env_overrides or {})
    env_cfg.pop("regime", None)
    env_cfg.pop("seed", None)
    env_cfg.pop("garch_params", None)

    dt = 1.0 / 252.0
    results = {}
    for regime in ("gbm", "garch"):
        paths = gen_paths(regime, n_paths, env_cfg["days_to_expiry"],
                          env_cfg["s0"], env_cfg["sigma"], dt, seed)
        rl_pnl, rl_cost, bs_pnl, bs_cost = [], [], [], []
        policy = DeltaHedgePolicy(
            env_cfg["strike"], env_cfg["sigma"], env_cfg["r"],
            env_cfg["days_to_expiry"], dt,
        )
        for path in paths:
            pnl, cost = rollout_rl(model, env_cfg, path)
            rl_pnl.append(pnl)
            rl_cost.append(cost)
            res = run_delta_hedge(
                path, env_cfg["strike"], env_cfg["sigma"], env_cfg["r"],
                env_cfg["cost_rate"], dt,
            )
            bs_pnl.append(res["pnl"])
            bs_cost.append(res["total_cost"])
        rl_pnl, bs_pnl = np.array(rl_pnl), np.array(bs_pnl)
        diff = rl_pnl - bs_pnl
        t_p = float(stats.ttest_rel(rl_pnl, bs_pnl).pvalue)
        try:
            w_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            w_p = float("nan")
        results[regime] = {
            "n_paths": int(n_paths),
            "rl": {
                "pnl_mean": float(rl_pnl.mean()),
                "pnl_variance": float(rl_pnl.var()),
                "pnl_std": float(rl_pnl.std()),
                "total_cost": float(np.sum(rl_cost)),
                "pnl_samples": rl_pnl.tolist(),
            },
            "baseline": {
                "pnl_mean": float(bs_pnl.mean()),
                "pnl_variance": float(bs_pnl.var()),
                "pnl_std": float(bs_pnl.std()),
                "total_cost": float(np.sum(bs_cost)),
                "pnl_samples": bs_pnl.tolist(),
            },
            "t_test_pvalue": t_p,
            "wilcoxon_pvalue": w_p,
        }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = ["| Regime | Agent | P&L mean | P&L var | Total cost | p-value |",
             "|--------|-------|----------|---------|------------|---------|"]
    for regime, r in results.items():
        for agent in ("rl", "baseline"):
            a = r[agent]
            lines.append(
                f"| {regime.upper()} | {agent} | {a['pnl_mean']:.4f} | "
                f"{a['pnl_variance']:.4f} | {a['total_cost']:.2f} | "
                f"{r['wilcoxon_pvalue']:.4g} |"
            )
    with open(os.path.join(out_dir, "backtest_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ppo_hedging.zip")
    ap.add_argument("--n-paths", type=int, default=200)
    args = ap.parse_args()
    res = run_backtest(args.model, n_paths=args.n_paths)
    for regime, r in res.items():
        print(regime, json.dumps({k: v for k, v in r.items() if k != "rl"}, indent=1)[:200])
        print("  rl:", {k: round(v, 4) for k, v in r["rl"].items() if k != "pnl_samples"})
        print("  bs:", {k: round(v, 4) for k, v in r["baseline"].items() if k != "pnl_samples"})
