"""Train a PPO agent (Stable-Baselines3) on HedgingEnv.

Deep-hedging formulation per Buehler et al. (2019); PPO implementation via
stable_baselines3.PPO on vectorized envs (DummyVecEnv). Training curves are
logged to TensorBoard under runs/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.hedging_env import HedgingEnv


def make_env(cfg, rank):
    def _init():
        return HedgingEnv(
            days_to_expiry=cfg["days_to_expiry"],
            s0=cfg["s0"],
            strike=cfg["strike"],
            sigma=cfg["sigma"],
            mu=cfg["mu"],
            r=cfg["r"],
            cost_rate=cfg["cost_rate"],
            regime=cfg["regime"],
            garch_params=cfg.get("garch_params"),
            seed=cfg.get("seed", 0) + rank,
        )

    return _init


DEFAULT_CFG = {
    "days_to_expiry": 30,
    "s0": 100.0,
    "strike": 100.0,
    "sigma": 0.2,
    "mu": 0.0,
    "r": 0.0,
    "cost_rate": 0.001,
    "regime": "gbm",
    "seed": 7,
}


def train(cfg=None, total_timesteps=50_000, n_envs=4, model_dir="models",
          tb_log="runs", learning_rate=3e-4, progress_cb=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    os.makedirs(model_dir, exist_ok=True)
    env = DummyVecEnv([make_env(cfg, i) for i in range(n_envs)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=cfg["seed"],
        tensorboard_log=tb_log,
        n_steps=512,
        batch_size=128,
        learning_rate=learning_rate,
    )
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(model_dir, "ppo_hedging.zip")
    model.save(model_path)
    meta_path = os.path.join(model_dir, "ppo_hedging_config.json")
    with open(meta_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return model_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--regime", choices=["gbm", "garch"], default="gbm")
    ap.add_argument("--n-envs", type=int, default=4)
    args = ap.parse_args()
    path = train({"regime": args.regime}, total_timesteps=args.timesteps,
                 n_envs=args.n_envs)
    print(f"saved: {path}")
