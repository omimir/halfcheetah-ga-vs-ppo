#!/usr/bin/env python3
"""
PPO baseline on HalfCheetah-v4, trained to the same environment-step budget as
the GA and evaluated with the same protocol.

Episode returns are logged against environment steps during training (via the
Monitor wrapper), so the learning curve is measured rather than assumed.

Usage:
    python run_ppo.py --seed 0 --budget 400000
"""
from __future__ import annotations
import argparse, csv, json, pathlib, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from common import make_env

torch.set_num_threads(1)


class EpisodeReturnLogger(BaseCallback):
    """Record (env_steps, episode_return) for every finished training episode."""

    def __init__(self):
        super().__init__()
        self.rows: list[tuple[int, float]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.rows.append((self.num_timesteps, float(ep["r"])))
        return True


def evaluate_model(model, gravity_scale: float, episodes: int = 10,
                   seed0: int = 10_000) -> tuple[float, float]:
    """Same eval protocol and seeds as the GA: deterministic policy, fixed seeds."""
    env = make_env(gravity_scale)
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed0 + k)
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            total += reward
        returns.append(total)
    env.close()
    a = np.asarray(returns)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--logdir", type=pathlib.Path, default=pathlib.Path("logs"))
    args = ap.parse_args()
    args.logdir.mkdir(exist_ok=True)

    tag = f"ppo_seed{args.seed}"
    t0 = time.time()

    env = Monitor(make_env(1.0))
    model = PPO("MlpPolicy", env, seed=args.seed, n_steps=2048, batch_size=64,
                learning_rate=3e-4, verbose=0, device="cpu")
    logger = EpisodeReturnLogger()
    model.learn(total_timesteps=args.budget, callback=logger)

    model_path = args.logdir / f"{tag}.zip"
    model.save(model_path)
    env.close()

    with (args.logdir / f"{tag}_history.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", "episode_return"])
        w.writerows(logger.rows)

    reloaded = PPO.load(model_path, device="cpu")
    normal_mean, normal_sem = evaluate_model(reloaded, 1.0, args.eval_episodes)
    reduced_mean, reduced_sem = evaluate_model(reloaded, 0.8, args.eval_episodes)

    result = {
        "method": "PPO", "sigma": None, "seed": args.seed,
        "train_steps": args.budget,
        "train_best_fitness": max((r for _, r in logger.rows), default=float("nan")),
        "eval_episodes": args.eval_episodes,
        "normal_gravity_mean": normal_mean, "normal_gravity_sem": normal_sem,
        "reduced_gravity_mean": reduced_mean, "reduced_gravity_sem": reduced_sem,
        "retention": reduced_mean / normal_mean if normal_mean > 0 else float("nan"),
        "wallclock_s": round(time.time() - t0, 1),
    }
    (args.logdir / f"{tag}_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
