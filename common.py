"""Shared environment, policy and evaluation helpers."""
from __future__ import annotations
import numpy as np
import gymnasium as gym

ENV_ID = "HalfCheetah-v4"
OBS_DIM = 17
ACT_DIM = 6
BASE_GRAVITY = -9.81


def make_env(gravity_scale: float = 1.0) -> gym.Env:
    """HalfCheetah with the vertical gravity component scaled."""
    env = gym.make(ENV_ID)
    env.unwrapped.model.opt.gravity[2] = BASE_GRAVITY * gravity_scale
    return env


def linear_policy(w: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Deterministic linear controller, clipped to the action range."""
    return np.clip(w @ obs, -1.0, 1.0)


def rollout(w: np.ndarray, env: gym.Env, seed: int | None = None) -> tuple[float, int]:
    """One episode. Returns (undiscounted return, environment steps consumed)."""
    obs, _ = env.reset(seed=seed)
    total, steps, done, trunc = 0.0, 0, False, False
    while not (done or trunc):
        obs, reward, done, trunc, _ = env.step(linear_policy(w, obs))
        total += reward
        steps += 1
    return total, steps


def evaluate_linear(w: np.ndarray, gravity_scale: float, episodes: int = 10,
                    seed0: int = 10_000) -> tuple[float, float, int]:
    """Mean and SEM return of a linear policy. Fixed eval seeds for reproducibility."""
    env = make_env(gravity_scale)
    returns, steps = [], 0
    for k in range(episodes):
        r, s = rollout(w, env, seed=seed0 + k)
        returns.append(r)
        steps += s
    env.close()
    a = np.asarray(returns)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))), steps
