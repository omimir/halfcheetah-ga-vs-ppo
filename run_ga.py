#!/usr/bin/env python3
"""
Microbial genetic algorithm on HalfCheetah-v4.

A population of linear policies (6x17 weight matrices) is evolved by repeated
tournaments: two individuals are drawn at random, both are evaluated for one
episode, and the LOSER is overwritten by a recombination of the pair plus
Gaussian mutation. The winner is left untouched, which is what makes this a
microbial GA rather than a generational one.

Every tournament costs two episodes, so the environment-step budget is spent at
a known rate and can be matched exactly against PPO.

Usage:
    python run_ga.py --sigma 0.05 --seed 0 --budget 400000
"""
from __future__ import annotations
import argparse, csv, json, pathlib, time
import numpy as np

from common import ACT_DIM, OBS_DIM, make_env, rollout, evaluate_linear

POP_SIZE = 20
RECOMBINATION_RATE = 0.5   # fraction of the winner's genes copied into the loser
INIT_SCALE = 0.1


def microbial_ga(sigma: float, seed: int, budget: int, log_path: pathlib.Path):
    """Run tournaments until the environment-step budget is exhausted.

    Returns (champion_weights, champion_fitness, history, steps_used).
    """
    rng = np.random.default_rng(seed)
    env = make_env(1.0)
    pop = rng.normal(0, INIT_SCALE, (POP_SIZE, ACT_DIM, OBS_DIM))

    champion = pop[0].copy()
    champion_fitness = -np.inf
    history = []          # (env_steps, tournament, best_this_tournament, best_so_far)
    steps_used = 0
    tournament = 0

    while steps_used < budget:
        i, j = rng.choice(POP_SIZE, size=2, replace=False)
        fi, si = rollout(pop[i], env)
        fj, sj = rollout(pop[j], env)
        steps_used += si + sj
        tournament += 1

        # --- selection: winner infects loser -------------------------------
        if fi >= fj:
            winner_idx, loser_idx, winner_fitness = i, j, fi
        else:
            winner_idx, loser_idx, winner_fitness = j, i, fj

        winner = pop[winner_idx]
        loser = pop[loser_idx]
        child = loser + RECOMBINATION_RATE * (winner - loser)
        pop[loser_idx] = child + rng.normal(0, sigma, child.shape)
        # -------------------------------------------------------------------

        if winner_fitness > champion_fitness:
            champion_fitness = float(winner_fitness)
            champion = winner.copy()

        history.append((steps_used, tournament, float(max(fi, fj)), champion_fitness))

        if tournament % 25 == 0:
            print(f"  sigma={sigma} seed={seed} tournament {tournament:4d} "
                  f"steps {steps_used:7d} best-so-far {champion_fitness:8.1f}", flush=True)

    env.close()
    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", "tournament", "best_this_tournament", "best_so_far"])
        w.writerows(history)
    return champion, champion_fitness, history, steps_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--logdir", type=pathlib.Path, default=pathlib.Path("logs"))
    args = ap.parse_args()
    args.logdir.mkdir(exist_ok=True)

    tag = f"ga_sigma{args.sigma}_seed{args.seed}"
    t0 = time.time()
    champion, champion_fitness, _, steps = microbial_ga(
        args.sigma, args.seed, args.budget, args.logdir / f"{tag}_history.csv")

    # Persist the champion so evaluation loads the SAME weights that were trained.
    weights_path = args.logdir / f"{tag}_champion.npz"
    np.savez_compressed(weights_path, weights=champion,
                        fitness=champion_fitness, sigma=args.sigma, seed=args.seed)

    reloaded = np.load(weights_path)["weights"]
    assert np.array_equal(reloaded, champion), "champion weights did not round-trip"

    normal_mean, normal_sem, _ = evaluate_linear(reloaded, 1.0, args.eval_episodes)
    reduced_mean, reduced_sem, _ = evaluate_linear(reloaded, 0.8, args.eval_episodes)

    result = {
        "method": "GA", "sigma": args.sigma, "seed": args.seed,
        "train_steps": steps, "train_best_fitness": champion_fitness,
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
