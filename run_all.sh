#!/usr/bin/env bash
# Full experiment: matched 400k environment-step budget for both methods.
set -u
cd "$(dirname "$0")"
B=400000
echo "=== phase 1: PPO seeds 0,1 ==="
python3 run_ppo.py --seed 0 --budget $B > logs/ppo0.out 2>&1 &
python3 run_ppo.py --seed 1 --budget $B > logs/ppo1.out 2>&1 &
wait
echo "=== phase 2: PPO seed 2 + all GA runs ==="
python3 run_ppo.py --seed 2 --budget $B > logs/ppo2.out 2>&1 &
(
  for s in 0 1 2; do
    python3 run_ga.py --sigma 0.05 --seed $s --budget $B > logs/ga005_$s.out 2>&1
  done
  for s in 0 1 2; do
    python3 run_ga.py --sigma 0.30 --seed $s --budget $B > logs/ga030_$s.out 2>&1
  done
) &
wait
echo "=== DONE ==="
date -u +%FT%TZ
