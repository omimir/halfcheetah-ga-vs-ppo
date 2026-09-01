#!/usr/bin/env python3
"""Figures and statistics for the GA vs PPO comparison."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

LOGS = pathlib.Path("logs")
FIGS = pathlib.Path("figures")
FIGS.mkdir(exist_ok=True)

# Categorical slots 1-3 of the validated palette (light surface #fcfcfb).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
SERIES = {"PPO": "#2a78d6", "GA sigma=0.05": "#eb6834", "GA sigma=0.30": "#1baf7a"}
SEEDS = (0, 1, 2)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 12, "legend.frameon": False,
})


def style(ax):
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_results() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(LOGS.glob("*_result.json"))]
    if not rows:
        sys.exit("no result files found - run run_all.sh first")
    df = pd.DataFrame(rows)
    df["label"] = np.where(df.method == "PPO", "PPO",
                           "GA sigma=" + df.sigma.map(lambda s: f"{s:.2f}" if s == s else ""))
    return df


def curve(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and SEM of return against environment steps, interpolated onto a common grid."""
    grid = np.linspace(0, 400_000, 200)
    series = []
    for seed in SEEDS:
        if label == "PPO":
            f = LOGS / f"ppo_seed{seed}_history.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            # best training episode observed so far - the same quantity the GA
            # curve reports, so the two lines are like-for-like
            y = d.episode_return.cummax().to_numpy()
            x = d.env_steps.to_numpy()
        else:
            sigma = label.split("=")[1]
            f = LOGS / f"ga_sigma{float(sigma)}_seed{seed}_history.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            y = d.best_so_far.to_numpy()
            x = d.env_steps.to_numpy()
        series.append(np.interp(grid, x, y))
    if not series:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    a = np.vstack(series)
    return grid, a.mean(0), a.std(0, ddof=1) / np.sqrt(a.shape[0])


def fig_learning_curves():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for label, colour in SERIES.items():
        x, m, se = curve(label)
        if np.isnan(m).all():
            continue
        ax.plot(x, m, color=colour, linewidth=2, label=label, solid_capstyle="round")
        ax.fill_between(x, m - se, m + se, color=colour, alpha=0.15, linewidth=0)
        ax.annotate(label, (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK_2, va="center", fontsize=9)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Best training episode return so far")
    ax.set_title("Learning under a matched 400k-step budget", loc="left", pad=12)
    ax.set_xlim(0, 400_000)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v/1000:.0f}k"))
    ax.margins(x=0.16)
    ax.legend(loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "learning_curves.png", dpi=160)
    plt.close(fig)


def fig_gravity(df: pd.DataFrame):
    order = ["PPO", "GA sigma=0.05", "GA sigma=0.30"]
    g = df.groupby("label").agg(
        normal=("normal_gravity_mean", "mean"), normal_sd=("normal_gravity_mean", "std"),
        reduced=("reduced_gravity_mean", "mean"), reduced_sd=("reduced_gravity_mean", "std"),
    ).reindex(order).dropna(how="all")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(g))
    w = 0.36
    for k, (col, sd, alpha, cond) in enumerate([("normal", "normal_sd", 1.0, "Standard gravity"),
                                                ("reduced", "reduced_sd", 0.45, "0.8 g")]):
        colours = [SERIES[i] for i in g.index]
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), g[col], w, color=colours, alpha=alpha,
                      edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x + (k - 0.5) * (w + 0.02), g[col], yerr=g[sd], fmt="none",
                    ecolor=INK_2, elinewidth=1.2, capsize=4)
        for b, v, e in zip(bars, g[col], g[sd].fillna(0)):
            top = v + (e if v >= 0 else -e)
            ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, top),
                        xytext=(0, 9 if v >= 0 else -18), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(x, g.index)
    ax.set_ylabel("Mean return over 10 episodes")
    ax.set_title("Final policy under standard and reduced gravity", loc="left", pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=INK_2, label="Standard gravity"),
                       Patch(facecolor=INK_2, alpha=0.45, label="0.8 g")],
              loc="upper right")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "reduced_gravity.png", dpi=160)
    plt.close(fig)


def statistics(df: pd.DataFrame) -> str:
    out = []
    ppo = df[df.label == "PPO"]
    for label in ["GA sigma=0.05", "GA sigma=0.30"]:
        ga = df[df.label == label]
        if ga.empty:
            continue
        for cond, col in [("standard gravity", "normal_gravity_mean"),
                          ("0.8 g", "reduced_gravity_mean")]:
            t, p = stats.ttest_ind(ppo[col], ga[col], equal_var=False)
            pooled = np.sqrt((ppo[col].var(ddof=1) + ga[col].var(ddof=1)) / 2)
            d = (ppo[col].mean() - ga[col].mean()) / pooled if pooled else np.nan
            out.append(f"| PPO vs {label} | {cond} | {ppo[col].mean():,.0f} | "
                       f"{ga[col].mean():,.0f} | {t:.2f} | {p:.4f} | {d:.2f} |")
    return "\n".join(out)


def main():
    df = load_results()
    df.to_csv(LOGS / "results_summary.csv", index=False)
    fig_learning_curves()
    fig_gravity(df)

    per = df.groupby("label").agg(
        n=("seed", "count"),
        standard=("normal_gravity_mean", "mean"),
        reduced=("reduced_gravity_mean", "mean"),
    )
    per["change"] = per.reduced - per.standard
    print(per.round(1).to_string())
    print()
    print("| comparison | condition | PPO | GA | t | p | Cohen d |")
    print("|---|---|---|---|---|---|---|")
    print(statistics(df))


if __name__ == "__main__":
    main()
