"""Figure for Experiment E (baseline pressure, 12 passengers / 4 drivers).

(a) Mean nominal welfare against mean platform budget. Every posted-price
    menu in the grid is a grey dot; the welfare-best menu is highlighted.
    ORA and the threshold rule share the same allocation and differ only in
    the budget. Error bars are 95% CIs over the 50 paired seeds.
(b) Share of agents allocated under truthful reporting who have a profitable
    unilateral misreport in the tested grid, by arm and side of the market.

Inputs:  results/budget_balanced_raw.csv, results/budget_balanced_pp_grid.csv,
         results/budget_balanced_manip_summary.csv
Output:  figures/budget_balanced_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import apply_ieee_style

N_PAX = 12
OUT = Path("figures") / "budget_balanced_comparison.png"

COLORS = {"ORA": "#1f77b4", "TB": "#e87ba4", "PP": "#eda100"}
LABELS = {"ORA": "ORA (Clarke)", "TB": "Threshold rule",
          "PP": "Posted price\n(best menu)"}
INK = "#222222"
MUTED = "#9a9a96"


def ci95(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    apply_ieee_style()
    raw = pd.read_csv("results/budget_balanced_raw.csv")
    grid = pd.read_csv("results/budget_balanced_pp_grid.csv")
    manip = pd.read_csv("results/budget_balanced_manip_summary.csv")
    raw = raw[raw.n_passengers == N_PAX]
    grid = grid[grid.n_passengers == N_PAX]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 3.0),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- (a) welfare vs budget ------------------------------------------
    menus = (grid.groupby(["fare_solo", "base_pay"])
             [["realised_welfare", "budget_balance"]].mean())
    ax1.scatter(menus.budget_balance, menus.realised_welfare, s=10,
                color=MUTED, alpha=0.55, lw=0, zorder=1,
                label="Other posted-price menus")
    ax1.axvline(0, color=INK, lw=0.6, ls=(0, (3, 3)), zorder=0)

    offsets = {"ORA": (6, -13), "TB": (-6, 7), "PP": (7, 6)}
    for arm in ("ORA", "TB", "PP"):
        part = raw[raw.arm == arm]
        x, y = part.budget_balance.mean(), part.realised_welfare.mean()
        ax1.errorbar(x, y, xerr=ci95(part.budget_balance),
                     yerr=ci95(part.realised_welfare), fmt="o", ms=6,
                     color=COLORS[arm], mec="white", mew=1.0, elinewidth=1.0,
                     capsize=2, zorder=3)
        ax1.annotate(LABELS[arm], (x, y), xytext=offsets[arm],
                     textcoords="offset points", fontsize=7.5, color=INK,
                     ha="right" if arm == "TB" else "left")
    ax1.set_xlabel("Platform budget $B$ (surplus > 0)")
    ax1.set_ylabel("Nominal welfare $W^{nom}$")
    ax1.set_title("(a) Welfare and platform budget", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.set_ylim(0, 108)
    ax1.legend(loc="center left", fontsize=7, frameon=False)

    # ---- (b) manipulation -----------------------------------------------
    share = manip[(manip.n_passengers == N_PAX)
                  & (manip.metric == "share_profitable")]
    arms = ["ORA", "TB", "PP"]
    kinds = [("passenger", "Passengers", None), ("driver", "Drivers", "////")]
    width = 0.36
    xs = np.arange(len(arms))
    for k, (kind, label, hatch) in enumerate(kinds):
        vals, errs = [], []
        for arm in arms:
            row = share[(share.arm == arm) & (share.agent_kind == kind)]
            vals.append(100 * float(row["mean"].iloc[0]))
            errs.append(100 * float(row["ci95"].iloc[0]))
        pos = xs + (k - 0.5) * (width + 0.03)
        bars = ax2.bar(pos, vals, width, yerr=errs, capsize=2,
                       color=[COLORS[a] for a in arms], edgecolor="white",
                       linewidth=0.8, hatch=hatch, error_kw={"lw": 0.8},
                       label=label)
        for b, v, e in zip(bars, vals, errs):
            ax2.text(b.get_x() + b.get_width() / 2, v + e + 2, f"{v:.0f}%",
                     ha="center", va="bottom", fontsize=7, color=INK)
    ax2.set_xticks(xs, ["ORA", "Threshold", "Posted\nprice"])
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Agents with a profitable misreport (%)")
    ax2.set_title("(b) Unilateral manipulation", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    leg = ax2.legend(loc="upper left", fontsize=7, frameon=False)
    for patch in leg.get_patches():
        patch.set_facecolor("#cfcfcb")

    fig.tight_layout(w_pad=1.2)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
