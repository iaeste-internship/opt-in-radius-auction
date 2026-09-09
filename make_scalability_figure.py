"""Create the runtime-decomposition figure from the scalability benchmark.

Reproduces Fig. 3 of the paper: median runtime with interquartile bands for the
main allocation WDP, the Clarke payment counterfactuals, and total clearing,
against passenger count at the fixed 3:1 passenger/driver ratio.

Run from the project root, after run_scalability_benchmark.py:

    python make_scalability_figure.py

Input (the fresh CSV is authoritative; the workbook is a checked fallback):

    results/scalability_summary.csv
    ORA_scalability.xlsx

Output:

    figures/scalability_runtime_breakdown.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_CSV = Path("results") / "scalability_summary.csv"
SUMMARY_WORKBOOK = Path("ORA_scalability.xlsx")
SUMMARY_SHEET = "summary"
import figure_style

figure_style.apply_ieee_style()

FIGURES_DIR = Path("figures")
OUTPUT_FILE = FIGURES_DIR / "scalability_runtime_breakdown.png"

DPI = 300

# (column stem, legend label, colour)
SERIES = [
    ("main_wdp_time_s", "Main allocation WDP", "tab:blue"),
    ("payment_overhead_s", "Clarke counterfactuals", "tab:orange"),
    ("full_clear_time_s", "Total clearing", "tab:green"),
]

REQUIRED_COLUMNS = ["n_passengers", "n_seeds"] + [
    f"{stem}_{suffix}" for stem, _, _ in SERIES for suffix in ("mean", "ci95")
]


def load_summary() -> pd.DataFrame:
    """Load the aggregated scalability summary.

    A newly generated CSV takes precedence, so the figure cannot silently show
    stale numbers just because the archived workbook has not been rebuilt.
    """

    if SUMMARY_CSV.exists():
        summary = pd.read_csv(SUMMARY_CSV)
        print(f"Loading scalability summary from: {SUMMARY_CSV}")
    elif SUMMARY_WORKBOOK.exists():
        summary = pd.read_excel(SUMMARY_WORKBOOK, sheet_name=SUMMARY_SHEET)
        print(f"Loading scalability summary from: {SUMMARY_WORKBOOK}")
    else:
        raise FileNotFoundError(
            f"Neither {SUMMARY_CSV} nor {SUMMARY_WORKBOOK} was found. "
            "Run run_scalability_benchmark.py first."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in summary.columns]
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")

    summary = summary.sort_values("n_passengers").reset_index(drop=True)

    if summary["n_passengers"].duplicated().any():
        raise ValueError("Summary contains duplicated passenger counts.")

    n_seeds = sorted(set(summary["n_seeds"].astype(int)))
    if len(n_seeds) != 1:
        print(f"WARNING: seed count varies across market sizes: {n_seeds}")

    return summary


STYLES = {
    "main_wdp_time_s": {"marker": "o", "linestyle": "-"},
    "payment_overhead_s": {"marker": "s", "linestyle": "--"},
    "full_clear_time_s": {"marker": "^", "linestyle": ":"},
}


def make_figure(summary: pd.DataFrame) -> None:
    x = summary["n_passengers"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figure_style.figsize_for("scalability_runtime_breakdown", (6.0, 4.0)))

    for stem, label, color in SERIES:
        median = summary[f"{stem}_median"].to_numpy(dtype=float)
        iqr = summary[f"{stem}_iqr"].to_numpy(dtype=float)
        # Clarke-counterfactual and total-clearing curves nearly coincide, so
        # they need distinct dash patterns and markers to stay separable.
        style = STYLES[stem]
        ax.plot(
            x,
            median,
            marker=style["marker"],
            markersize=4,
            linestyle=style["linestyle"],
            color=color,
            label=label,
        )
        # Medians with interquartile bands: runtime distributions are skewed by
        # occasional slow solver starts, which distort means and their intervals.
        # The lower edge is clipped because the vertical axis is logarithmic.
        lower = [max(m - h / 2, 1e-4) for m, h in zip(median, iqr)]
        upper = [m + h / 2 for m, h in zip(median, iqr)]
        ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("Passengers (3:1 passenger/driver ratio)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=DPI)
    plt.close(fig)

    n_seeds = int(summary["n_seeds"].max())
    print(f"Wrote {OUTPUT_FILE} ({len(x)} market sizes, {n_seeds} seeds)")
    print(
        "Caption should read: median runtimes with interquartile ranges "
        f"over {n_seeds} seeds."
    )


def main() -> None:
    make_figure(load_summary())


if __name__ == "__main__":
    main()
