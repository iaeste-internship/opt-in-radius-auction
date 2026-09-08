"""Create publication-ready NYC TLC real-data figures for the ORA radius sweep."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_CSV = Path("results") / "real_radius_sweep_summary.csv"
FIGURES_DIR = Path("figures")

DPI = 300

ORA_COLOR = "tab:blue"
ABLATION_COLOR = "tab:orange"
SOMEYA_COLOR = "tab:green"

# The experiment uses 0.25 km steps, but figures display 0.5 km increments
# for readability. The complete results remain in the source CSV.
DISPLAY_RADIUS_STEP_KM = 0.5


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the aggregated NYC TLC radius-sweep results."""

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Real-data summary not found: {SUMMARY_CSV}. "
            "Run run_real_radius_sweep.py first."
        )

    summary = pd.read_csv(SUMMARY_CSV)
    print(f"Loading real-data radius summary from: {SUMMARY_CSV}")

    required_columns = {
        "arm",
        "radius_km",
        "realised_welfare_mean",
        "realised_welfare_ci95",
        "match_rate_mean",
        "match_rate_ci95",
        "r_max_violation_rate_mean",
        "r_max_violation_rate_ci95",
        "mean_adjusted_passenger_utility_mean",
    }

    missing = required_columns.difference(summary.columns)
    if missing:
        raise ValueError(
            "The summary file is missing the following columns: "
            + ", ".join(sorted(missing))
        )

    ora = (
        summary[summary["arm"] == "ORA"]
        .sort_values("radius_km")
        .reset_index(drop=True)
    )
    ablation = (
        summary[summary["arm"] == "VCG ablation"]
        .sort_values("radius_km")
        .reset_index(drop=True)
    )
    someya = (
        summary[summary["arm"] == "Someya"]
        .sort_values("radius_km")
        .reset_index(drop=True)
    )

    if ora.empty:
        raise ValueError("No ORA rows were found in the summary file.")
    if ablation.empty:
        raise ValueError("No VCG ablation rows were found in the summary file.")

    return ora, ablation, someya



def select_display_radius_values(data: pd.DataFrame) -> pd.DataFrame:
    """Keep radius values at 0.5 km increments for clearer figures."""

    if data.empty:
        return data

    scaled_radius = data["radius_km"] / DISPLAY_RADIUS_STEP_KM
    mask = np.isclose(scaled_radius, np.round(scaled_radius))

    return (
        data.loc[mask]
        .sort_values("radius_km")
        .reset_index(drop=True)
    )



def _horizontal_reference(axis, radius_values, mean, ci95, color, label):
    """Draw a radius-independent mechanism as a dashed horizontal CI band."""

    x_min = float(radius_values.min())
    x_max = float(radius_values.max())

    axis.hlines(
        mean,
        x_min,
        x_max,
        color=color,
        linestyle="--",
        linewidth=1.5,
        label=label,
    )
    axis.fill_between(
        [x_min, x_max],
        mean - ci95,
        mean + ci95,
        color=color,
        alpha=0.12,
    )



def _row_at_radius(data: pd.DataFrame, radius_km: float):
    """Return the row for a displayed radius, or None when absent."""

    matches = data[np.isclose(data["radius_km"], radius_km)]
    if matches.empty:
        return None
    return matches.iloc[0]



def _annotate_tradeoff_radii(axis, ablation, y_column):
    """Annotate selected radii without overlapping the 0.5 and 1.0 km labels."""

    row_05 = _row_at_radius(ablation, 0.5)
    row_10 = _row_at_radius(ablation, 1.0)

    # In the real-data experiment these two settings occupy essentially the
    # same trade-off point. Use one combined label instead of overprinting two.
    if row_05 is not None and row_10 is not None:
        x_05 = 100.0 * float(row_05["r_max_violation_rate_mean"])
        y_05 = float(row_05[y_column])
        x_10 = 100.0 * float(row_10["r_max_violation_rate_mean"])
        y_10 = float(row_10[y_column])

        if np.isclose(x_05, x_10, atol=0.5) and np.isclose(y_05, y_10, atol=0.5):
            axis.annotate(
                "0.5–1.0 km",
                xy=((x_05 + x_10) / 2.0, (y_05 + y_10) / 2.0),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
            )
            skip = {0.5, 1.0}
        else:
            skip = set()
    else:
        skip = set()

    label_offsets = {
        0.5: (8, 8),
        1.0: (8, 8),
        1.5: (8, 8),
        2.0: (8, 8),
        2.5: (8, 8),
        3.0: (8, 8),
        4.0: (8, 8),
    }

    for radius_km, offset in label_offsets.items():
        if radius_km in skip:
            continue

        row = _row_at_radius(ablation, radius_km)
        if row is None:
            continue

        axis.annotate(
            f"{radius_km:.1f} km",
            xy=(
                100.0 * float(row["r_max_violation_rate_mean"]),
                float(row[y_column]),
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )


# ---------------------------------------------------------------------------
# Figure 1: welfare and match rate
# ---------------------------------------------------------------------------


def plot_welfare_and_match_rate(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot welfare and match rate for the NYC TLC validation."""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 9),
        sharex=True,
    )

    welfare_axis = axes[0]

    _horizontal_reference(
        welfare_axis,
        ablation["radius_km"],
        ora["realised_welfare_mean"].mean(),
        ora["realised_welfare_ci95"].mean(),
        ORA_COLOR,
        "ORA",
    )

    welfare_axis.plot(
        ablation["radius_km"],
        ablation["realised_welfare_mean"],
        marker="o",
        color=ABLATION_COLOR,
        label="VCG ablation",
    )
    welfare_axis.fill_between(
        ablation["radius_km"],
        ablation["realised_welfare_mean"] - ablation["realised_welfare_ci95"],
        ablation["realised_welfare_mean"] + ablation["realised_welfare_ci95"],
        color=ABLATION_COLOR,
        alpha=0.15,
    )

    if not someya.empty:
        _horizontal_reference(
            welfare_axis,
            ablation["radius_km"],
            someya["realised_welfare_mean"].mean(),
            someya["realised_welfare_ci95"].mean(),
            SOMEYA_COLOR,
            "Someya",
        )

    welfare_axis.set_ylabel(r"Nominal welfare $W^{\mathrm{nom}}$")
    welfare_axis.set_title(r"Nominal welfare $W^{\mathrm{nom}}$")
    # Both panels are read independently in the paper, so the upper panel keeps
    # its own tick labels even though the x axis is shared.
    welfare_axis.tick_params(labelbottom=True)
    welfare_axis.grid(alpha=0.3)
    welfare_axis.legend()

    match_axis = axes[1]

    _horizontal_reference(
        match_axis,
        ablation["radius_km"],
        100.0 * ora["match_rate_mean"].mean(),
        100.0 * ora["match_rate_ci95"].mean(),
        ORA_COLOR,
        "ORA",
    )

    match_mean = 100.0 * ablation["match_rate_mean"]
    match_ci95 = 100.0 * ablation["match_rate_ci95"]

    match_axis.plot(
        ablation["radius_km"],
        match_mean,
        marker="o",
        color=ABLATION_COLOR,
        label="VCG ablation",
    )
    match_axis.fill_between(
        ablation["radius_km"],
        match_mean - match_ci95,
        match_mean + match_ci95,
        color=ABLATION_COLOR,
        alpha=0.15,
    )

    if not someya.empty:
        _horizontal_reference(
            match_axis,
            ablation["radius_km"],
            100.0 * someya["match_rate_mean"].mean(),
            100.0 * someya["match_rate_ci95"].mean(),
            SOMEYA_COLOR,
            "Someya",
        )

    match_axis.set_xlabel("Global-radius value used by the ablation (km)")
    match_axis.set_ylabel("Match rate (%)")
    match_axis.set_title("Match rate")
    match_axis.grid(alpha=0.3)
    match_axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "real_radius_welfare_match_rate.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Figure 2: pickup-radius violations
# ---------------------------------------------------------------------------


def plot_radius_violations(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot pickup-radius violations for the NYC TLC validation."""

    fig, axis = plt.subplots(figsize=(8, 5))

    _horizontal_reference(
        axis,
        ablation["radius_km"],
        100.0 * ora["r_max_violation_rate_mean"].mean(),
        100.0 * ora["r_max_violation_rate_ci95"].mean(),
        ORA_COLOR,
        "ORA",
    )

    violation_mean = 100.0 * ablation["r_max_violation_rate_mean"]
    violation_ci95 = 100.0 * ablation["r_max_violation_rate_ci95"]

    axis.plot(
        ablation["radius_km"],
        violation_mean,
        marker="o",
        color=ABLATION_COLOR,
        label="VCG ablation",
    )
    axis.fill_between(
        ablation["radius_km"],
        violation_mean - violation_ci95,
        violation_mean + violation_ci95,
        color=ABLATION_COLOR,
        alpha=0.15,
    )

    if not someya.empty:
        _horizontal_reference(
            axis,
            ablation["radius_km"],
            100.0 * someya["r_max_violation_rate_mean"].mean(),
            100.0 * someya["r_max_violation_rate_ci95"].mean(),
            SOMEYA_COLOR,
            "Someya",
        )

    axis.set_xlabel("Global-radius value used by the ablation (km)")
    axis.set_ylabel(r"Served passengers beyond true $r_{\max}$ (%)")
    axis.set_title("Pickup-radius violations")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "real_radius_violation_rate.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Figure 3: nominal welfare versus violations
# ---------------------------------------------------------------------------


def plot_welfare_vs_violations(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot nominal welfare against pickup-radius violations."""

    fig, axis = plt.subplots(figsize=(8, 6))

    violation_percentage = 100.0 * ablation["r_max_violation_rate_mean"]

    axis.plot(
        violation_percentage,
        ablation["realised_welfare_mean"],
        marker="o",
        color=ABLATION_COLOR,
        label="VCG ablation",
    )

    _annotate_tradeoff_radii(
        axis,
        ablation,
        "realised_welfare_mean",
    )

    ora_welfare = ora["realised_welfare_mean"].mean()
    ora_violation = 100.0 * ora["r_max_violation_rate_mean"].mean()

    axis.scatter(
        ora_violation,
        ora_welfare,
        marker="s",
        s=70,
        color=ORA_COLOR,
        label="ORA",
        zorder=3,
    )
    axis.annotate(
        "ORA",
        xy=(ora_violation, ora_welfare),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    if not someya.empty:
        someya_welfare = someya["realised_welfare_mean"].mean()
        someya_violation = 100.0 * someya["r_max_violation_rate_mean"].mean()

        axis.scatter(
            someya_violation,
            someya_welfare,
            marker="^",
            s=80,
            color=SOMEYA_COLOR,
            label="Someya",
            zorder=3,
        )
        axis.annotate(
            "Someya",
            xy=(someya_violation, someya_welfare),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    axis.set_xlabel(r"Served passengers beyond true $r_{\max}$ (%)")
    axis.set_ylabel(r"Nominal welfare $W^{\mathrm{nom}}$")
    axis.set_title("Nominal welfare versus pickup-radius violations")
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "real_welfare_vs_radius_violations.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Figure 4: adjusted passenger utility versus violations
# ---------------------------------------------------------------------------


def plot_passenger_utility_vs_violations(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot acceptability-adjusted passenger utility against violations."""

    fig, axis = plt.subplots(figsize=(8, 6))

    violation_percentage = 100.0 * ablation["r_max_violation_rate_mean"]

    axis.plot(
        violation_percentage,
        ablation["mean_adjusted_passenger_utility_mean"],
        marker="o",
        color=ABLATION_COLOR,
        label="VCG ablation",
    )

    _annotate_tradeoff_radii(
        axis,
        ablation,
        "mean_adjusted_passenger_utility_mean",
    )

    ora_violation = 100.0 * ora["r_max_violation_rate_mean"].mean()
    ora_utility = ora["mean_adjusted_passenger_utility_mean"].mean()

    axis.scatter(
        ora_violation,
        ora_utility,
        marker="s",
        s=70,
        color=ORA_COLOR,
        label="ORA",
        zorder=3,
    )
    axis.annotate(
        "ORA",
        xy=(ora_violation, ora_utility),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    if not someya.empty:
        someya_violation = 100.0 * someya["r_max_violation_rate_mean"].mean()
        someya_utility = someya["mean_adjusted_passenger_utility_mean"].mean()

        axis.scatter(
            someya_violation,
            someya_utility,
            marker="^",
            s=80,
            color=SOMEYA_COLOR,
            label="Someya",
            zorder=3,
        )
        axis.annotate(
            "Someya",
            xy=(someya_violation, someya_utility),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    axis.axhline(0.0, linewidth=1.0, linestyle=":", color="0.35")
    axis.set_xlabel(r"Served passengers beyond true $r_{\max}$ (%)")
    axis.set_ylabel("Mean adjusted passenger utility")
    axis.set_title("Adjusted passenger utility versus pickup-radius violations")
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "real_passenger_utility_vs_violations.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Create all NYC TLC real-data radius-sweep figures."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ora, ablation, someya = load_results()

    ora_display = select_display_radius_values(ora)
    ablation_display = select_display_radius_values(ablation)
    someya_display = (
        select_display_radius_values(someya)
        if not someya.empty
        else someya
    )

    plot_welfare_and_match_rate(
        ora_display,
        ablation_display,
        someya_display,
    )
    plot_radius_violations(
        ora_display,
        ablation_display,
        someya_display,
    )
    plot_welfare_vs_violations(
        ora_display,
        ablation_display,
        someya_display,
    )
    plot_passenger_utility_vs_violations(
        ora_display,
        ablation_display,
        someya_display,
    )

    print("All NYC TLC real-data figures created successfully.")


if __name__ == "__main__":
    main()
