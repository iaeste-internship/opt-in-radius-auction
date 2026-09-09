"""Create the main figures from the ORA radius-sweep experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prefer the live CSV produced by run_radius_sweep.py.  If it is absent,
# fall back to the checked consolidated workbook shipped with the project.
SUMMARY_CSV = Path("results") / "radius_sweep_summary.csv"
SUMMARY_WORKBOOK = Path("ORA_radius_sweep.xlsx")
SUMMARY_SHEET = "summary"
import figure_style

figure_style.apply_ieee_style()

FIGURES_DIR = Path("figures")

DPI = 300

ORA_COLOR = "tab:blue"
ABLATION_COLOR = "tab:orange"
SOMEYA_COLOR = "tab:green"

# Show only 0.5 km increments to avoid repeated staircase points.

DISPLAY_RADIUS_STEP_KM = 0.5


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the aggregated radius-sweep results.

    A newly generated CSV is authoritative when present.  The workbook is a
    checked archive/fallback, so figure generation cannot silently ignore a
    fresh sweep just because ORA_radius_sweep.xlsx has not been rebuilt yet.
    """

    if SUMMARY_CSV.exists():
        summary = pd.read_csv(SUMMARY_CSV)
        print(f"Loading radius summary from: {SUMMARY_CSV}")
    elif SUMMARY_WORKBOOK.exists():
        summary = pd.read_excel(
            SUMMARY_WORKBOOK,
            sheet_name=SUMMARY_SHEET,
        )
        print(
            "Live CSV not found; loading archived radius summary from: "
            f"{SUMMARY_WORKBOOK} [{SUMMARY_SHEET}]"
        )
    else:
        raise FileNotFoundError(
            "No radius-sweep summary was found. Expected either "
            f"{SUMMARY_CSV} (preferred) or {SUMMARY_WORKBOOK} "
            f"sheet '{SUMMARY_SHEET}'. Run run_radius_sweep.py to create "
            "fresh CSV results, or place the archived workbook next to "
            "this script."
        )

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
        raise ValueError(
            "No VCG ablation rows were found in the summary file."
        )

    # Someya is optional: older summaries may predate the third arm. Figures
    # still render with just ORA and the ablation when it is absent.
    return ora, ablation, someya


def select_display_radius_values(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Keep radius values at 0.5 km increments for clearer figures."""

    scaled_radius = data["radius_km"] / DISPLAY_RADIUS_STEP_KM

    mask = np.isclose(
        scaled_radius,
        np.round(scaled_radius),
    )

    return (
        data.loc[mask]
        .sort_values("radius_km")
        .reset_index(drop=True)
    )


def _horizontal_reference(axis, radius_values, mean, ci95, color, label):
    """Draw a radius-independent arm as a dashed horizontal band.

    Someya (like ORA) does not depend on the ablation's platform-wide radius,
    so it is shown as a dashed horizontal line spanning the radius axis, with a
    shaded 95% CI band, and a label that makes the radius-independence explicit
    rather than implying a sweep.
    """
    x_min = float(radius_values.min())
    x_max = float(radius_values.max())
    axis.hlines(
        mean,
        x_min,
        x_max,
        color=color,
        linestyle="--",
        label=label,
    )
    axis.fill_between(
        [x_min, x_max],
        mean - ci95,
        mean + ci95,
        color=color,
        alpha=0.12,
    )


def plot_welfare_and_match_rate(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot welfare and match rate across global-radius values."""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figure_style.figsize_for("radius_welfare_match_rate", (8, 9)),
        sharex=True,
    )

    welfare_axis = axes[0]

    for data, label, color in (
        (ora, "ORA", ORA_COLOR),
        (ablation, "VCG ablation", ABLATION_COLOR),
    ):
        welfare_axis.plot(
            data["radius_km"],
            data["realised_welfare_mean"],
            marker="o",
            color=color,
            label=label,
        )

        welfare_axis.fill_between(
            data["radius_km"],
            data["realised_welfare_mean"]
            - data["realised_welfare_ci95"],
            data["realised_welfare_mean"]
            + data["realised_welfare_ci95"],
            color=color,
            alpha=0.15,
        )

    if not someya.empty:
        _horizontal_reference(
            welfare_axis,
            ablation["radius_km"],
            someya["realised_welfare_mean"].mean(),
            someya["realised_welfare_ci95"].mean(),
            SOMEYA_COLOR,
            "Someya (no radius)",
        )

    welfare_axis.set_ylabel(r"Nominal welfare $W^{\mathrm{nom}}$")
    welfare_axis.set_title(
        r"Nominal welfare $W^{\mathrm{nom}}$ across global-radius values"
    )
    # Both panels are read independently in the paper, so the upper panel keeps
    # its own tick labels even though the x axis is shared.
    welfare_axis.tick_params(labelbottom=True)
    welfare_axis.grid(alpha=0.3)
    welfare_axis.legend()

    match_axis = axes[1]

    for data, label, color in (
        (ora, "ORA", ORA_COLOR),
        (ablation, "VCG ablation", ABLATION_COLOR),
    ):
        match_mean = 100.0 * data["match_rate_mean"]
        match_ci95 = 100.0 * data["match_rate_ci95"]

        match_axis.plot(
            data["radius_km"],
            match_mean,
            marker="o",
            color=color,
            label=label,
        )

        match_axis.fill_between(
            data["radius_km"],
            match_mean - match_ci95,
            match_mean + match_ci95,
            color=color,
            alpha=0.15,
        )

    if not someya.empty:
        _horizontal_reference(
            match_axis,
            ablation["radius_km"],
            100.0 * someya["match_rate_mean"].mean(),
            100.0 * someya["match_rate_ci95"].mean(),
            SOMEYA_COLOR,
            "Someya (no radius)",
        )

    match_axis.set_xlabel(
        "Global radius used by ablation (km)"
    )
    match_axis.set_ylabel("Match rate (%)")
    match_axis.set_title(
        "Match rate across global-radius values"
    )
    match_axis.grid(alpha=0.3)
    match_axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "radius_welfare_match_rate.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


def plot_radius_violations(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot pickup-radius violations across global-radius values."""

    fig, axis = plt.subplots(figsize=figure_style.figsize_for("radius_violation_rate", (8, 5)))

    for data, label, color in (
        (ora, "ORA", ORA_COLOR),
        (ablation, "VCG ablation", ABLATION_COLOR),
    ):
        violation_mean = (
            100.0 * data["r_max_violation_rate_mean"]
        )

        violation_ci95 = (
            100.0 * data["r_max_violation_rate_ci95"]
        )

        axis.plot(
            data["radius_km"],
            violation_mean,
            marker="o",
            color=color,
            label=label,
        )

        axis.fill_between(
            data["radius_km"],
            violation_mean - violation_ci95,
            violation_mean + violation_ci95,
            color=color,
            alpha=0.15,
        )

    if not someya.empty:
        _horizontal_reference(
            axis,
            ablation["radius_km"],
            100.0 * someya["r_max_violation_rate_mean"].mean(),
            100.0 * someya["r_max_violation_rate_ci95"].mean(),
            SOMEYA_COLOR,
            "Someya (no radius)",
        )

    axis.set_xlabel(
        "Global radius used by ablation (km)"
    )
    axis.set_ylabel(
        r"Served passengers beyond true $r_{\max}$ (%)"
    )
    axis.set_title(
        "Pickup-radius violations across global-radius values"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "radius_violation_rate.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


def plot_welfare_vs_violations(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot the welfare-versus-acceptability trade-off."""

    fig, axis = plt.subplots(figsize=(8, 6))

    violation_percentage = (
        100.0 * ablation["r_max_violation_rate_mean"]
    )

    axis.plot(
        violation_percentage,
        ablation["realised_welfare_mean"],
        marker="o",
        color=ABLATION_COLOR,
        label="VCG ablation",
    )

    # Label selected radius values while avoiding excessive overlap.
    label_offsets = {
        0.5: (6, 6),
        1.0: (6, 6),
        1.5: (6, 6),
        2.0: (6, 6),
        2.5: (6, 6),
        3.0: (6, 6),
        4.0: (6, 6),
    }

    for _, row in ablation.iterrows():
        radius_km = float(row["radius_km"])

        if radius_km not in label_offsets:
            continue

        axis.annotate(
            f"{radius_km:.1f} km",
            xy=(
                100.0 * row["r_max_violation_rate_mean"],
                row["realised_welfare_mean"],
            ),
            xytext=label_offsets[radius_km],
            textcoords="offset points",
            fontsize=figure_style.ANNOTATION_PT,
        )

    # ORA is represented by one point because its outcome does not depend
    # on the platform-wide radius used by the ablation.
    ora_welfare = ora["realised_welfare_mean"].mean()

    ora_violation = (
        100.0 * ora["r_max_violation_rate_mean"].mean()
    )

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
        fontsize=figure_style.ANNOTATION_PT,
    )

    # Someya is also a single point: its outcome does not depend on the
    # ablation's platform-wide radius. It sits up and to the right of the
    # ablation curve, the highest welfare and the highest violation rate.
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
            fontsize=figure_style.ANNOTATION_PT,
        )

    axis.set_xlabel(
        r"Served passengers beyond true $r_{\max}$ (%)"
    )
    axis.set_ylabel(r"Nominal welfare $W^{\mathrm{nom}}$")
    axis.set_title(
        r"Nominal welfare $W^{\mathrm{nom}}$ versus pickup-radius violations"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "welfare_vs_radius_violations.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


def plot_passenger_utility_vs_violations(
    ora: pd.DataFrame,
    ablation: pd.DataFrame,
    someya: pd.DataFrame,
) -> None:
    """Plot passenger utility against pickup-radius violations.

    The companion to the welfare trade-off figure, and its counterweight. In
    that figure welfare rises as violations rise, which on its own reads as an
    argument for ignoring the reported bounds. Here the vertical axis is what
    the passenger actually ends up holding once the part of her route beyond
    those bounds is charged for, and it falls as violations rise. The two
    figures together say that the welfare bought by overshooting the bounds is
    not welfare the passenger receives.
    """

    fig, axis = plt.subplots(figsize=(8, 6))

    violation_percentage = (
        100.0 * ablation["r_max_violation_rate_mean"]
    )

    axis.plot(
        violation_percentage,
        ablation["mean_adjusted_passenger_utility_mean"],
        marker="o",
        color=ABLATION_COLOR,
        label="VCG ablation",
    )

    label_offsets = {
        0.5: (6, 6),
        1.0: (6, 6),
        1.5: (6, 6),
        2.0: (6, 6),
        2.5: (6, 6),
        3.0: (6, 6),
        4.0: (6, 6),
    }

    for _, row in ablation.iterrows():
        radius_km = float(row["radius_km"])

        if radius_km not in label_offsets:
            continue

        axis.annotate(
            f"{radius_km:.1f} km",
            xy=(
                100.0 * row["r_max_violation_rate_mean"],
                row["mean_adjusted_passenger_utility_mean"],
            ),
            xytext=label_offsets[radius_km],
            textcoords="offset points",
            fontsize=figure_style.ANNOTATION_PT,
        )

    # ORA is one point: its outcome does not depend on the ablation's radius.
    axis.scatter(
        100.0 * ora["r_max_violation_rate_mean"].mean(),
        ora["mean_adjusted_passenger_utility_mean"].mean(),
        marker="s",
        s=70,
        color=ORA_COLOR,
        label="ORA",
        zorder=3,
    )

    axis.annotate(
        "ORA",
        xy=(
            100.0 * ora["r_max_violation_rate_mean"].mean(),
            ora["mean_adjusted_passenger_utility_mean"].mean(),
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=figure_style.ANNOTATION_PT,
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
            fontsize=figure_style.ANNOTATION_PT,
        )

    axis.set_xlabel(
        r"Served passengers beyond true $r_{\max}$ (%)"
    )
    axis.set_ylabel("Mean adjusted passenger utility")
    axis.set_title(
        "Passenger utility versus pickup-radius violations"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()

    output = FIGURES_DIR / "passenger_utility_vs_violations.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


def main() -> None:
    """Create all radius-sweep figures."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ora, ablation, someya = load_results()

    # Keep the full experimental results in the source data, but display only
    # 0.5 km increments to make the staircase pattern easier to read.
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

    print("All figures created successfully.")


if __name__ == "__main__":
    main()
