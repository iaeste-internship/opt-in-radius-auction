"""Run a global-radius sweep for the VCG ablation and compare it with ORA."""

from pathlib import Path

import numpy as np
import pandas as pd

from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.metrics import compute_metrics
from ora.baselines import run_ablation, radius_violations
from ora.someya import run_someya


# Start with 50 seeds to verify that the experiment runs correctly.
SEEDS = range(50)

# Global-radius values from 0.5 km to 4.0 km in 0.25 km increments.
RADII_KM = np.round(np.arange(0.5, 4.01, 0.25), 2)

PARAMS = DemandParams(
    n_passengers=12,
    n_drivers=4,
    capacity=3,
)

RESULTS_DIR = Path("results")
RAW_OUTPUT = RESULTS_DIR / "radius_sweep_raw.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "radius_sweep_summary.csv"


METRIC_FIELDS = [
    "realised_welfare",
    "match_rate",
    "pooling_rate",
    "mean_wait_min",
    "p95_wait_min",
    "mean_wait_all_min",
    "mean_detour_km",
    "idle_ratio",
    "gini_wait_all",
    "jain_wait_all",
    "gini_earnings",
    "jain_earnings",
    "driver_ir_violation_rate",
    "mean_allocated_driver_utility",
    "min_allocated_driver_utility",
    # Passenger-side utility and acceptability. These are measured for every
    # arm including Someya: they are precisely what the comparison is about,
    # and unlike the driver-side fields they are well defined for a mechanism
    # that treats drivers as fixed-rate executors.
    "passenger_ir_violation_rate",
    "mean_allocated_passenger_utility",
    "min_allocated_passenger_utility",
    "mean_adjusted_passenger_utility",
    "total_acceptability_cost",
    "deficit_pct_of_welfare",
]


# Driver-side fields are out of scope for Someya, which models drivers as
# fixed-rate executors with no reservation and no payment. They are written as
# NaN for that arm so the summary averages skip them (pandas .mean() ignores
# NaN) rather than reporting a misleading number. ORA and the ablation are
# unaffected.
DRIVER_SIDE_FIELDS = [
    "gini_earnings",
    "jain_earnings",
    "driver_ir_violation_rate",
    "mean_allocated_driver_utility",
    "min_allocated_driver_utility",
    "deficit_pct_of_welfare",
]


def make_row(seed, arm, radius_km, metrics, violation_rate,
             driver_side_na=False):
    """Convert the metrics from one run into a CSV row.

    When driver_side_na is True, the six driver-side fields are written as NaN
    instead of their computed values, marking them out of scope for that arm.
    """

    row = {
        "seed": seed,
        "arm": arm,
        "radius_km": radius_km,
        "r_max_violation_rate": violation_rate,
    }

    for field in METRIC_FIELDS:
        if driver_side_na and field in DRIVER_SIDE_FIELDS:
            row[field] = np.nan
        else:
            row[field] = getattr(metrics, field)

    return row


def build_summary(raw):
    """Calculate the mean, standard deviation, and 95% confidence interval."""

    measured_fields = METRIC_FIELDS + ["r_max_violation_rate"]

    grouped = raw.groupby(
        ["arm", "radius_km"],
        as_index=False,
    )

    means = grouped[measured_fields].mean()
    standard_deviations = grouped[measured_fields].std()
    counts = grouped.size().rename(columns={"size": "n_seeds"})

    summary = means.merge(
        standard_deviations,
        on=["arm", "radius_km"],
        suffixes=("_mean", "_std"),
    )

    summary = summary.merge(
        counts,
        on=["arm", "radius_km"],
    )

    for field in measured_fields:
        summary[f"{field}_ci95"] = (
            1.96
            * summary[f"{field}_std"]
            / np.sqrt(summary["n_seeds"])
        )

    return summary


def run():
    """Run ORA and the radius ablation over all seeds."""

    rows = []

    # The same network is used for every seed and mechanism arm.
    network = grid_network(6, 6)

    for seed in SEEDS:
        print(f"Running seed {seed}...")

        rng = np.random.default_rng(seed)

        # Generate passengers and drivers only once for each seed.
        # This ensures that ORA and every ablation radius use the same market.
        passengers, drivers = generate(
            PARAMS,
            network,
            rng,
        )

        passenger_lookup = {
            passenger.id: passenger
            for passenger in passengers
        }

        driver_lookup = {
            driver.id: driver
            for driver in drivers
        }

        # --------------------------------------------------------------
        # ORA: run only once for each seed.
        # --------------------------------------------------------------
        groups, _ = enumerate_groups(
            drivers,
            passengers,
            network,
        )

        ora_result = clear_market(
            groups,
            passenger_lookup,
            driver_lookup,
        )

        ora_metrics = compute_metrics(
            ora_result.allocation,
            passengers,
            drivers,
            network,
        )

        ora_violation_rate = radius_violations(
            ora_result.allocation,
            passengers,
        )[1]

        # --------------------------------------------------------------
        # Someya & Namerikawa (2026): capacity-only feasibility, soft detour
        # and wait penalties, drivers as fixed-rate executors. Like ORA it does
        # not depend on the ablation radius, so it is computed once per seed and
        # repeated across radii as a horizontal reference line.
        # --------------------------------------------------------------
        someya_result = run_someya(
            passengers,
            drivers,
            network,
        )

        someya_metrics = compute_metrics(
            someya_result.allocation,
            passengers,
            drivers,
            network,
        )

        someya_violation_rate = radius_violations(
            someya_result.allocation,
            passengers,
        )[1]

        # --------------------------------------------------------------
        # VCG ablation: run once for each global-radius value.
        # --------------------------------------------------------------
        for radius_km in RADII_KM:
            print(
                f"  radius={radius_km:.2f} km",
                end="\r",
            )

            ablation_result = run_ablation(
                passengers,
                drivers,
                network,
                radius_km=radius_km,
            )

            # Measure the ablation allocation using the passengers' true types,
            # not the platform-wide assumptions used during allocation.
            ablation_metrics = compute_metrics(
                ablation_result.allocation,
                passengers,
                drivers,
                network,
            )

            ablation_violation_rate = radius_violations(
                ablation_result.allocation,
                passengers,
            )[1]

            # Repeat the ORA result for every radius so that it can be plotted
            # directly as a horizontal reference line.
            rows.append(
                make_row(
                    seed=seed,
                    arm="ORA",
                    radius_km=radius_km,
                    metrics=ora_metrics,
                    violation_rate=ora_violation_rate,
                )
            )

            rows.append(
                make_row(
                    seed=seed,
                    arm="VCG ablation",
                    radius_km=radius_km,
                    metrics=ablation_metrics,
                    violation_rate=ablation_violation_rate,
                )
            )

            # Repeat Someya for every radius as a horizontal reference line,
            # with driver-side fields marked NaN (out of scope for its model).
            rows.append(
                make_row(
                    seed=seed,
                    arm="Someya",
                    radius_km=radius_km,
                    metrics=someya_metrics,
                    violation_rate=someya_violation_rate,
                    driver_side_na=True,
                )
            )

        print()

    raw = pd.DataFrame(rows)
    summary = build_summary(raw)

    # Create the results directory if it does not already exist.
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw.to_csv(
        RAW_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print(f"Saved {len(raw)} raw result rows")
    print(f"Raw results: {RAW_OUTPUT}")
    print(f"Summary results: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    run()