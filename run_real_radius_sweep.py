"""Run the ORA radius sweep on real NYC TLC HVFHV demand.

This is the real-data twin of run_radius_sweep.py. The mechanism arms, radius
values, metrics, paired seeds, and summary/CI calculation are unchanged.
Only the market source changes.

OPTION B (empirical trip-distance matrix):
1. Build the Manhattan taxi-zone mapping and a complete geometric fallback
   network from TLC zone geometry.
2. Extract the Manhattan HVFHV trip pool.
3. Replace every directed O/D cell supported by at least MIN_OD_OBSERVATIONS
   real trips with the median observed TLC trip distance. Thin O/D cells retain
   the geometric fallback. This empirical ZoneNetwork is the network supplied
   to ORA, the VCG ablation, Someya, and the common metrics layer.
4. Synthetic O/D generation is replaced by ora.real_demand.generate_from_trips.

Passenger valuations and flexibility parameters still come from the same
DemandParams distributions used by the synthetic experiment.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ora.demand import DemandParams
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.metrics import compute_metrics
from ora.baselines import run_ablation, radius_violations
from ora.someya import run_someya
from ora.osm_network import (
    build_real_network,
    build_empirical_trip_distance_network,
)
from ora.real_demand import (
    DEFAULT_DAYPARTS,
    extract_pool,
    generate_from_trips,
    stratum_weights,
)


# Start with the same 50 paired seeds as the synthetic radius sweep.
SEEDS = range(50)

# Same global-radius values: 0.5 km to 4.0 km in 0.25 km increments.
RADII_KM = np.round(np.arange(0.5, 4.01, 0.25), 2)

# Identical market size and valuation/tolerance distributions.
PARAMS = DemandParams(
    n_passengers=12,
    n_drivers=4,
    capacity=3,
)

# ---------------------------------------------------------------------------
# Real-data inputs.
#
# Edit only these paths for your local TLC data layout. The .npz caches should
# preferably live outside the OneDrive-synced project tree on your Windows
# machine to avoid intermittent file-locking errors.
# ---------------------------------------------------------------------------
TLC_ZONE_SHP = Path("data") / "taxi_zones" / "taxi_zones.shp"
TLC_ZONE_LOOKUP = Path("data") / "taxi_zones" / "taxi_zone_lookup.csv"
HVFHV_PARQUET = Path("data") / "hvfhv"

CACHE_DIR = Path.home() / ".ora_cache"
NETWORK_CACHE = CACHE_DIR / "manhattan_zone_network.npz"
TRIP_POOL_CACHE = CACHE_DIR / "manhattan_hvfhv_pool.npz"
EMPIRICAL_NETWORK_CACHE = CACHE_DIR / "manhattan_empirical_trip_network.npz"
MIN_OD_OBSERVATIONS = 20

RESULTS_DIR = Path("results")
RAW_OUTPUT = RESULTS_DIR / "real_radius_sweep_raw.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "real_radius_sweep_summary.csv"
DISTANCE_DIAGNOSTICS_OUTPUT = RESULTS_DIR / "real_distance_matrix_diagnostics.csv"
DEMAND_STRATA_OUTPUT = RESULTS_DIR / "real_demand_strata.csv"


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
# NaN for that arm so the summary averages skip them rather than reporting a
# misleading number. ORA and the ablation are unaffected.
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
    """Run ORA, Someya, and the radius ablation on real HVFHV markets."""

    rows = []

    # ------------------------------------------------------------------
    # REAL-DATA SWAP 1: use Manhattan taxi-zone geography instead of the
    # synthetic 6 x 6 grid. The returned ZoneNetwork has the same interface
    # used by every mechanism and metric downstream.
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fallback_network, location_ids, loc_to_index = build_real_network(
        shp_path=str(TLC_ZONE_SHP),
        lookup_csv=str(TLC_ZONE_LOOKUP),
        borough="Manhattan",
        cache_path=str(NETWORK_CACHE),
    )

    # Extract/load the slim Manhattan HVFHV pool once. If the cache already
    # exists, extract_pool loads it without scanning the raw Parquet again.
    pool = extract_pool(
        parquet_path=str(HVFHV_PARQUET),
        loc_to_index=loc_to_index,
        cache_path=str(TRIP_POOL_CACHE),
    )

    # OPTION B: replace well-observed directed O/D cells with median empirical
    # TLC trip distances. Only thin cells retain the geometric fallback.
    network, distance_diagnostics = build_empirical_trip_distance_network(
        pool,
        fallback_network,
        location_ids,
        cache_path=str(EMPIRICAL_NETWORK_CACHE),
        min_observations=MIN_OD_OBSERVATIONS,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([distance_diagnostics]).to_csv(
        DISTANCE_DIAGNOSTICS_OUTPUT,
        index=False,
    )

    # Save the real daypart x weekday/weekend demand composition used by the
    # market sampler, so the paper can document full-day demand coverage.
    weights = stratum_weights(pool, DEFAULT_DAYPARTS)
    strata_rows = []
    n_pool = len(pool["o_zone"])
    for (daypart_index, weekend), weight in sorted(weights.items()):
        daypart_name, start_hour, end_hour = DEFAULT_DAYPARTS[daypart_index]
        strata_rows.append(
            {
                "daypart": daypart_name,
                "start_hour": start_hour,
                "end_hour": end_hour,
                "is_weekend": bool(weekend),
                "demand_share": float(weight),
                "approx_trip_count": int(round(float(weight) * n_pool)),
            }
        )
    pd.DataFrame(strata_rows).to_csv(DEMAND_STRATA_OUTPUT, index=False)

    print(
        f"Real Manhattan network: {network.n_zones} zones "
        f"({len(location_ids)} TLC LocationIDs)"
    )
    print(f"HVFHV Manhattan trip pool: {len(pool['o_zone'])} trips")
    print(
        "OPTION B empirical OD pair coverage: "
        f"{100.0 * distance_diagnostics['empirical_od_pair_coverage']:.2f}% "
        f"(threshold >= {MIN_OD_OBSERVATIONS} trips/pair)"
    )
    print(
        "OPTION B trip mass supporting empirical OD cells: "
        f"{100.0 * distance_diagnostics['empirical_trip_coverage']:.2f}%"
    )
    print(f"Distance diagnostics: {DISTANCE_DIAGNOSTICS_OUTPUT}")
    print(f"Demand strata: {DEMAND_STRATA_OUTPUT}")

    for seed in SEEDS:
        print(f"Running seed {seed}...")

        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # REAL-DATA SWAP 2: generate the market from real HVFHV trip O/D and
        # real drop-off locations. Passenger valuations/tolerances and driver
        # economic parameters still come from the identical PARAMS object.
        # ------------------------------------------------------------------
        passengers, drivers = generate_from_trips(
            pool,
            PARAMS,
            network,
            loc_to_index,
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
        # Someya & Namerikawa (2026): unchanged from the synthetic sweep.
        # It does not depend on the ablation radius, so it is computed once per
        # seed and repeated across radii as a horizontal reference line.
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
        # VCG ablation: unchanged; run once for every global-radius value.
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

            # Repeat ORA at every radius so it can be plotted directly as a
            # horizontal reference line, exactly as in the synthetic sweep.
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
