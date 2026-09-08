"""Experiment C: does ORA's value change with demand/supply pressure?

Scientific question
-------------------
The factorial ablation identified what passenger-specific radius and pooling-value
elicitation contribute. The heterogeneity sweep showed when those contributions
matter. This experiment asks a third question: do those effects change when the
market becomes supply constrained?

Design
------
Drivers and vehicle capacity are held fixed at 4 drivers x 3 seats. Passenger
counts are 8, 12, and 16, giving passenger/driver ratios 2, 3, and 4 and nominal
demand/seat-capacity ratios 0.67, 1.00, and 1.33.

The markets are NESTED within seed:
- the 12-passenger market is generated exactly as in the existing factorial
  ablation, so it is an anchor to Experiment A;
- the 8-passenger market uses the first 8 passengers from that same market;
- the 16-passenger market adds 4 independently generated passengers while
  keeping the exact same 4 drivers.

This isolates increasing demand pressure while holding the driver realization
fixed. The four factorial arms are rerun at every pressure level:
- ORA
- No radius elicitation
- No pooling elicitation
- Full ablation

The global settings match Experiment A: radius = 2.0 km and common share
discount = 0.75. Effects are computed seed-by-seed using the same 2x2 factorial
contrasts as Experiment A, then trends are estimated within seed across the
three pressure levels.

Outputs
-------
results/market_pressure_raw.csv
results/market_pressure_summary.csv
results/market_pressure_effects.csv
results/market_pressure_trends.csv

If results/factorial_ablation_raw.csv exists, the script checks that the
12-passenger rows reproduce Experiment A (within numerical tolerance).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.metrics import compute_metrics
from ora.baselines import (
    DEFAULT_RADIUS_KM,
    DEFAULT_SHARE_DISCOUNT,
    radius_violations,
    run_ablation,
    run_radius_only_ablation,
    run_share_only_ablation,
)

SEEDS = range(50)
N_DRIVERS = 4
CAPACITY = 3
PASSENGER_LEVELS = (8, 12, 16)
BASE_PASSENGERS = 12
EXTRA_PASSENGERS = max(PASSENGER_LEVELS) - BASE_PASSENGERS
RADIUS_KM = DEFAULT_RADIUS_KM
SHARE_DISCOUNT = DEFAULT_SHARE_DISCOUNT

RESULTS_DIR = Path("results")
RAW_OUTPUT = RESULTS_DIR / "market_pressure_raw.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "market_pressure_summary.csv"
EFFECTS_OUTPUT = RESULTS_DIR / "market_pressure_effects.csv"
TRENDS_OUTPUT = RESULTS_DIR / "market_pressure_trends.csv"
FACTORIAL_ANCHOR = RESULTS_DIR / "factorial_ablation_raw.csv"

ARM_ORA = "ORA"
ARM_NO_RADIUS = "No radius elicitation"
ARM_NO_POOLING = "No pooling elicitation"
ARM_FULL = "Full ablation"
ARMS = (ARM_ORA, ARM_NO_RADIUS, ARM_NO_POOLING, ARM_FULL)

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
    "passenger_ir_violation_rate",
    "mean_allocated_passenger_utility",
    "min_allocated_passenger_utility",
    "mean_adjusted_passenger_utility",
    "total_acceptability_cost",
    "deficit_pct_of_welfare",
]

HEADLINE = [
    "realised_welfare",
    "match_rate",
    "pooling_rate",
    "mean_wait_all_min",
    "r_max_violation_rate",
    "passenger_ir_violation_rate",
    "mean_allocated_passenger_utility",
    "mean_adjusted_passenger_utility",
    "total_acceptability_cost",
    "deficit_pct_of_welfare",
]


def build_nested_markets(seed, network):
    """Return {8: ..., 12: ..., 16: ...} using one fixed driver realization.

    The 12-passenger market is generated with the exact Experiment-A call, so
    the baseline pressure level can be checked against the existing factorial
    results. Low pressure removes passengers; high pressure adds independent
    passengers without redrawing drivers.
    """
    base_rng = np.random.default_rng(seed)
    base_params = DemandParams(
        n_passengers=BASE_PASSENGERS,
        n_drivers=N_DRIVERS,
        capacity=CAPACITY,
    )
    base_passengers, drivers = generate(base_params, network, base_rng)

    markets = {
        8: (base_passengers[:8], drivers),
        12: (base_passengers, drivers),
    }

    if EXTRA_PASSENGERS > 0:
        # Separate deterministic stream: add demand without moving the original
        # passengers or redrawing the original drivers.
        extra_seed = np.random.SeedSequence([int(seed), 20260816, 314159])
        extra_rng = np.random.default_rng(extra_seed)
        extra_params = DemandParams(
            n_passengers=EXTRA_PASSENGERS,
            n_drivers=0,
            capacity=CAPACITY,
        )
        extra, _ = generate(extra_params, network, extra_rng)
        extra = [
            replace(passenger, id=BASE_PASSENGERS + i)
            for i, passenger in enumerate(extra)
        ]
        markets[16] = (base_passengers + extra, drivers)

    return markets


def run_ora(passengers, drivers, network):
    groups, _ = enumerate_groups(drivers, passengers, network)
    return clear_market(
        groups,
        {p.id: p for p in passengers},
        {d.id: d for d in drivers},
    )


def result_row(seed, n_passengers, arm, metrics, violation_rate):
    pdr = n_passengers / N_DRIVERS
    seat_pressure = n_passengers / (N_DRIVERS * CAPACITY)
    row = {
        "seed": int(seed),
        "n_passengers": int(n_passengers),
        "n_drivers": N_DRIVERS,
        "capacity": CAPACITY,
        "passenger_driver_ratio": float(pdr),
        "demand_seat_ratio": float(seat_pressure),
        "arm": arm,
        "r_max_violation_rate": float(violation_rate),
    }
    for field in METRIC_FIELDS:
        row[field] = float(getattr(metrics, field))
    return row


def summarize(raw):
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    rows = []
    keys = [
        "n_passengers",
        "n_drivers",
        "capacity",
        "passenger_driver_ratio",
        "demand_seat_ratio",
        "arm",
    ]
    for key_values, part in raw.groupby(keys, sort=True):
        base = dict(zip(keys, key_values))
        for field in fields:
            x = part[field].astype(float)
            rows.append(
                {
                    **base,
                    "metric": field,
                    "n_seeds": len(x),
                    "mean": x.mean(),
                    "std": x.std(ddof=1),
                    "ci95": 1.96 * x.std(ddof=1) / np.sqrt(len(x)),
                }
            )
    return pd.DataFrame(rows)


def factorial_effects(raw):
    """2x2 paired effects at each market-pressure level."""
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    rows = []

    for n_passengers, part in raw.groupby("n_passengers", sort=True):
        pdr = float(part["passenger_driver_ratio"].iloc[0])
        seat_pressure = float(part["demand_seat_ratio"].iloc[0])

        for field in fields:
            wide = part.pivot(index="seed", columns="arm", values=field)
            ora = wide[ARM_ORA]
            no_r = wide[ARM_NO_RADIUS]
            no_s = wide[ARM_NO_POOLING]
            full = wide[ARM_FULL]

            contrasts = {
                "radius_effect_with_individual_share": ora - no_r,
                "share_effect_with_individual_radius": ora - no_s,
                "radius_main_effect": 0.5 * ((ora - no_r) + (no_s - full)),
                "share_main_effect": 0.5 * ((ora - no_s) + (no_r - full)),
                "interaction": ora - no_r - no_s + full,
            }

            for effect_name, values in contrasts.items():
                rows.append(
                    {
                        "n_passengers": int(n_passengers),
                        "passenger_driver_ratio": pdr,
                        "demand_seat_ratio": seat_pressure,
                        "metric": field,
                        "effect": effect_name,
                        "n_seeds": len(values),
                        "mean": values.mean(),
                        "std": values.std(ddof=1),
                        "ci95": 1.96 * values.std(ddof=1) / np.sqrt(len(values)),
                    }
                )

    return pd.DataFrame(rows)


def pressure_trends(raw):
    """Within-seed slope of each factorial effect per +1 passenger/driver ratio."""
    effects = factorial_effects_seed_level(raw)
    rows = []

    for (metric, effect_name), part in effects.groupby(["metric", "effect"]):
        slopes = []
        for seed, seed_part in part.groupby("seed"):
            seed_part = seed_part.sort_values("passenger_driver_ratio")
            x = seed_part["passenger_driver_ratio"].to_numpy(dtype=float)
            y = seed_part["value"].to_numpy(dtype=float)
            slopes.append(float(np.polyfit(x, y, 1)[0]))

        slopes = np.asarray(slopes, dtype=float)
        rows.append(
            {
                "metric": metric,
                "effect": effect_name,
                "n_seeds": len(slopes),
                "reference_step_passengers_per_driver": 1.0,
                "effect_change_per_reference_step": slopes.mean(),
                "effect_change_std": slopes.std(ddof=1),
                "effect_change_ci95_per_reference_step": (
                    1.96 * slopes.std(ddof=1) / np.sqrt(len(slopes))
                ),
            }
        )

    return pd.DataFrame(rows)


def factorial_effects_seed_level(raw):
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    rows = []

    for n_passengers, part in raw.groupby("n_passengers", sort=True):
        pdr = float(part["passenger_driver_ratio"].iloc[0])
        for field in fields:
            wide = part.pivot(index="seed", columns="arm", values=field)
            ora = wide[ARM_ORA]
            no_r = wide[ARM_NO_RADIUS]
            no_s = wide[ARM_NO_POOLING]
            full = wide[ARM_FULL]
            contrasts = {
                "radius_main_effect": 0.5 * ((ora - no_r) + (no_s - full)),
                "share_main_effect": 0.5 * ((ora - no_s) + (no_r - full)),
                "interaction": ora - no_r - no_s + full,
            }
            for effect_name, values in contrasts.items():
                for seed, value in values.items():
                    rows.append(
                        {
                            "seed": int(seed),
                            "n_passengers": int(n_passengers),
                            "passenger_driver_ratio": pdr,
                            "metric": field,
                            "effect": effect_name,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def anchor_check(raw):
    """Check whether n=12 reproduces the existing factorial experiment."""
    if not FACTORIAL_ANCHOR.exists():
        print("\nAnchor check skipped: results/factorial_ablation_raw.csv not found.")
        return

    old = pd.read_csv(FACTORIAL_ANCHOR)
    current = raw[raw["n_passengers"] == BASE_PASSENGERS].copy()
    fields = METRIC_FIELDS + ["r_max_violation_rate"]

    old = old[["seed", "arm"] + fields].sort_values(["seed", "arm"]).reset_index(drop=True)
    current = current[["seed", "arm"] + fields].sort_values(["seed", "arm"]).reset_index(drop=True)

    if list(old["arm"].unique()) != list(current["arm"].unique()):
        # Order can differ, but the set must not.
        if set(old["arm"].unique()) != set(current["arm"].unique()):
            print("\nWARNING: factorial anchor has different arm names; check skipped.")
            return

    merged = old.merge(current, on=["seed", "arm"], suffixes=("_old", "_new"))
    max_abs = 0.0
    for field in fields:
        diff = (merged[f"{field}_old"] - merged[f"{field}_new"]).abs().max()
        max_abs = max(max_abs, float(diff))

    print(f"\nExperiment-A anchor check at n=12: max absolute difference = {max_abs:.3g}")
    if max_abs <= 1e-9:
        print("PASS: the 12-passenger pressure level reproduces the factorial ablation.")
    else:
        print("WARNING: n=12 does not exactly reproduce Experiment A; inspect code tree before interpretation.")


def run():
    network = grid_network(6, 6)
    rows = []

    print("=== Experiment C: market-pressure sweep ===")
    print("Passenger levels:", PASSENGER_LEVELS)
    print("Drivers:", N_DRIVERS, " Capacity:", CAPACITY)

    for seed in SEEDS:
        print(f"seed {seed}")
        markets = build_nested_markets(seed, network)

        for n_passengers in PASSENGER_LEVELS:
            passengers, drivers = markets[n_passengers]

            ora = run_ora(passengers, drivers, network)
            no_radius = run_radius_only_ablation(
                passengers, drivers, network, radius_km=RADIUS_KM
            )
            no_pooling = run_share_only_ablation(
                passengers, drivers, network, share_discount=SHARE_DISCOUNT
            )
            full = run_ablation(
                passengers,
                drivers,
                network,
                radius_km=RADIUS_KM,
                share_discount=SHARE_DISCOUNT,
            )

            for arm, result in (
                (ARM_ORA, ora),
                (ARM_NO_RADIUS, no_radius),
                (ARM_NO_POOLING, no_pooling),
                (ARM_FULL, full),
            ):
                metrics = compute_metrics(
                    result.allocation, passengers, drivers, network
                )
                violation = radius_violations(
                    result.allocation, passengers
                )[1]
                rows.append(
                    result_row(
                        seed,
                        n_passengers,
                        arm,
                        metrics,
                        violation,
                    )
                )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    effects = factorial_effects(raw)
    trends = pressure_trends(raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    effects.to_csv(EFFECTS_OUTPUT, index=False)
    trends.to_csv(TRENDS_OUTPUT, index=False)

    print("\nSaved:")
    for output in (RAW_OUTPUT, SUMMARY_OUTPUT, EFFECTS_OUTPUT, TRENDS_OUTPUT):
        print(" ", output)

    anchor_check(raw)

    print("\n=== Headline factorial effects by market pressure ===")
    selected = effects[
        effects["metric"].isin(HEADLINE)
        & effects["effect"].isin(
            ["radius_main_effect", "share_main_effect", "interaction"]
        )
    ]
    for effect_name in ("radius_main_effect", "share_main_effect", "interaction"):
        print(f"\n{effect_name}")
        table = (
            selected[selected["effect"] == effect_name]
            .pivot(index="metric", columns="passenger_driver_ratio", values="mean")
            .reindex(HEADLINE)
        )
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Within-seed trend per +1 passenger/driver ===")
    print(
        trends[
            trends["metric"].isin(HEADLINE)
            & trends["effect"].isin(["radius_main_effect", "share_main_effect"])
        ][
            [
                "metric",
                "effect",
                "effect_change_per_reference_step",
                "effect_change_ci95_per_reference_step",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run()
