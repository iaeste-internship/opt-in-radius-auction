"""2x2 factorial ablation of ORA's two elicited passenger components.

Factors
-------
Pickup radius:
    individual r_max (elicited) vs one global radius = 2.0 km
Pooling value:
    individual v_share (elicited) vs v_solo * 0.75 for everyone

The four cells are:
    ORA                 individual radius, individual share value
    No radius elic.     global radius,     individual share value
    No pooling elic.    individual radius, global share discount
    Full ablation       global radius,     global share discount

Every arm uses the same market instance for a seed, the same exact WDP and the
same Clarke-payment code. Allocations produced under assumed types are measured
against the true passenger population.
"""

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
PARAMS = DemandParams(n_passengers=12, n_drivers=4, capacity=3)
RADIUS_KM = DEFAULT_RADIUS_KM
SHARE_DISCOUNT = DEFAULT_SHARE_DISCOUNT

RESULTS_DIR = Path("results")
RAW_OUTPUT = RESULTS_DIR / "factorial_ablation_raw.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "factorial_ablation_summary.csv"
EFFECTS_OUTPUT = RESULTS_DIR / "factorial_ablation_effects.csv"

ARM_ORA = "ORA"
ARM_NO_RADIUS = "No radius elicitation"
ARM_NO_POOLING = "No pooling elicitation"
ARM_FULL = "Full ablation"

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


def result_row(seed, arm, metrics, violation_rate):
    row = {
        "seed": seed,
        "arm": arm,
        "radius_setting": (
            "individual"
            if arm in {ARM_ORA, ARM_NO_POOLING}
            else f"global_{RADIUS_KM:.2f}km"
        ),
        "share_setting": (
            "individual"
            if arm in {ARM_ORA, ARM_NO_RADIUS}
            else f"global_{SHARE_DISCOUNT:.2f}"
        ),
        "r_max_violation_rate": violation_rate,
    }
    for field in METRIC_FIELDS:
        row[field] = getattr(metrics, field)
    return row


def summarize(raw):
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    rows = []
    for arm, part in raw.groupby("arm", sort=False):
        for field in fields:
            x = part[field].astype(float)
            rows.append({
                "arm": arm,
                "metric": field,
                "n_seeds": len(x),
                "mean": x.mean(),
                "std": x.std(ddof=1),
                "ci95": 1.96 * x.std(ddof=1) / np.sqrt(len(x)),
            })
    return pd.DataFrame(rows)


def paired_effects(raw):
    """Paired 2x2 contrasts, computed seed by seed before averaging.

    Positive 'radius_main_effect' means moving from a global radius to elicited
    individual radii increases the metric, averaging over the two share-value
    settings. Positive 'share_main_effect' means moving from a global share
    discount to individual share values increases the metric.
    """
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    rows = []

    for field in fields:
        wide = raw.pivot(index="seed", columns="arm", values=field)
        ora = wide[ARM_ORA]
        no_r = wide[ARM_NO_RADIUS]
        no_s = wide[ARM_NO_POOLING]
        full = wide[ARM_FULL]

        contrasts = {
            # Simple effects directly attached to ORA.
            "radius_effect_with_individual_share": ora - no_r,
            "share_effect_with_individual_radius": ora - no_s,
            # Factorial main effects, averaged across the other factor.
            "radius_main_effect": 0.5 * ((ora - no_r) + (no_s - full)),
            "share_main_effect": 0.5 * ((ora - no_s) + (no_r - full)),
            # Difference-in-differences: whether the two elicitation gains
            # reinforce or offset each other.
            "interaction": ora - no_r - no_s + full,
        }

        for name, values in contrasts.items():
            rows.append({
                "metric": field,
                "effect": name,
                "n_seeds": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "ci95": 1.96 * values.std(ddof=1) / np.sqrt(len(values)),
            })

    return pd.DataFrame(rows)


def run():
    network = grid_network(6, 6)
    rows = []

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        rng = np.random.default_rng(seed)
        passengers, drivers = generate(PARAMS, network, rng)

        passenger_lookup = {p.id: p for p in passengers}
        driver_lookup = {d.id: d for d in drivers}

        # 1) ORA: individual radius + individual share value.
        groups, _ = enumerate_groups(drivers, passengers, network)
        ora = clear_market(groups, passenger_lookup, driver_lookup)

        # 2) Remove only radius elicitation.
        no_radius = run_radius_only_ablation(
            passengers, drivers, network, radius_km=RADIUS_KM
        )

        # 3) Remove only pooling-value elicitation.
        no_pooling = run_share_only_ablation(
            passengers, drivers, network, share_discount=SHARE_DISCOUNT
        )

        # 4) Existing full ablation: remove both at once.
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
            violation_rate = radius_violations(
                result.allocation, passengers
            )[1]
            rows.append(result_row(seed, arm, metrics, violation_rate))

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    effects = paired_effects(raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    effects.to_csv(EFFECTS_OUTPUT, index=False)

    print(f"Saved: {RAW_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")
    print(f"Saved: {EFFECTS_OUTPUT}")

    headline = [
        "realised_welfare",
        "match_rate",
        "pooling_rate",
        "r_max_violation_rate",
        "passenger_ir_violation_rate",
        "mean_allocated_passenger_utility",
        "mean_adjusted_passenger_utility",
        "total_acceptability_cost",
    ]

    table = (
        summary[summary["metric"].isin(headline)]
        .pivot(index="metric", columns="arm", values="mean")
        .reindex(headline)
    )
    print("\nHeadline means:\n")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    effect_table = effects[
        effects["metric"].isin(headline)
        & effects["effect"].isin(
            ["radius_main_effect", "share_main_effect", "interaction"]
        )
    ].pivot(index="metric", columns="effect", values="mean").reindex(headline)

    print("\nPaired factorial effects (individualized - global):\n")
    print(effect_table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    run()
