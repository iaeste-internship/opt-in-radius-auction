"""Sensitivity experiment: does the value of elicitation grow with heterogeneity?

This script extends the 2x2 factorial ablation with two controlled one-factor
heterogeneity sweeps. It DOES NOT overwrite the existing radius-sweep or
factorial-ablation results.

Scientific design
-----------------
1) Radius heterogeneity sweep
   - Vary r_max_sd while keeping r_max_mean = 1.5 km.
   - Compare ORA against "No radius elicitation" only.
   - The comparator uses a single global radius equal to the REALIZED mean r_max
     in that seed and heterogeneity level, not the 2.0 km literature benchmark.
     This deliberately gives the global-radius comparator the correct market
     mean, so any divergence is driven by dispersion rather than a mean mismatch.

2) Pooling-preference heterogeneity sweep
   - Vary tau_sd while keeping tau_mean = 0.75.
   - Compare ORA against "No pooling elicitation" only.
   - The comparator uses one common share discount equal to the REALIZED mean tau
     in that seed and heterogeneity level. This controls for the slight shift in
     the realized mean caused by clipping tau to [0.1, 1.0]. When tau_sd=0, the
     mechanisms must coincide exactly.

The same 50 seeds are reused at every heterogeneity level. Because DemandParams
changes only the scale of already-drawn normal shocks, the geography and all
other random draws stay paired across levels for a given seed.

Outputs
-------
results/heterogeneity_mean_matched_raw.csv
results/heterogeneity_mean_matched_summary.csv
results/heterogeneity_mean_matched_effects.csv
results/heterogeneity_mean_matched_trends.csv

Effects are always coded as ORA - comparison arm. Rate effects are stored as
fractions in the CSV; multiply by 100 for percentage points in the paper.
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
    radius_violations,
    run_radius_only_ablation,
    run_share_only_ablation,
)


SEEDS = range(50)

# Baseline population used throughout the existing paper.
N_PASSENGERS = 12
N_DRIVERS = 4
CAPACITY = 3
R_MAX_MEAN = 1.5
TAU_MEAN = 0.75
BASE_R_MAX_SD = 0.5
BASE_TAU_SD = 0.15

# 0 = homogeneous control, then half/base/1.5x the current dispersion.
RADIUS_SD_LEVELS = (0.0, 0.25, 0.50, 0.75)
TAU_SD_LEVELS = (0.0, 0.075, 0.150, 0.225)

RESULTS_DIR = Path("results")
RAW_OUTPUT = RESULTS_DIR / "heterogeneity_mean_matched_raw.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "heterogeneity_mean_matched_summary.csv"
EFFECTS_OUTPUT = RESULTS_DIR / "heterogeneity_mean_matched_effects.csv"
TRENDS_OUTPUT = RESULTS_DIR / "heterogeneity_mean_matched_trends.csv"

ARM_ORA = "ORA"
ARM_NO_RADIUS = "No radius elicitation"
ARM_NO_POOLING = "No pooling elicitation"

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
    "r_max_violation_rate",
    "passenger_ir_violation_rate",
    "mean_allocated_passenger_utility",
    "mean_adjusted_passenger_utility",
    "total_acceptability_cost",
]


def _market_stats(passengers):
    """Observed population moments after clipping/flooring in demand.generate."""
    r = np.asarray([p.r_max for p in passengers], dtype=float)
    tau = np.asarray([p.tau for p in passengers], dtype=float)
    return {
        "realized_r_max_mean": float(r.mean()),
        "realized_r_max_sd": float(r.std(ddof=0)),
        "realized_tau_mean": float(tau.mean()),
        "realized_tau_sd": float(tau.std(ddof=0)),
    }


def _row(
    *,
    experiment,
    level,
    seed,
    arm,
    metrics,
    violation_rate,
    market_stats,
):
    row = {
        "experiment": experiment,
        "heterogeneity_level": float(level),
        "seed": int(seed),
        "arm": arm,
        "r_max_violation_rate": float(violation_rate),
        **market_stats,
    }
    for field in METRIC_FIELDS:
        row[field] = float(getattr(metrics, field))
    return row


def _run_ora(passengers, drivers, network):
    groups, _ = enumerate_groups(drivers, passengers, network)
    return clear_market(
        groups,
        {p.id: p for p in passengers},
        {d.id: d for d in drivers},
    )


def _measure(result, passengers, drivers, network):
    metrics = compute_metrics(result.allocation, passengers, drivers, network)
    violation_rate = radius_violations(result.allocation, passengers)[1]
    return metrics, violation_rate


def run_radius_sweep(network):
    rows = []
    print("\n=== Radius heterogeneity sweep ===")

    for level in RADIUS_SD_LEVELS:
        params = DemandParams(
            n_passengers=N_PASSENGERS,
            n_drivers=N_DRIVERS,
            capacity=CAPACITY,
            r_max_mean=R_MAX_MEAN,
            r_max_sd=level,
            tau_mean=TAU_MEAN,
            tau_sd=BASE_TAU_SD,
        )
        print(f"r_max_sd={level:.3f}")

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            passengers, drivers = generate(params, network, rng)
            stats = _market_stats(passengers)

            ora = _run_ora(passengers, drivers, network)
            no_radius = run_radius_only_ablation(
                passengers,
                drivers,
                network,
                radius_km=stats["realized_r_max_mean"],
            )

            for arm, result in ((ARM_ORA, ora), (ARM_NO_RADIUS, no_radius)):
                metrics, violation = _measure(
                    result, passengers, drivers, network
                )
                rows.append(
                    _row(
                        experiment="radius",
                        level=level,
                        seed=seed,
                        arm=arm,
                        metrics=metrics,
                        violation_rate=violation,
                        market_stats=stats,
                    )
                )

    return rows


def run_share_sweep(network):
    rows = []
    print("\n=== Pooling-preference heterogeneity sweep ===")

    for level in TAU_SD_LEVELS:
        params = DemandParams(
            n_passengers=N_PASSENGERS,
            n_drivers=N_DRIVERS,
            capacity=CAPACITY,
            r_max_mean=R_MAX_MEAN,
            r_max_sd=BASE_R_MAX_SD,
            tau_mean=TAU_MEAN,
            tau_sd=level,
        )
        print(f"tau_sd={level:.3f}")

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            passengers, drivers = generate(params, network, rng)
            stats = _market_stats(passengers)

            ora = _run_ora(passengers, drivers, network)
            no_pooling = run_share_only_ablation(
                passengers,
                drivers,
                network,
                share_discount=stats["realized_tau_mean"],
            )

            for arm, result in ((ARM_ORA, ora), (ARM_NO_POOLING, no_pooling)):
                metrics, violation = _measure(
                    result, passengers, drivers, network
                )
                rows.append(
                    _row(
                        experiment="share",
                        level=level,
                        seed=seed,
                        arm=arm,
                        metrics=metrics,
                        violation_rate=violation,
                        market_stats=stats,
                    )
                )

    return rows


def build_summary(raw):
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    rows = []

    for (experiment, level, arm), part in raw.groupby(
        ["experiment", "heterogeneity_level", "arm"], sort=False
    ):
        for field in fields:
            x = part[field].astype(float)
            rows.append(
                {
                    "experiment": experiment,
                    "heterogeneity_level": level,
                    "arm": arm,
                    "metric": field,
                    "n_seeds": len(x),
                    "mean": x.mean(),
                    "std": x.std(ddof=1),
                    "ci95": 1.96 * x.std(ddof=1) / np.sqrt(len(x)),
                    "realized_r_max_mean": part["realized_r_max_mean"].mean(),
                    "realized_r_max_sd": part["realized_r_max_sd"].mean(),
                    "realized_tau_mean": part["realized_tau_mean"].mean(),
                    "realized_tau_sd": part["realized_tau_sd"].mean(),
                }
            )

    return pd.DataFrame(rows)


def build_effects(raw):
    """Paired ORA-minus-ablation effects at each heterogeneity level."""
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    rows = []

    comparator = {
        "radius": ARM_NO_RADIUS,
        "share": ARM_NO_POOLING,
    }

    for experiment in ("radius", "share"):
        exp = raw[raw["experiment"] == experiment]
        for level in sorted(exp["heterogeneity_level"].unique()):
            part = exp[np.isclose(exp["heterogeneity_level"], level)]
            for field in fields:
                wide = part.pivot(index="seed", columns="arm", values=field)
                delta = wide[ARM_ORA] - wide[comparator[experiment]]
                rows.append(
                    {
                        "experiment": experiment,
                        "heterogeneity_level": float(level),
                        "metric": field,
                        "effect": "ORA_minus_comparator",
                        "n_seeds": len(delta),
                        "mean": delta.mean(),
                        "std": delta.std(ddof=1),
                        "ci95": 1.96 * delta.std(ddof=1) / np.sqrt(len(delta)),
                    }
                )

    return pd.DataFrame(rows)


def build_trends(effects):
    """Summarize whether the paired effect grows with heterogeneity.

    A separate linear slope is fitted within each seed conceptually by using the
    raw paired effects across levels. Because `effects` is already aggregated,
    this function is populated later from raw data by `build_seed_trends`.
    """
    return effects


def build_seed_trends(raw):
    """Mean within-seed slope of the paired effect vs heterogeneity level.

    This avoids treating the repeated heterogeneity levels from the same seed as
    independent observations. One slope is estimated per seed, then the 50
    slopes are summarized with a 95% CI.
    """
    fields = METRIC_FIELDS + ["r_max_violation_rate"]
    comparator = {
        "radius": ARM_NO_RADIUS,
        "share": ARM_NO_POOLING,
    }
    reference_step = {"radius": 0.25, "share": 0.075}
    rows = []

    for experiment in ("radius", "share"):
        exp = raw[raw["experiment"] == experiment]

        for field in fields:
            seed_slopes = []
            for seed, seed_part in exp.groupby("seed"):
                wide = seed_part.pivot(
                    index="heterogeneity_level", columns="arm", values=field
                ).sort_index()
                delta = wide[ARM_ORA] - wide[comparator[experiment]]
                x = delta.index.to_numpy(dtype=float)
                y = delta.to_numpy(dtype=float)

                if len(x) < 2 or np.allclose(x, x[0]):
                    continue
                slope = float(np.polyfit(x, y, deg=1)[0])
                seed_slopes.append(slope)

            s = pd.Series(seed_slopes, dtype=float)
            mean_slope = s.mean()
            ci95 = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
            step = reference_step[experiment]

            rows.append(
                {
                    "experiment": experiment,
                    "metric": field,
                    "n_seeds": len(s),
                    "slope_per_1unit_heterogeneity": mean_slope,
                    "slope_ci95": ci95,
                    "reference_step": step,
                    "effect_change_per_reference_step": mean_slope * step,
                    "effect_change_ci95_per_reference_step": ci95 * step,
                }
            )

    return pd.DataFrame(rows)


def _print_headline(effects, trends):
    for experiment in ("radius", "share"):
        print(f"\n=== {experiment.upper()} paired effects: ORA - comparator ===\n")
        table = effects[
            (effects["experiment"] == experiment)
            & (effects["metric"].isin(HEADLINE))
        ].pivot(index="metric", columns="heterogeneity_level", values="mean")
        print(table.reindex(HEADLINE).to_string(float_format=lambda x: f"{x:.4f}"))

        print(f"\n=== {experiment.upper()} within-seed trend per reference step ===\n")
        t = trends[
            (trends["experiment"] == experiment)
            & (trends["metric"].isin(HEADLINE))
        ].copy()
        t = t[
            [
                "metric",
                "reference_step",
                "effect_change_per_reference_step",
                "effect_change_ci95_per_reference_step",
            ]
        ]
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run():
    network = grid_network(6, 6)

    rows = []
    rows.extend(run_radius_sweep(network))
    rows.extend(run_share_sweep(network))

    raw = pd.DataFrame(rows)
    summary = build_summary(raw)
    effects = build_effects(raw)
    trends = build_seed_trends(raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    effects.to_csv(EFFECTS_OUTPUT, index=False)
    trends.to_csv(TRENDS_OUTPUT, index=False)

    print("\nSaved:")
    print(f"  {RAW_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")
    print(f"  {EFFECTS_OUTPUT}")
    print(f"  {TRENDS_OUTPUT}")

    # Strong internal validity check: at zero heterogeneity, the corresponding
    # global assumption exactly matches every passenger's true type.
    zero = effects[np.isclose(effects["heterogeneity_level"], 0.0)]
    max_abs_zero = float(zero["mean"].abs().max()) if not zero.empty else np.nan
    print(
        "\nZero-heterogeneity sanity check: "
        f"max absolute paired mean effect = {max_abs_zero:.12g}"
    )
    if not np.isclose(max_abs_zero, 0.0, atol=1e-10):
        print(
            "WARNING: zero-heterogeneity arms did not coincide exactly. "
            "Inspect the raw output before interpreting the sweep."
        )
    else:
        print("PASS: ORA and the mean-matched comparator coincide at zero heterogeneity.")

    # Report the realized moments so the paper can state what was actually
    # varied after clipping/flooring in the demand generator.
    print("\n=== Realized passenger-type moments ===\n")
    moments = (
        summary[[
            "experiment", "heterogeneity_level",
            "realized_r_max_mean", "realized_r_max_sd",
            "realized_tau_mean", "realized_tau_sd",
        ]]
        .drop_duplicates()
        .sort_values(["experiment", "heterogeneity_level"])
    )
    print(moments.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    _print_headline(effects, trends)


if __name__ == "__main__":
    run()
