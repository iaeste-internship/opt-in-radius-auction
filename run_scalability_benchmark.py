"""Empirical scalability benchmark for the Opt-In Radius Auction (ORA).

The goal is not to change the mechanism or compare economic outcomes. It is to
measure where exact ORA spends computational time as market size increases.

For every (market size, seed) pair the script records:
  1. feasible-hyperedge enumeration time;
  2. number of candidate subsets and feasible groups;
  3. one exact winner-determination (WDP) solve time;
  4. full market-clearing time, including Clarke exclusion solves;
  5. total end-to-end time = enumeration + full market clearing.

The benchmark keeps passenger/driver pressure fixed at 3 passengers per driver,
uses capacity 3, and uses a fixed 10 x 10 Manhattan grid. The larger grid is
intentional for this computational benchmark: it avoids turning the experiment
into a density/saturation experiment as n grows.

The script saves after every completed run and can resume safely. If interrupted,
the completed rows remain on disk and partial summary files are rebuilt.

Run from the project root:

    python run_scalability_benchmark.py

Quick smoke test:

    python run_scalability_benchmark.py --sizes 12 24 --n-seeds 2 --fresh

Default command-line subset:

    sizes = 12, 24, 36, 48, 60 passengers
    passenger/driver ratio = 3
    seeds = 0, 1, 2, 3, 4
    capacity = 3
    network = 10 x 10 grid

Completed report benchmark:

    python run_scalability_benchmark.py --sizes 12 24 36 48 60 72 84 96 120 --n-seeds 5

Outputs:
    results/scalability_raw.csv
    results/scalability_summary.csv
    results/scalability_scaling_fits.csv
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.network import grid_network
from ora.payments import clear_market
from ora.wdp import solve_wdp


RESULTS_DIR = Path("results")
RAW_FILE = RESULTS_DIR / "scalability_raw.csv"
SUMMARY_FILE = RESULTS_DIR / "scalability_summary.csv"
FITS_FILE = RESULTS_DIR / "scalability_scaling_fits.csv"

DEFAULT_SIZES = [12, 24, 36, 48, 60]
DEFAULT_N_SEEDS = 5
PASSENGERS_PER_DRIVER = 3
CAPACITY = 3
GRID_ROWS = 10
GRID_COLS = 10

TIMING_FIELDS = [
    "enumeration_time_s",
    "main_wdp_time_s",
    "full_clear_time_s",
    "payment_overhead_s",
    "total_time_s",
]

COUNT_FIELDS = [
    "candidates_considered",
    "groups_generated",
    "rejected_by_routing",
    "n_clarke_solves",
    "matched_passengers",
    "allocated_drivers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical ORA scalability/runtime benchmark."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Passenger counts to benchmark (default subset: 12 24 36 48 60).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of seeds starting from 0 (default: 5).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing scalability outputs before running.",
    )
    return parser.parse_args()


def _ci95(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def _iqr(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if values.empty:
        return float("nan")
    return float(values.quantile(0.75) - values.quantile(0.25))


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runtime/count metrics by passenger count.

    Runtime distributions can be skewed, so both mean +/- 95% CI and
    median/IQR are reported. The paper can use the median/IQR as the headline
    runtime and retain mean/CI for consistency with the other experiments.
    """
    rows: list[dict] = []

    for n_passengers, group in raw.groupby("n_passengers", sort=True):
        row: dict[str, float | int] = {
            "n_passengers": int(n_passengers),
            "n_drivers": int(group["n_drivers"].iloc[0]),
            "passenger_driver_ratio": float(
                group["passenger_driver_ratio"].iloc[0]
            ),
            "n_seeds": int(group["seed"].nunique()),
        }

        for field in TIMING_FIELDS + COUNT_FIELDS:
            values = group[field].astype(float)
            row[f"{field}_mean"] = float(values.mean())
            row[f"{field}_ci95"] = _ci95(values)
            row[f"{field}_median"] = float(values.median())
            row[f"{field}_iqr"] = _iqr(values)

        # Useful decomposition shares.
        total = group["total_time_s"].astype(float)
        enum = group["enumeration_time_s"].astype(float)
        clear = group["full_clear_time_s"].astype(float)
        row["enumeration_share_of_total_mean"] = float(
            (enum / total.replace(0.0, np.nan)).mean()
        )
        row["clearing_share_of_total_mean"] = float(
            (clear / total.replace(0.0, np.nan)).mean()
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_passengers").reset_index(drop=True)


def build_scaling_fits(summary: pd.DataFrame) -> pd.DataFrame:
    """Estimate empirical log-log scaling exponents from median values.

    A fit y = a * n^b becomes log(y) = log(a) + b*log(n). The exponent b is
    descriptive, not a complexity proof. It is useful for reporting the
    observed growth rate over the measured range.
    """
    targets = {
        "groups_generated": "groups_generated_median",
        "enumeration_time_s": "enumeration_time_s_median",
        "main_wdp_time_s": "main_wdp_time_s_median",
        "full_clear_time_s": "full_clear_time_s_median",
        "total_time_s": "total_time_s_median",
    }

    x_all = summary["n_passengers"].to_numpy(dtype=float)
    rows: list[dict] = []

    for label, column in targets.items():
        y_all = summary[column].to_numpy(dtype=float)
        mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
        x = x_all[mask]
        y = y_all[mask]

        if len(x) < 2:
            rows.append(
                {
                    "quantity": label,
                    "n_levels": len(x),
                    "scaling_exponent": np.nan,
                    "coefficient": np.nan,
                    "r_squared": np.nan,
                }
            )
            continue

        log_x = np.log(x)
        log_y = np.log(y)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        predicted = intercept + slope * log_x
        ss_res = float(np.sum((log_y - predicted) ** 2))
        ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        rows.append(
            {
                "quantity": label,
                "n_levels": len(x),
                "scaling_exponent": float(slope),
                "coefficient": float(math.exp(intercept)),
                "r_squared": float(r_squared),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = raw.sort_values(["n_passengers", "seed"]).reset_index(drop=True)
    raw.to_csv(RAW_FILE, index=False)

    summary = build_summary(raw)
    summary.to_csv(SUMMARY_FILE, index=False)

    fits = build_scaling_fits(summary)
    fits.to_csv(FITS_FILE, index=False)
    return summary, fits


def validate_existing(raw: pd.DataFrame, sizes: list[int]) -> None:
    """Prevent accidentally mixing results from incompatible settings."""
    if raw.empty:
        return

    expected = {
        "passengers_per_driver": PASSENGERS_PER_DRIVER,
        "capacity": CAPACITY,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
    }
    for column, value in expected.items():
        if column not in raw.columns:
            raise ValueError(
                f"Existing {RAW_FILE} predates this benchmark format. "
                "Use --fresh or rename the old file."
            )
        observed = set(raw[column].dropna().unique())
        if observed != {value}:
            raise ValueError(
                f"Existing {RAW_FILE} uses {column}={sorted(observed)}, "
                f"but this run expects {value}. Use --fresh or rename it."
            )

    # Existing sizes outside the current command are fine: resume mode may be
    # used to extend the benchmark. We only guard settings, not the size list.
    _ = sizes


def run_one(n_passengers: int, seed: int) -> dict:
    if n_passengers % PASSENGERS_PER_DRIVER != 0:
        raise ValueError(
            f"n_passengers={n_passengers} is not divisible by "
            f"{PASSENGERS_PER_DRIVER}; choose sizes such as 12, 24, 36, 48, 60."
        )

    n_drivers = n_passengers // PASSENGERS_PER_DRIVER
    network = grid_network(GRID_ROWS, GRID_COLS)
    rng = np.random.default_rng(seed)
    passengers, drivers = generate(
        DemandParams(
            n_passengers=n_passengers,
            n_drivers=n_drivers,
            capacity=CAPACITY,
        ),
        network,
        rng,
    )

    passenger_lookup = {p.id: p for p in passengers}
    driver_lookup = {d.id: d for d in drivers}

    # 1) Exhaustive feasible-group construction.
    t0 = time.perf_counter()
    groups, grouping_stats = enumerate_groups(drivers, passengers, network)
    enumeration_time = time.perf_counter() - t0

    # 2) One exact allocation solve without pricing counterfactuals.
    t0 = time.perf_counter()
    winners, welfare = solve_wdp(groups, passenger_lookup, driver_lookup)
    main_wdp_time = time.perf_counter() - t0

    # 3) Complete ORA clearing, including a fresh main solve plus the Clarke
    #    passenger/driver exclusion solves. We deliberately time the public
    #    clear_market API rather than duplicating payment internals here.
    t0 = time.perf_counter()
    clearing = clear_market(groups, passenger_lookup, driver_lookup)
    full_clear_time = time.perf_counter() - t0

    matched_passengers = len(
        {
            pid
            for group in clearing.allocation.groups
            for pid in group.passenger_ids
        }
    )
    allocated_drivers = len(
        {group.driver_id for group in clearing.allocation.groups}
    )

    # clear_market includes its own main WDP solve, so subtracting the separately
    # timed main solve is an approximate decomposition of payment/counterfactual
    # overhead. Both timings use the same fixed group list and solver settings.
    payment_overhead = max(0.0, full_clear_time - main_wdp_time)

    return {
        "seed": seed,
        "n_passengers": n_passengers,
        "n_drivers": n_drivers,
        "passenger_driver_ratio": n_passengers / n_drivers,
        "passengers_per_driver": PASSENGERS_PER_DRIVER,
        "capacity": CAPACITY,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "candidates_considered": grouping_stats.candidates_considered,
        "groups_generated": len(groups),
        "rejected_by_routing": grouping_stats.rejected_by_routing,
        "enumeration_time_s": enumeration_time,
        "main_wdp_time_s": main_wdp_time,
        "full_clear_time_s": full_clear_time,
        "payment_overhead_s": payment_overhead,
        "total_time_s": enumeration_time + full_clear_time,
        "n_clarke_solves": clearing.n_exclusion_solves,
        "matched_passengers": matched_passengers,
        "allocated_drivers": allocated_drivers,
        "main_wdp_welfare": welfare,
        "clearing_welfare": clearing.welfare,
        "winner_count": len(winners),
    }


def print_report(summary: pd.DataFrame, fits: pd.DataFrame) -> None:
    if summary.empty:
        return

    display = summary[
        [
            "n_passengers",
            "n_drivers",
            "groups_generated_median",
            "enumeration_time_s_median",
            "main_wdp_time_s_median",
            "full_clear_time_s_median",
            "total_time_s_median",
            "n_clarke_solves_median",
        ]
    ].copy()

    print("\n=== Median scalability profile ===")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Empirical log-log scaling fits (descriptive, not complexity proofs) ===")
    print(
        fits.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan",
        )
    )

    largest = summary.sort_values("n_passengers").iloc[-1]
    enum_share = 100.0 * largest["enumeration_share_of_total_mean"]
    clear_share = 100.0 * largest["clearing_share_of_total_mean"]
    print(
        "\nLargest completed market: "
        f"n={int(largest['n_passengers'])}, "
        f"median total={largest['total_time_s_median']:.3f}s, "
        f"mean time shares: enumeration={enum_share:.1f}%, "
        f"clearing={clear_share:.1f}%."
    )


def main() -> None:
    args = parse_args()
    sizes = sorted(set(args.sizes))
    seeds = list(range(args.n_seeds))

    if args.fresh:
        for path in (RAW_FILE, SUMMARY_FILE, FITS_FILE):
            if path.exists():
                path.unlink()

    if RAW_FILE.exists():
        raw = pd.read_csv(RAW_FILE)
        validate_existing(raw, sizes)
    else:
        raw = pd.DataFrame()

    completed: set[tuple[int, int]] = set()
    if not raw.empty:
        completed = {
            (int(row.n_passengers), int(row.seed))
            for row in raw[["n_passengers", "seed"]].itertuples(index=False)
        }

    print("ORA scalability benchmark")
    print(f"  sizes: {sizes}")
    print(f"  seeds: {seeds}")
    print(f"  passenger/driver ratio: {PASSENGERS_PER_DRIVER}:1")
    print(f"  capacity: {CAPACITY}")
    print(f"  network: {GRID_ROWS} x {GRID_COLS} Manhattan grid")
    print("  outputs are saved after every completed run")

    try:
        for n_passengers in sizes:
            for seed in seeds:
                key = (n_passengers, seed)
                if key in completed:
                    print(f"SKIP n={n_passengers:3d} seed={seed}: already complete")
                    continue

                print(f"RUN  n={n_passengers:3d} seed={seed} ...", flush=True)
                row = run_one(n_passengers, seed)

                raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
                summary, fits = save_outputs(raw)
                completed.add(key)

                print(
                    "     "
                    f"groups={int(row['groups_generated']):6d}  "
                    f"enum={row['enumeration_time_s']:8.3f}s  "
                    f"wdp={row['main_wdp_time_s']:7.3f}s  "
                    f"clear={row['full_clear_time_s']:8.3f}s  "
                    f"total={row['total_time_s']:8.3f}s  "
                    f"solves={int(row['n_clarke_solves']):3d}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed rows have already been saved.")
        if not raw.empty:
            summary, fits = save_outputs(raw)
            print_report(summary, fits)
        print("Re-run the same command to resume missing (size, seed) pairs.")
        return

    summary, fits = save_outputs(raw)

    print("\nSaved:")
    print(f"  {RAW_FILE}")
    print(f"  {SUMMARY_FILE}")
    print(f"  {FITS_FILE}")
    print_report(summary, fits)


if __name__ == "__main__":
    main()
