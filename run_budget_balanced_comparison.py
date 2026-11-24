"""Experiment E: what does budget balance cost?

Reviewer question
-----------------
ORA runs a platform deficit in almost every market. Who pays for it, and how
does ORA compare with a mechanism that does not need a subsidy?

Design
------
Same nested markets as Experiment C (4 drivers x 3 seats; 8, 12 and 16
passengers; 50 paired seeds; 6 x 6 grid). Three arms on every market:

  ORA  exact WDP + bilateral Clarke payments (efficient, DSIC, IR, deficit)
  TB   ORA's allocation re-priced with the Parkes-Kalagnanam-Eso threshold
       rule (efficient, IR, budget balanced, not DSIC)
  PP   posted prices with ORA's bound-respecting feasibility (IR, budget
       balanced, not efficient)

PP needs a price menu. A grid of solo fares and driver participation
payments is evaluated on every market; the menu reported in the paper is the
one with the highest mean true welfare at each pressure level. Choosing it on
the same seeds it is evaluated on favours PP, so PP's welfare is an optimistic
upper bound for a posted-price platform.

Incentives are measured, not assumed. For every agent allocated under truthful
reporting, a grid of unilateral misreports is tried with all other reports
held at the truth, and the agent's TRUE utility is recorded:

  passengers: both valuations shaded down by x, and declining the shared mode
  drivers:    reservation value inflated by x

with x in MISREPORT_GRID. ORA should show no profitable misreport (a numerical
check of Propositions 1 and 2 on these value misreports); TB and PP are
expected to show some.

Outputs (results/)
------------------
budget_balanced_raw.csv           one row per seed x pressure x arm
budget_balanced_pp_grid.csv       PP outcomes for every menu
budget_balanced_summary.csv       mean, sd, 95% CI per pressure x arm x metric
budget_balanced_manipulation.csv  one row per agent x arm (best misreport)
budget_balanced_manip_summary.csv aggregated manipulation statistics

Run from the repository root:  python run_budget_balanced_comparison.py
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

from ora.budget_balanced import (
    PostedPrices,
    posted_price_clear,
    threshold_rule,
)
from ora.grouping import enumerate_groups
from ora.metrics import compute_metrics
from ora.network import grid_network
from ora.payments import budget_balance, clear_market
from ora.types import Mode

from run_market_pressure_sweep import build_nested_markets

SEEDS = range(int(os.environ.get("ORA_N_SEEDS", 50)))
PRESSURES = (8, 12, 16)
N_DRIVERS = 4
GRID_ROWS = GRID_COLS = 6

SHARE_RATIO = 0.75          # common shared-fare ratio, as in the ablations
PER_KM_PAY = 0.8            # equals the observable operating cost
FARE_GRID = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
BASE_PAY_GRID = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0)

MISREPORT_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)

RESULTS = Path("results")

METRIC_FIELDS = (
    "realised_welfare",
    "match_rate",
    "pooling_rate",
    "mean_wait_min",
    "mean_wait_all_min",
    "idle_ratio",
    "gini_wait_all",
    "jain_wait_all",
    "gini_earnings",
    "jain_earnings",
    "driver_ir_violation_rate",
    "mean_allocated_driver_utility",
    "passenger_ir_violation_rate",
    "mean_allocated_passenger_utility",
    "total_payments",
    "total_driver_earnings",
    "budget_balance",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _row(seed, n_pax, arm, allocation, passengers, drivers, network,
         extra=None):
    m = compute_metrics(allocation, passengers, drivers, network)
    matched = m.matched_passengers
    subsidy = max(0.0, -m.budget_balance)
    row = {
        "seed": int(seed),
        "n_passengers": int(n_pax),
        "passenger_driver_ratio": n_pax / N_DRIVERS,
        "arm": arm,
        "subsidy": subsidy,
        "deficit": float(m.budget_balance < -1e-9),
        "subsidy_per_served": subsidy / matched if matched else 0.0,
        "subsidy_pct_of_welfare": (
            100.0 * subsidy / m.realised_welfare
            if m.realised_welfare > 1e-9 else 0.0
        ),
    }
    for field in METRIC_FIELDS:
        row[field] = float(getattr(m, field))
    if extra:
        row.update(extra)
    return row


def _true_value(passenger, mode):
    return passenger.v_solo if mode is Mode.SOLO else passenger.v_share


def _true_utilities(allocation, passengers, drivers):
    """True quasi-linear utility of every agent under an allocation."""
    pax = {p.id: 0.0 for p in passengers}
    drv = {d.id: 0.0 for d in drivers}
    p_by = {p.id: p for p in passengers}
    d_by = {d.id: d for d in drivers}
    for g in allocation.groups:
        for pid in g.passenger_ids:
            pax[pid] = (_true_value(p_by[pid], g.mode)
                        - allocation.payments.get(pid, 0.0))
        d = d_by[g.driver_id]
        drv[d.id] = (allocation.driver_earnings.get(d.id, 0.0)
                     - g.route_cost - d.reservation)
    return pax, drv


def _passenger_misreports(passenger):
    out = []
    for x in MISREPORT_GRID:
        v_solo = max(0.0, passenger.v_solo - x)
        v_share = max(0.0, passenger.v_share - x)
        out.append((f"shade_{x:g}", replace(passenger, v_solo=v_solo,
                                            v_share=v_share)))
    out.append(("decline_shared", replace(passenger, v_share=0.0)))
    return out


def _driver_misreports(driver):
    return [(f"inflate_{x:g}", replace(driver, reservation=driver.reservation + x))
            for x in MISREPORT_GRID]


# --------------------------------------------------------------------------
# One market
# --------------------------------------------------------------------------

def run_market(task):
    seed, n_pax, pp_menu = task
    network = grid_network(GRID_ROWS, GRID_COLS)
    markets = build_nested_markets(seed, network)
    passengers, drivers = markets[n_pax]
    p_by = {p.id: p for p in passengers}
    d_by = {d.id: d for d in drivers}
    groups, _ = enumerate_groups(drivers, passengers, network)

    rows, grid_rows, manip_rows = [], [], []

    # ---- ORA and TB ------------------------------------------------------
    result = clear_market(groups, p_by, d_by)
    tb_alloc, tb_info = threshold_rule(result, p_by, d_by)
    rows.append(_row(seed, n_pax, "ORA", result.allocation,
                     passengers, drivers, network))
    rows.append(_row(seed, n_pax, "TB", tb_alloc, passengers, drivers,
                     network,
                     extra={"tb_threshold": tb_info.threshold,
                            "tb_capped_agents": tb_info.n_capped_agents,
                            "tb_payment_distance": (
                                tb_info.sum_vickrey_discounts
                                - tb_info.sum_capped_discounts)}))

    # ---- PP grid ---------------------------------------------------------
    if pp_menu is None:
        for fare, base in product(FARE_GRID, BASE_PAY_GRID):
            prices = PostedPrices(fare, SHARE_RATIO, base, PER_KM_PAY)
            alloc = posted_price_clear(groups, p_by, d_by, prices)
            grid_rows.append(_row(seed, n_pax, "PP", alloc, passengers,
                                  drivers, network,
                                  extra={"fare_solo": fare,
                                         "base_pay": base}))
        return rows, grid_rows, manip_rows

    prices = PostedPrices(pp_menu[0], SHARE_RATIO, pp_menu[1], PER_KM_PAY)
    pp_alloc = posted_price_clear(groups, p_by, d_by, prices)
    rows.append(_row(seed, n_pax, "PP", pp_alloc, passengers, drivers,
                     network, extra={"fare_solo": pp_menu[0],
                                     "base_pay": pp_menu[1]}))

    # ---- Manipulation ----------------------------------------------------
    truth = {
        "ORA": _true_utilities(result.allocation, passengers, drivers),
        "TB": _true_utilities(tb_alloc, passengers, drivers),
        "PP": _true_utilities(pp_alloc, passengers, drivers),
    }
    served = {
        "ORA": result.allocation, "TB": tb_alloc, "PP": pp_alloc,
    }

    def agents_of(alloc):
        pax = {pid for g in alloc.groups for pid in g.passenger_ids}
        drv = {g.driver_id for g in alloc.groups}
        return pax, drv

    best = {}  # (arm, kind, id) -> (gain, label)

    def record(arm, kind, agent_id, label, utility):
        base = truth[arm][0 if kind == "passenger" else 1][agent_id]
        gain = utility - base
        key = (arm, kind, agent_id)
        if key not in best or gain > best[key][0]:
            best[key] = (gain, label, base)

    # ORA and TB share one clearing per misreport, because TB only re-prices
    # ORA's allocation. Only agents allocated under truth are examined.
    pax_ora, drv_ora = agents_of(result.allocation)
    pax_pp, drv_pp = agents_of(pp_alloc)

    for pid in sorted(pax_ora | pax_pp):
        for label, fake in _passenger_misreports(p_by[pid]):
            reported = dict(p_by)
            reported[pid] = fake
            if pid in pax_ora:
                r = clear_market(groups, reported, d_by)
                t, _ = threshold_rule(r, reported, d_by)
                record("ORA", "passenger", pid, label,
                       _true_utilities(r.allocation, passengers, drivers)[0][pid])
                record("TB", "passenger", pid, label,
                       _true_utilities(t, passengers, drivers)[0][pid])
            if pid in pax_pp:
                a = posted_price_clear(groups, reported, d_by, prices)
                record("PP", "passenger", pid, label,
                       _true_utilities(a, passengers, drivers)[0][pid])

    for did in sorted(drv_ora | drv_pp):
        for label, fake in _driver_misreports(d_by[did]):
            reported = dict(d_by)
            reported[did] = fake
            if did in drv_ora:
                r = clear_market(groups, p_by, reported)
                t, _ = threshold_rule(r, p_by, reported)
                record("ORA", "driver", did, label,
                       _true_utilities(r.allocation, passengers, drivers)[1][did])
                record("TB", "driver", did, label,
                       _true_utilities(t, passengers, drivers)[1][did])
            if did in drv_pp:
                a = posted_price_clear(groups, p_by, reported, prices)
                record("PP", "driver", did, label,
                       _true_utilities(a, passengers, drivers)[1][did])

    for (arm, kind, agent_id), (gain, label, base) in best.items():
        manip_rows.append({
            "seed": int(seed),
            "n_passengers": int(n_pax),
            "passenger_driver_ratio": n_pax / N_DRIVERS,
            "arm": arm,
            "agent_kind": kind,
            "agent_id": int(agent_id),
            "truthful_utility": base,
            "best_gain": gain,
            "best_misreport": label,
            "profitable": float(gain > 1e-6),
        })

    return rows, grid_rows, manip_rows


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarize(raw, keys, fields):
    out = []
    for key_values, part in raw.groupby(keys, sort=True):
        base = dict(zip(keys, key_values if isinstance(key_values, tuple)
                        else (key_values,)))
        for field in fields:
            x = part[field].astype(float)
            n = len(x)
            sd = x.std(ddof=1) if n > 1 else 0.0
            out.append({**base, "metric": field, "n": n, "mean": x.mean(),
                        "std": sd, "ci95": 1.96 * sd / np.sqrt(n) if n else 0})
    return pd.DataFrame(out)


def main():
    RESULTS.mkdir(exist_ok=True)
    workers = int(os.environ.get("ORA_WORKERS", os.cpu_count() or 1))

    # Pass 1: PP grid (cheap), to pick the welfare-best menu per pressure.
    tasks = [(s, n, None) for n in PRESSURES for s in SEEDS]
    with Pool(workers) as pool:
        out = pool.map(run_market, tasks)
    grid = pd.DataFrame([r for _, g, _ in out for r in g])
    grid.to_csv(RESULTS / "budget_balanced_pp_grid.csv", index=False)

    menu_means = (grid.groupby(["n_passengers", "fare_solo", "base_pay"])
                  [["realised_welfare", "match_rate", "budget_balance"]]
                  .mean().reset_index())
    best_menu = {}
    for n, part in menu_means.groupby("n_passengers"):
        top = part.sort_values("realised_welfare", ascending=False).iloc[0]
        best_menu[int(n)] = (float(top.fare_solo), float(top.base_pay))
    print("welfare-best PP menu per pressure:", best_menu, flush=True)

    # Pass 2: headline arms and manipulation, PP at its best menu.
    tasks = [(s, n, best_menu[n]) for n in PRESSURES for s in SEEDS]
    with Pool(workers) as pool:
        out = pool.map(run_market, tasks)
    raw = pd.DataFrame([r for rows, _, _ in out for r in rows])
    manip = pd.DataFrame([r for _, _, m in out for r in m])
    raw.to_csv(RESULTS / "budget_balanced_raw.csv", index=False)
    manip.to_csv(RESULTS / "budget_balanced_manipulation.csv", index=False)

    fields = list(METRIC_FIELDS) + ["subsidy", "deficit",
                                    "subsidy_per_served",
                                    "subsidy_pct_of_welfare"]
    summarize(raw, ["n_passengers", "arm"], fields).to_csv(
        RESULTS / "budget_balanced_summary.csv", index=False)

    # Manipulation is aggregated per seed first (share of allocated agents
    # with a profitable misreport, mean best gain), then across seeds, so
    # every seed carries equal weight.
    per_seed = (manip.groupby(["n_passengers", "arm", "agent_kind", "seed"])
                .agg(share_profitable=("profitable", "mean"),
                     mean_best_gain=("best_gain", lambda g: g.clip(lower=0).mean()),
                     max_best_gain=("best_gain", "max"))
                .reset_index())
    summarize(per_seed, ["n_passengers", "arm", "agent_kind"],
              ["share_profitable", "mean_best_gain", "max_best_gain"]).to_csv(
        RESULTS / "budget_balanced_manip_summary.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
