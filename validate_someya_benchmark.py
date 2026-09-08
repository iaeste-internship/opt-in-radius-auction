"""Validate the Someya & Namerikawa (2026) reimplementation.

This is a within-mechanism consistency check.  It compares the implemented
preference-aware Someya mechanism against a Standard-VCG variant that uses the
same groups, routing, set-packing solver and Clarke-pricing code but sets the
Someya detour/wait disutility weights to zero during allocation and pricing.

The source paper reports that explicitly accounting for passenger detour/wait
preferences improves passenger utility relative to Standard VCG, while the
standard variant tends to match/group more aggressively.  The goal here is to
check whether our reimplementation reproduces that qualitative behaviour on the
same ORA synthetic markets; this is not a claim to reproduce the Yokohama
numerical values exactly.

IMPORTANT IMPLEMENTATION DETAIL
-------------------------------
ora.someya stores the imported omega function in the module-global alias
``_omega``.  Therefore the comparator must patch ``ora.someya._omega`` itself.
Patching ``ora.acceptability.omega_for`` after ora.someya has been imported does
not change the alias and leaves the two mechanisms identical.

Outputs
-------
results/someya_validation_raw.csv
results/someya_validation_summary.csv
results/someya_validation_diagnostics.csv
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ora.network import grid_network
from ora.demand import DemandParams, generate
import ora.someya as someya
from ora.acceptability import omega_for as true_omega_for


SEEDS = range(50)
PARAMS = DemandParams(n_passengers=12, n_drivers=4, capacity=3)

RESULTS_DIR = Path("results")
RAW_OUTPUT = RESULTS_DIR / "someya_validation_raw.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "someya_validation_summary.csv"
DIAGNOSTICS_OUTPUT = RESULTS_DIR / "someya_validation_diagnostics.csv"


@dataclass(frozen=True)
class NativeMetrics:
    total_travel_km: float
    service_rate: float
    average_occupancy: float
    matched_passengers: int
    mean_passenger_utility: float
    min_passenger_utility: float
    passenger_ir_violation_rate: float


@contextmanager
def zero_someya_disutility():
    """Temporarily create Someya's Standard-VCG comparator.

    Patch the exact alias used by someya_valuation.  This is the critical fix
    relative to the previous validation script.
    """
    if not hasattr(someya, "_omega"):
        raise RuntimeError(
            "ora.someya has no module-global _omega alias. Inspect someya.py "
            "before running this validation."
        )

    original = someya._omega
    someya._omega = lambda passenger, network: (0.0, 0.0)
    try:
        yield
    finally:
        someya._omega = original


def passenger_native_value(group, pid, passengers, network) -> float:
    """Return the preference-aware passenger valuation V_i(S_k).

    This mirrors the implemented Someya valuation.  A singleton carries no
    sharing disadvantage; a shared group carries detour and waiting disutility.
    """
    p = passengers[pid]
    if group.mode.name == "SOLO":
        return float(p.v_solo)

    idx = group.passenger_ids.index(pid)
    detour_km = float(group.detour_km[idx])
    pickup_km = float(group.pickup_km[idx])
    wait_min = float(network.minutes(pickup_km))
    omega_det, omega_wait = true_omega_for(p, network)

    det_term = 0.0 if detour_km == 0.0 else omega_det * detour_km
    wait_term = 0.0 if wait_min == 0.0 else omega_wait * wait_min
    disadvantage = det_term + wait_term
    return float(p.v_solo - disadvantage)


def native_metrics(result, passengers, drivers, network) -> NativeMetrics:
    """Evaluate either allocation with the true Someya passenger preferences."""
    allocation = result.allocation
    by_id = {p.id: p for p in passengers}

    matched = sum(len(g.passenger_ids) for g in allocation.groups)
    busy_drivers = len({g.driver_id for g in allocation.groups})
    total_km = float(sum(g.route_km for g in allocation.groups))

    utilities: list[float] = []
    for group in allocation.groups:
        for pid in group.passenger_ids:
            value = passenger_native_value(group, pid, by_id, network)
            payment = float(allocation.payments.get(pid, 0.0))
            utilities.append(value - payment)

    if utilities:
        arr = np.asarray(utilities, dtype=float)
        mean_u = float(arr.mean())
        min_u = float(arr.min())
        ir_rate = float(np.mean(arr < -1e-9))
    else:
        mean_u = 0.0
        min_u = 0.0
        ir_rate = 0.0

    return NativeMetrics(
        total_travel_km=total_km,
        service_rate=(matched / len(passengers)) if passengers else 0.0,
        average_occupancy=(matched / busy_drivers) if busy_drivers else 0.0,
        matched_passengers=matched,
        mean_passenger_utility=mean_u,
        min_passenger_utility=min_u,
        passenger_ir_violation_rate=ir_rate,
    )


def allocation_signature(result):
    return tuple(sorted(
        (g.driver_id, tuple(sorted(g.passenger_ids)), g.mode.name)
        for g in result.allocation.groups
    ))


def group_weight_diagnostic(passengers, drivers, network):
    """Prove that the zero-disutility comparator actually changes weights."""
    pax = {p.id: p for p in passengers}
    drv = {d.id: d for d in drivers}
    groups = someya.enumerate_someya_groups(drivers, passengers, network)

    pref = np.asarray([
        someya.someya_valuation(g, pax, drv, network)
        for g in groups
    ], dtype=float)

    with zero_someya_disutility():
        standard = np.asarray([
            someya.someya_valuation(g, pax, drv, network)
            for g in groups
        ], dtype=float)

    finite = np.isfinite(pref) & np.isfinite(standard)
    if not finite.any():
        return len(groups), 0, 0.0, 0.0

    delta = standard[finite] - pref[finite]
    n_changed = int(np.sum(np.abs(delta) > 1e-10))
    max_abs = float(np.max(np.abs(delta)))
    mean_abs = float(np.mean(np.abs(delta)))
    return len(groups), n_changed, max_abs, mean_abs


def ci95(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def paired_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import ttest_rel
    except Exception:
        return float("nan")
    return float(ttest_rel(a, b, nan_policy="omit").pvalue)


def dz(diff: np.ndarray) -> float:
    if len(diff) < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    mean = float(np.mean(diff))
    if np.isclose(sd, 0.0):
        return 0.0 if np.isclose(mean, 0.0) else float("inf")
    return mean / sd


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "total_travel_km",
        "service_rate",
        "average_occupancy",
        "matched_passengers",
        "mean_passenger_utility",
        "min_passenger_utility",
        "passenger_ir_violation_rate",
    ]

    expected_direction = {
        "service_rate": -1,
        "average_occupancy": -1,
        "matched_passengers": -1,
        "mean_passenger_utility": +1,
    }

    out = []
    for metric in metrics:
        pref = raw.loc[
            raw.arm == "Someya preference-aware", metric
        ].to_numpy(float)
        std = raw.loc[
            raw.arm == "Someya Standard VCG", metric
        ].to_numpy(float)
        diff = pref - std

        row = {
            "metric": metric,
            "someya_mean": float(np.mean(pref)),
            "someya_ci95": ci95(pref),
            "standard_vcg_mean": float(np.mean(std)),
            "standard_vcg_ci95": ci95(std),
            "paired_difference_someya_minus_standard": float(np.mean(diff)),
            "paired_difference_ci95": ci95(diff),
            "paired_p_value": paired_pvalue(pref, std),
            "cohen_dz": dz(diff),
        }

        if metric in expected_direction:
            sign = expected_direction[metric]
            if sign > 0:
                row["source_expected_direction"] = "Someya >= Standard VCG"
                row["seed_direction_consistency"] = float(np.mean(diff >= -1e-12))
            else:
                row["source_expected_direction"] = "Someya <= Standard VCG"
                row["seed_direction_consistency"] = float(np.mean(diff <= 1e-12))
        else:
            row["source_expected_direction"] = "diagnostic"
            row["seed_direction_consistency"] = float("nan")

        out.append(row)

    return pd.DataFrame(out)


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    network = grid_network(6, 6)
    rows = []
    diagnostics = []

    print("Someya benchmark validation -- corrected comparator")
    print(f"  seeds: {len(list(SEEDS))}")
    print("  Standard VCG patches ora.someya._omega -> (0, 0)")

    any_weight_change = False
    allocation_changes = 0

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        passengers, drivers = generate(PARAMS, network, rng)

        n_groups, n_changed, max_abs, mean_abs = group_weight_diagnostic(
            passengers, drivers, network
        )
        any_weight_change = any_weight_change or (n_changed > 0)

        pref = someya.run_someya(passengers, drivers, network)
        m_pref = native_metrics(pref, passengers, drivers, network)

        with zero_someya_disutility():
            standard = someya.run_someya(passengers, drivers, network)
        m_std = native_metrics(standard, passengers, drivers, network)

        changed_alloc = allocation_signature(pref) != allocation_signature(standard)
        allocation_changes += int(changed_alloc)

        diagnostics.append({
            "seed": seed,
            "n_groups": n_groups,
            "n_group_weights_changed": n_changed,
            "max_abs_group_weight_change": max_abs,
            "mean_abs_group_weight_change": mean_abs,
            "allocation_changed": changed_alloc,
        })

        for arm, metrics in (
            ("Someya preference-aware", m_pref),
            ("Someya Standard VCG", m_std),
        ):
            row = {"seed": seed, "arm": arm}
            row.update(asdict(metrics))
            rows.append(row)

        print(
            f"seed {seed:02d}: weights changed {n_changed:4d}/{n_groups:4d}; "
            f"allocation_changed={changed_alloc}; "
            f"service {m_pref.service_rate:.3f} vs {m_std.service_rate:.3f}; "
            f"utility {m_pref.mean_passenger_utility:.3f} vs "
            f"{m_std.mean_passenger_utility:.3f}; "
            f"IR {m_pref.passenger_ir_violation_rate:.3f}"
        )

    if not any_weight_change:
        raise RuntimeError(
            "VALIDATION ABORTED: zeroing ora.someya._omega did not change any "
            "candidate-group weight. The comparator is still not active."
        )

    raw = pd.DataFrame(rows)
    summary = build_summary(raw)
    diag = pd.DataFrame(diagnostics)

    raw.to_csv(RAW_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    diag.to_csv(DIAGNOSTICS_OUTPUT, index=False)

    print("\nSaved:")
    print(f"  {RAW_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")
    print(f"  {DIAGNOSTICS_OUTPUT}")
    print(f"\nAllocations changed in {allocation_changes}/{len(list(SEEDS))} seeds.")

    if allocation_changes == 0:
        print(
            "WARNING: group weights changed, but the selected allocation did "
            "not change in any seed. The comparator is technically active, "
            "but this market design would be weak evidence for behavioural "
            "validation."
        )

    print("\nHeadline validation checks:")
    for metric in (
        "service_rate",
        "average_occupancy",
        "matched_passengers",
        "mean_passenger_utility",
        "passenger_ir_violation_rate",
    ):
        row = summary.loc[summary.metric == metric].iloc[0]
        print(
            f"  {metric:30s} "
            f"Someya={row['someya_mean']:.4f} "
            f"StdVCG={row['standard_vcg_mean']:.4f} "
            f"diff={row['paired_difference_someya_minus_standard']:+.4f} "
            f"p={row['paired_p_value']:.4g} "
            f"dz={row['cohen_dz']:.3f}"
        )


if __name__ == "__main__":
    run()
