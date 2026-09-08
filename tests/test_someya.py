"""Tests for the Someya & Namerikawa (2026) comparison arm.

The most important test here is `test_ora_and_ablation_unchanged_by_arm`. It is
the executable form of the promise that adding this arm does not move any
previously reported number. If it ever fails, the isolation contract in
someya.py has been broken and the earlier results are no longer trustworthy.

The remaining tests pin the behaviour the arm exists to demonstrate: that a
capacity-only mechanism, run on our population and measured against our true
constraints, picks passengers up beyond their radius and routes them past their
detour tolerance, both of which ORA has zero of by construction.
"""
import numpy as np
import pytest

from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.baselines import run_ablation, radius_violations
from ora.someya import (
    run_someya,
    detour_violations,
    someya_valuation,
    enumerate_someya_groups,
    _omega,
)
from ora.types import Passenger, Driver, Mode


SEEDS = [0, 1, 2, 3, 4]
PARAMS = DemandParams(n_passengers=12, n_drivers=4, capacity=3)


def _instance(seed):
    net = grid_network(6, 6)
    rng = np.random.default_rng(seed)
    pax, drv = generate(PARAMS, net, rng)
    return net, pax, drv


def _capture_ora_ablation():
    """Deterministic snapshot of every ORA and ablation output we report."""
    out = {}
    for seed in SEEDS:
        net, pax, drv = _instance(seed)
        groups, _ = enumerate_groups(drv, pax, net)
        ora = clear_market(groups, {p.id: p for p in pax},
                           {d.id: d for d in drv})
        abl = run_ablation(pax, drv, net)
        out[seed] = {
            "ora_welfare": ora.welfare,
            "ora_payments": dict(sorted(ora.allocation.payments.items())),
            "ora_earnings": dict(sorted(
                ora.allocation.driver_earnings.items())),
            "ora_viol": radius_violations(ora.allocation, pax)[1],
            "abl_welfare": abl.welfare,
            "abl_payments": dict(sorted(abl.allocation.payments.items())),
            "abl_viol": radius_violations(abl.allocation, pax)[1],
        }
    return out


def test_ora_and_ablation_unchanged_by_arm():
    """ORA and the ablation must be bit-identical whether or not the arm runs.

    This is the regression guard for the whole extension. The comparison is
    exact equality, not approximate: the arm shares no mutable state with the
    ORA path, so any difference at all means something leaked.
    """
    before = _capture_ora_ablation()

    # Exercise the arm on every seed, which is what a real run does.
    for seed in SEEDS:
        net, pax, drv = _instance(seed)
        run_someya(pax, drv, net)

    after = _capture_ora_ablation()

    assert before == after


def test_arm_returns_ora_shaped_output():
    """The metrics layer must not be able to tell the arm apart from ORA."""
    net, pax, drv = _instance(0)
    result = run_someya(pax, drv, net)
    # Same ClearingResult contract as ORA.
    assert hasattr(result, "allocation")
    assert hasattr(result, "welfare")
    # Someya does not pay drivers (Option A: driver side is out of scope).
    assert result.allocation.driver_earnings == {}


def test_arm_generates_radius_violations():
    """The headline result: a capacity-only mechanism violates r_max.

    ORA has zero radius violations by construction on the same instance, so a
    strictly positive rate here is the acceptability cost the arm exists to
    expose. Averaged over seeds to avoid depending on a single draw.
    """
    ora_rate = []
    som_rate = []
    for seed in SEEDS:
        net, pax, drv = _instance(seed)
        groups, _ = enumerate_groups(drv, pax, net)
        ora = clear_market(groups, {p.id: p for p in pax},
                           {d.id: d for d in drv})
        som = run_someya(pax, drv, net)
        ora_rate.append(radius_violations(ora.allocation, pax)[1])
        som_rate.append(radius_violations(som.allocation, pax)[1])

    assert np.mean(ora_rate) == 0.0
    assert np.mean(som_rate) > 0.0


def test_arm_generates_detour_violations():
    """The arm also routes some passengers past their true delta_max.

    ORA has zero by construction. The rate is lower than the radius rate
    because the router still minimises distance and often stays within
    tolerance by accident, so this only asserts strict positivity in
    aggregate, not a specific level.
    """
    ora_rate = []
    som_rate = []
    for seed in SEEDS:
        net, pax, drv = _instance(seed)
        groups, _ = enumerate_groups(drv, pax, net)
        ora = clear_market(groups, {p.id: p for p in pax},
                           {d.id: d for d in drv})
        som = run_someya(pax, drv, net)
        ora_rate.append(detour_violations(ora.allocation, pax)[1])
        som_rate.append(detour_violations(som.allocation, pax)[1])

    assert np.mean(ora_rate) == 0.0
    assert np.mean(som_rate) > 0.0


def test_zero_disutility_reduces_to_solo_value():
    """With no detour and no wait, a passenger's Someya value is her v_solo.

    This pins the valuation's base case: b_i vanishes when there is nothing to
    penalise, so V_i collapses to C_i^self = v_solo. Uses a one-passenger solo
    group, where detour and shared wait are both zero.
    """
    net = grid_network(6, 6)
    passenger = Passenger(id=0, origin=0, destination=5,
                          v_solo=10.0, v_share=7.0,
                          r_max=5.0, delta_max=2.0)
    driver = Driver(id=0, location=0, capacity=3,
                    cost_per_km=0.0, reservation=0.0)
    groups = enumerate_someya_groups([driver], [passenger], net)
    solo = next(g for g in groups if g.mode is Mode.SOLO)
    # cost_per_km is zero, so route cost drops out and the weight is v_solo.
    weight = someya_valuation(solo, {0: passenger}, {0: driver}, net)
    assert weight == pytest.approx(passenger.v_solo)


def test_omega_derivation_hits_surplus_at_bound():
    """omega_det scaled by delta_max returns exactly the sharing surplus.

    This is the economic meaning of the derivation closed in design: the soft
    penalty reaches v_solo - v_share at exactly the hard tolerance, so the two
    models agree at the boundary.
    """
    net = grid_network(6, 6)
    passenger = Passenger(id=0, origin=0, destination=5,
                          v_solo=10.0, v_share=7.0,
                          r_max=5.0, delta_max=2.0)
    omega_det, _ = _omega(passenger, net)
    surplus = passenger.v_solo - passenger.v_share
    assert omega_det * passenger.delta_max == pytest.approx(surplus)


def test_zero_tolerance_makes_shared_group_worthless():
    """A passenger who tolerates no detour cannot be pooled under any penalty.

    delta_max == 0 gives an infinite omega_det, which the valuation reads as a
    group she would never accept, returning -inf so the optimiser never selects
    it. This mirrors ORA refusing the group outright.
    """
    net = grid_network(6, 6)
    strict = Passenger(id=0, origin=0, destination=8,
                       v_solo=10.0, v_share=7.0,
                       r_max=5.0, delta_max=0.0)
    other = Passenger(id=1, origin=1, destination=9,
                      v_solo=10.0, v_share=7.0,
                      r_max=5.0, delta_max=5.0)
    driver = Driver(id=0, location=0, capacity=3,
                    cost_per_km=1.0, reservation=0.0)
    groups = enumerate_someya_groups([driver], [strict, other], net)
    shared = [g for g in groups
              if g.mode is Mode.SHARED and 0 in g.passenger_ids]
    pax = {0: strict, 1: other}
    drv = {0: driver}
    # Every shared group containing the zero-tolerance passenger that actually
    # detours her is worthless. A shared group with zero realised detour for
    # her is allowed, so only assert on groups where her detour is positive.
    for g in shared:
        idx = g.passenger_ids.index(0)
        if g.detour_km[idx] > 0.0:
            assert someya_valuation(g, pax, drv, net) == float("-inf")
