"""Tests for passenger acceptability accounting.

The load-bearing property is that ORA pays no acceptability cost while a
mechanism that ignores the reported bounds does. ORA's zero is not hardcoded
anywhere: the same penalty function runs on all three arms, and ORA comes out at
zero because its feasibility never admits a group that breaches a bound. That
makes these tests a check on the feasibility layer as much as on the metric.
"""
import numpy as np
import pytest

from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.metrics import compute_metrics
from ora.baselines import run_ablation
from ora.someya import run_someya
from ora.acceptability import omega_for, excess_penalty, nominal_value
from ora.types import Passenger, Mode


SEEDS = [0, 1, 2, 3, 4]
PARAMS = DemandParams(n_passengers=12, n_drivers=4, capacity=3)


def _passenger(**overrides):
    base = dict(id=0, origin=0, destination=5, v_solo=10.0, v_share=7.0,
                r_max=5.0, delta_max=2.0)
    base.update(overrides)
    return Passenger(**base)


def test_omega_reaches_surplus_at_the_bound():
    """The rate is set so the penalty equals the sharing surplus at the bound.

    This is the economic content of the derivation: hard bound and soft rate
    agree exactly at the tolerance the passenger reported.
    """
    net = grid_network(6, 6)
    p = _passenger()
    omega_det, _ = omega_for(p, net)
    assert omega_det * p.delta_max == pytest.approx(p.v_solo - p.v_share)


def test_no_penalty_inside_the_bounds():
    """A ride within both reported bounds costs the passenger nothing."""
    net = grid_network(6, 6)
    p = _passenger(delta_max=2.0, r_max=5.0)
    # Detour and wait both comfortably inside.
    assert excess_penalty(p, 1.0, 1.0, net) == 0.0
    # Exactly at the bound is still inside: the bound is what she accepts.
    assert excess_penalty(p, 2.0, 0.0, net) == 0.0


def test_penalty_charges_only_the_excess():
    """Only the overshoot is charged, and it scales with how far it went."""
    net = grid_network(6, 6)
    p = _passenger(delta_max=2.0, r_max=5.0)
    omega_det, _ = omega_for(p, net)

    small = excess_penalty(p, 2.5, 0.0, net)
    large = excess_penalty(p, 4.0, 0.0, net)

    assert small == pytest.approx(omega_det * 0.5)
    assert large == pytest.approx(omega_det * 2.0)
    assert large > small


def test_zero_tolerance_gives_infinite_penalty_only_when_exceeded():
    """A passenger with no tolerance is unharmed until the bound is breached.

    Guards the 0 * inf case: an infinite rate must not turn a zero excess into
    nan, because a ride that never overshot cost her nothing.
    """
    net = grid_network(6, 6)
    strict = _passenger(delta_max=0.0)
    assert excess_penalty(strict, 0.0, 0.0, net) == 0.0
    assert excess_penalty(strict, 0.5, 0.0, net) == float("inf")


def test_nominal_value_matches_group_welfare_convention():
    """Per-passenger valuation reconciles with how welfare is aggregated."""
    p = _passenger()
    assert nominal_value(p, Mode.SOLO) == p.v_solo
    assert nominal_value(p, Mode.SHARED) == p.v_share


def test_ora_pays_no_acceptability_cost():
    """ORA never breaches a reported bound, so its adjustment is exactly zero.

    Not an exemption: the same penalty runs on every arm. A non-zero value here
    would mean ORA's feasibility admitted a group it should have refused.
    """
    net = grid_network(6, 6)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pax, drv = generate(PARAMS, net, rng)
        groups, _ = enumerate_groups(drv, pax, net)
        ora = clear_market(groups, {p.id: p for p in pax},
                           {d.id: d for d in drv})
        metrics = compute_metrics(ora.allocation, pax, drv, net)

        assert metrics.total_acceptability_cost == pytest.approx(0.0)
        assert metrics.mean_adjusted_passenger_utility == pytest.approx(
            metrics.mean_allocated_passenger_utility
        )


def test_someya_pays_a_positive_acceptability_cost():
    """A capacity-only mechanism books welfare its passengers would refuse."""
    net = grid_network(6, 6)
    costs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pax, drv = generate(PARAMS, net, rng)
        som = run_someya(pax, drv, net)
        metrics = compute_metrics(som.allocation, pax, drv, net)
        costs.append(metrics.total_acceptability_cost)

    assert np.mean(costs) > 0.0


def test_adjusted_never_exceeds_nominal():
    """The adjustment can only subtract, never add, on every arm."""
    net = grid_network(6, 6)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pax, drv = generate(PARAMS, net, rng)

        groups, _ = enumerate_groups(drv, pax, net)
        for allocation in (
            clear_market(groups, {p.id: p for p in pax},
                         {d.id: d for d in drv}).allocation,
            run_ablation(pax, drv, net).allocation,
            run_someya(pax, drv, net).allocation,
        ):
            m = compute_metrics(allocation, pax, drv, net)
            assert (m.mean_adjusted_passenger_utility
                    <= m.mean_allocated_passenger_utility + 1e-9)
