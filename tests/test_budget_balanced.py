"""Tests for the budget-balanced comparators of Experiment E.

Two kinds of test. Hand-built markets check the threshold arithmetic on
numbers worked out by hand. Random markets check the properties each
comparator is supposed to have by construction: budget balance, individual
rationality, and (for the threshold rule) an unchanged allocation.
"""
import numpy as np
import pytest

from ora.budget_balanced import (
    PostedPrices,
    _threshold_for,
    posted_price_clear,
    threshold_rule,
    vickrey_discounts,
)
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups, group_welfare
from ora.network import grid_network
from ora.payments import budget_balance, clear_market
from ora.types import Mode

TOL = 1e-6


def random_market(seed, n_passengers=12, n_drivers=4):
    network = grid_network(6, 6)
    rng = np.random.default_rng(seed)
    params = DemandParams(n_passengers=n_passengers, n_drivers=n_drivers,
                          capacity=3)
    passengers, drivers = generate(params, network, rng)
    groups, _ = enumerate_groups(drivers, passengers, network)
    return (groups, {p.id: p for p in passengers},
            {d.id: d for d in drivers})


# --------------------------------------------------------------------------
# Threshold arithmetic
# --------------------------------------------------------------------------

def test_threshold_caps_only_the_largest_discounts():
    # Discounts 1, 2, 7 sum to 10. Target 6: cap C with 1 + 2 + C = 6 -> C = 3.
    assert _threshold_for([7.0, 1.0, 2.0], 6.0) == pytest.approx(3.0)


def test_threshold_can_fall_below_the_smallest_discount():
    # Target 1.5 with three discounts of at least 1: C = 0.5 for everyone.
    assert _threshold_for([1.0, 2.0, 3.0], 1.5) == pytest.approx(0.5)


def test_threshold_zero_target_gives_zero_cap():
    assert _threshold_for([1.0, 2.0], 0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Threshold rule on random markets
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_threshold_rule_keeps_allocation_and_balances_budget(seed):
    groups, passengers, drivers = random_market(seed)
    result = clear_market(groups, passengers, drivers)
    allocation, info = threshold_rule(result, passengers, drivers)

    assert allocation.groups == result.allocation.groups
    budget = budget_balance(allocation)
    if info.vcg_budget < 0:
        assert budget == pytest.approx(0.0, abs=TOL)
    else:
        assert budget == pytest.approx(info.vcg_budget, abs=TOL)


@pytest.mark.parametrize("seed", range(8))
def test_threshold_rule_is_individually_rational(seed):
    groups, passengers, drivers = random_market(seed)
    result = clear_market(groups, passengers, drivers)
    allocation, _ = threshold_rule(result, passengers, drivers)

    for group in allocation.groups:
        for pid in group.passenger_ids:
            p = passengers[pid]
            value = p.v_solo if group.mode is Mode.SOLO else p.v_share
            assert value - allocation.payments[pid] >= -TOL
        d = drivers[group.driver_id]
        utility = (allocation.driver_earnings[d.id]
                   - group.route_cost - d.reservation)
        assert utility >= -TOL


@pytest.mark.parametrize("seed", range(8))
def test_threshold_discounts_never_exceed_vickrey(seed):
    groups, passengers, drivers = random_market(seed)
    result = clear_market(groups, passengers, drivers)
    pax_before, drv_before = vickrey_discounts(result, passengers, drivers)
    allocation, info = threshold_rule(result, passengers, drivers)

    for group in allocation.groups:
        for pid in group.passenger_ids:
            p = passengers[pid]
            value = p.v_solo if group.mode is Mode.SOLO else p.v_share
            after = value - allocation.payments[pid]
            assert after <= pax_before[pid] + TOL
            assert after <= info.threshold + TOL
    assert info.sum_capped_discounts <= info.sum_vickrey_discounts + TOL


def test_vickrey_discounts_account_for_the_clarke_budget():
    # B = W* - sum of discounts, the identity the threshold rule relies on.
    groups, passengers, drivers = random_market(3)
    result = clear_market(groups, passengers, drivers)
    pax, drv = vickrey_discounts(result, passengers, drivers)
    identity = result.welfare - sum(pax.values()) - sum(drv.values())
    assert budget_balance(result.allocation) == pytest.approx(identity,
                                                              abs=TOL)


# --------------------------------------------------------------------------
# Posted price
# --------------------------------------------------------------------------

PRICES = PostedPrices(fare_solo=14.0, share_ratio=0.75, base_pay=7.0,
                      per_km_pay=0.8)


@pytest.mark.parametrize("seed", range(8))
def test_posted_price_is_budget_balanced_and_rational(seed):
    groups, passengers, drivers = random_market(seed)
    allocation = posted_price_clear(groups, passengers, drivers, PRICES)

    assert budget_balance(allocation) >= -TOL
    for group in allocation.groups:
        fare = PRICES.fare(group.mode)
        for pid in group.passenger_ids:
            p = passengers[pid]
            value = p.v_solo if group.mode is Mode.SOLO else p.v_share
            assert allocation.payments[pid] == pytest.approx(fare)
            assert value - fare >= -TOL
        d = drivers[group.driver_id]
        assert (allocation.driver_earnings[d.id]
                - group.route_cost - d.reservation) >= -TOL
        # Admissible groups always create non-negative true welfare.
        assert group_welfare(group, passengers, drivers) >= -TOL


@pytest.mark.parametrize("seed", range(4))
def test_posted_price_only_uses_bound_respecting_groups(seed):
    groups, passengers, drivers = random_market(seed)
    allocation = posted_price_clear(groups, passengers, drivers, PRICES)
    available = set(groups)
    for group in allocation.groups:
        assert group in available


def test_posted_price_at_prohibitive_fare_serves_nobody():
    groups, passengers, drivers = random_market(0)
    prices = PostedPrices(fare_solo=1e6, share_ratio=0.75, base_pay=7.0,
                          per_km_pay=0.8)
    allocation = posted_price_clear(groups, passengers, drivers, prices)
    assert allocation.groups == ()


@pytest.mark.parametrize("seed", range(4))
def test_posted_price_each_agent_at_most_once(seed):
    groups, passengers, drivers = random_market(seed, n_passengers=16)
    allocation = posted_price_clear(groups, passengers, drivers, PRICES)
    served = [pid for g in allocation.groups for pid in g.passenger_ids]
    used = [g.driver_id for g in allocation.groups]
    assert len(served) == len(set(served))
    assert len(used) == len(set(used))
