"""Invariant tests: properties that must hold on any generated market.

These are different in kind from the unit tests. A unit test checks that a
function returns the number I worked out by hand on a small example. An
invariant test asserts a property of the mechanism itself over markets I did
not construct, drawn from the demand generator at several seeds. If one of
these fails, the claim in my write-up is false, not just the code.

Each test names the claim it is defending, because a failure message that says
only "assert False" is useless three weeks later.
"""
import numpy as np
import pytest

from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import (clear_market, passenger_utility, driver_utility,
                          budget_balance)

SEEDS = [0, 1, 2, 3, 4]


def market(seed, n_passengers=12, n_drivers=4):
    """One cleared market at a given seed. Small enough to stay fast."""
    net = grid_network(6, 6)
    params = DemandParams(n_passengers=n_passengers, n_drivers=n_drivers,
                          capacity=3)
    rng = np.random.default_rng(seed)
    passengers, drivers = generate(params, net, rng)
    groups, _ = enumerate_groups(drivers, passengers, net)
    result = clear_market(groups,
                          {p.id: p for p in passengers},
                          {d.id: d for d in drivers})
    return passengers, drivers, result


def adverse_market(seed, n_passengers=6, n_drivers=8):
    """A demanding market for driver-side individual rationality.

    The market contains more drivers than passengers and uses relatively high
    reservation values. It is deliberately chosen so that driver participation
    costs are important for the allocation rather than automatically covered
    by a large demand surplus.

    Since reservation values now enter the welfare objective, inefficient
    driver participation should be rejected and every allocated driver should
    retain non-negative utility.
    """
    net = grid_network(6, 6)
    params = DemandParams(n_passengers=n_passengers, n_drivers=n_drivers,
                          capacity=3, reservation_mean=15.0, reservation_sd=3.0)
    rng = np.random.default_rng(seed)
    passengers, drivers = generate(params, net, rng)
    groups, _ = enumerate_groups(drivers, passengers, net)
    result = clear_market(groups,
                          {p.id: p for p in passengers},
                          {d.id: d for d in drivers})
    return passengers, drivers, result


@pytest.mark.parametrize("seed", SEEDS)
def test_allocation_is_feasible(seed):
    """No passenger in two groups, no driver serving two groups.

    The set packing constraints, verified on the output rather than trusted
    from the solver. If this fails the constraints are wrong, not CBC.
    """
    _, _, result = market(seed)
    seen_p, seen_d = set(), set()
    for g in result.allocation.groups:
        for pid in g.passenger_ids:
            assert pid not in seen_p, f"passenger {pid} allocated twice"
            seen_p.add(pid)
        assert g.driver_id not in seen_d, f"driver {g.driver_id} allocated twice"
        seen_d.add(g.driver_id)


@pytest.mark.parametrize("seed", SEEDS)
def test_reported_constraints_respected(seed):
    """Every allocated group respects the radius and detour its members
    reported.

    This is the elicitation claim made testable. If a passenger ends up in a
    group violating her own r_max or delta_max, then reporting them bought her
    nothing and the whole premise of ORA is empty.
    """
    passengers, _, result = market(seed)
    by_id = {p.id: p for p in passengers}
    for g in result.allocation.groups:
        for pid, pickup, detour in zip(g.passenger_ids, g.pickup_km,
                                       g.detour_km):
            p = by_id[pid]
            assert pickup <= p.r_max + 1e-9, f"passenger {pid} over r_max"
            assert detour <= p.delta_max + 1e-9, f"passenger {pid} over delta_max"


@pytest.mark.parametrize("seed", SEEDS)
def test_passenger_individual_rationality(seed):
    """No passenger ends up worse off than not participating.

    Standard Clarke property and the one I expect to hold unconditionally,
    since the empty allocation is always feasible in the maximisation.
    """
    passengers, _, result = market(seed)
    for p in passengers:
        u = passenger_utility(p, result.allocation)
        assert u >= -1e-9, f"passenger {p.id} has negative utility {u}"


@pytest.mark.parametrize("seed", SEEDS)
def test_driver_individual_rationality(seed):
    """Every driver obtains at least the idle outside option.

    This is tested on the adverse market, where reservations are high and
    driver competition is strong. Passing only on the default market would not
    provide a meaningful test of the reservation-value correction.
    """
    _, drivers, result = adverse_market(seed)
    for d in drivers:
        u = driver_utility(d, result.allocation)
        assert u >= -1e-9, f"driver {d.id} has negative utility {u}"


@pytest.mark.parametrize("seed", SEEDS)
def test_payments_non_negative(seed):
    """Passengers are charged, never subsidised.

    Clarke payments are non-negative for served agents in this setting. A
    negative charge would mean the pivot term came out backwards, which is the
    most likely sign error in the whole module.
    """
    _, _, result = market(seed)
    for pid, amount in result.allocation.payments.items():
        assert amount >= -1e-9, f"passenger {pid} charged {amount}"


@pytest.mark.parametrize("seed", SEEDS)
def test_deficit_is_reported_not_hidden(seed):
    """Budget balance is computed and finite.

    Deliberately not asserting it is non-negative. Two-sided Clarke is expected
    to run a deficit, and that number is a result rather than a failure. This
    test only guards against it being undefined or silently absent.
    """
    _, _, result = market(seed)
    bb = budget_balance(result.allocation)
    assert np.isfinite(bb)