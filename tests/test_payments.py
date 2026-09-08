"""Unit tests for the Clarke payment rule.

Different in kind from test_invariants. Here the markets are hand-built, tiny,
and the expected numbers are worked out on paper before the code runs. If one
of these fails I know exactly which arithmetic is wrong, because there is only
one thing happening in each test.

The convention throughout: passenger payments positive means money flows to the
platform, driver earnings positive means money flows from the platform. Every
expected value below is derived under that convention, so a sign error shows up
here rather than three modules downstream.
"""
import numpy as np
import pytest

from ora.types import Passenger, Driver, Mode
from ora.network import grid_network
from ora.grouping import enumerate_groups
from ora.payments import (clear_market, passenger_utility, driver_utility,
                          budget_balance)

TOL = 1e-6


def net():
    """A 3x3 grid with 1 km cells. Distances are integers I can check by eye."""
    return grid_network(3, 3, cell_km=1.0)


def pax(id, origin, destination, v_solo, tau=0.8, r_max=10.0, delta_max=10.0):
    """Passenger with generous flexibility unless a test narrows it.

    Defaults are deliberately loose so that a test about payments is not
    accidentally a test about feasibility. When a test wants to constrain
    matching it says so explicitly.
    """
    return Passenger(id=id, origin=origin, destination=destination,
                     v_solo=v_solo, v_share=v_solo * tau,
                     r_max=r_max, delta_max=delta_max)


def drv(id, location, cost_per_km=1.0, capacity=3, reservation=0.0):
    """Driver with zero reservation unless a test sets one.

    Zero is the default in most hand-built payment tests so that the expected
    Clarke amounts remain simple. Tests involving driver participation costs set a
    non-zero reservation explicitly.
    """
    return Driver(id=id, location=location, capacity=capacity,
                  cost_per_km=cost_per_km, reservation=reservation)


def clear(passengers, drivers, network=None):
    """Enumerate and clear a hand-built market."""
    network = network or net()
    groups, _ = enumerate_groups(drivers, passengers, network)
    return clear_market(groups,
                        {p.id: p for p in passengers},
                        {d.id: d for d in drivers})


# --------------------------------------------------------------------------
# The degenerate cases. These pin down the boundaries of the rule.
# --------------------------------------------------------------------------

def test_empty_market_clears_to_nothing():
    """No agents means no groups, no payments, zero welfare.

    Guards the early return in solve_wdp: an empty market is a legitimate
    counterfactual the payment rule will hit whenever removing an agent empties
    the market, so it must return cleanly rather than raise.
    """
    result = clear([], [])
    assert result.allocation.groups == ()
    assert result.allocation.payments == {}
    assert result.allocation.driver_earnings == {}
    assert result.welfare == pytest.approx(0.0)


def test_single_passenger_alone_pays_her_route_cost():
    """One passenger, one driver: she pays exactly the cost of driving her.

    My first version of this test expected zero, on the reasoning that with no
    competition there is nobody to displace. That reasoning was wrong, and the
    difference matters.

    The Clarke term is (welfare others get without her) minus (welfare others
    get with her). Without her the market is empty, so the first term is zero.
    With her, the others' welfare is her own valuation removed but the driver's
    route cost left in, because that cost is the driver's contribution and not
    hers. So the second term is negative, equal to minus the route cost, and
    the payment comes out as the route cost itself.

    That is the correct behaviour: her presence does impose an externality even
    with no rival, namely the fuel and time the driver spends on her. What she
    does not pay is any scarcity rent, because there is no scarcity. Her
    surplus is her whole valuation minus the cost of serving her, which is the
    efficient outcome.
    """
    p = pax(0, origin=0, destination=8, v_solo=50.0)
    d = drv(0, location=0)
    result = clear([p], [d])

    assert len(result.allocation.groups) == 1
    g = result.allocation.groups[0]
    assert result.allocation.payments[0] == pytest.approx(g.route_cost, abs=TOL)

    # And she keeps the rest, which is what makes participation worthwhile.
    assert passenger_utility(p, result.allocation) == pytest.approx(
        p.v_solo - g.route_cost, abs=TOL)   


def test_unserved_agents_are_absent_from_payments():
    """An agent who gets nothing pays nothing and is not in the dict.

    Absence rather than a zero entry, because the metrics layer distinguishes
    unserved from served-and-charged-zero, and conflating them would corrupt
    the Gini computation over payments.
    """
    # Two passengers, one driver, capacity 1 so only one can be served.
    p0 = pax(0, origin=0, destination=8, v_solo=50.0)
    p1 = pax(1, origin=2, destination=6, v_solo=5.0)
    d = drv(0, location=0, capacity=1)
    result = clear([p0, p1], [d])

    served = {pid for g in result.allocation.groups for pid in g.passenger_ids}
    for pid in (0, 1):
        if pid not in served:
            assert pid not in result.allocation.payments


# --------------------------------------------------------------------------
# Competition. This is where the pivot term becomes non-zero.
# --------------------------------------------------------------------------

def test_competing_passenger_pays_displaced_welfare():
    """Two passengers, one driver, capacity 1: the winner pays what the loser
    would have created.

    This is the textbook Vickrey case and the reason the mechanism is called a
    pivot. The winning passenger's presence costs the others exactly the
    welfare the displaced passenger would have generated, so that number is
    her payment. If this test passes, the exclusion machinery is wired
    correctly.
    """
    p0 = pax(0, origin=0, destination=1, v_solo=50.0)
    p1 = pax(1, origin=0, destination=1, v_solo=20.0)
    d = drv(0, location=0, cost_per_km=1.0, capacity=1)
    result = clear([p0, p1], [d])

    winners = result.allocation.groups
    assert len(winners) == 1
    won_by = winners[0].passenger_ids[0]
    assert won_by == 0, "higher valuation should win"

    # Without p0 the market serves p1: welfare 20 minus the same route cost.
    # With p0 present, the others' welfare is just minus that route cost.
    # The difference is 20.
    assert result.allocation.payments[0] == pytest.approx(20.0, abs=TOL)


def test_payment_never_exceeds_valuation():
    """Individual rationality, in its sharpest form.

    A Clarke payment above the agent's own reported value would mean she is
    better off not participating, which the mechanism must never produce
    because the empty allocation is always feasible in the maximisation. This
    holds by construction, and the test exists because "by construction" is
    what people say right before a sign error.
    """
    p0 = pax(0, origin=0, destination=1, v_solo=50.0)
    p1 = pax(1, origin=0, destination=1, v_solo=49.0)
    d = drv(0, location=0, capacity=1)
    result = clear([p0, p1], [d])

    for pid, amount in result.allocation.payments.items():
        p = p0 if pid == 0 else p1
        assert amount <= p.v_solo + TOL


def test_driver_earnings_cover_route_cost():
    """A selected driver's earnings cover his route cost.

    This checks the operating-cost component separately from the reservation-cost
    component. Full driver individual rationality is tested using driver_utility in
    the invariant tests.
    """
    p = pax(0, origin=0, destination=8, v_solo=100.0)
    d = drv(0, location=0, cost_per_km=1.0)
    result = clear([p], [d])

    g = result.allocation.groups[0]
    earned = result.allocation.driver_earnings[g.driver_id]
    assert earned >= g.route_cost - TOL


# --------------------------------------------------------------------------
# Mode pricing. The part of the rule that is mine rather than inherited.
# --------------------------------------------------------------------------

def test_low_tau_passenger_is_not_pooled():
    """A passenger who barely values sharing rides alone or not at all.

    The core ORA claim made testable. In the reviewed literature this
    preference does not exist: pooling is a platform decision and the passenger
    has no way to express that she would rather not.

    The scenario needs care. My first attempt gave both passengers identical
    origins and destinations, which made the shared route identical to the solo
    route: pooling cost nothing, so even a valuation of 6 was worth adding and
    the optimiser pooled her correctly. The test passed the wrong claim.

    Here the destinations diverge, so serving them together forces a real
    detour and a longer route. Now the comparison is honest: adding the
    low-tau passenger contributes v_share against extra driving, and if her
    share tolerance is low enough the optimiser should decline.
    """
    # Same origin, opposite corners of the grid, so pooling means real detour.
    p0 = pax(0, origin=0, destination=2, v_solo=40.0, tau=0.95)
    p1 = pax(1, origin=0, destination=6, v_solo=40.0, tau=0.10)
    d = drv(0, location=0, cost_per_km=1.0, capacity=2)
    result = clear([p0, p1], [d])

    for g in result.allocation.groups:
        if 1 in g.passenger_ids and g.mode is Mode.SHARED:
            pytest.fail("passenger with tau=0.10 was pooled despite a costly "
                        "shared route")
            
def test_high_tau_passenger_is_pooled_in_the_same_market():
    """The control for the test above, and the reason it means anything.

    Identical geometry, only tau raised. If the high-tau version also refuses
    to pool, then the previous test was passing because of the routing cost
    alone and tau had nothing to do with it. Running both is what isolates the
    mode decision as the thing being tested.
    """
    p0 = pax(0, origin=0, destination=2, v_solo=40.0, tau=0.95)
    p1 = pax(1, origin=0, destination=6, v_solo=40.0, tau=0.95)
    d = drv(0, location=0, cost_per_km=1.0, capacity=2)
    result = clear([p0, p1], [d])

    pooled = any(g.mode is Mode.SHARED and len(g.passenger_ids) == 2
                 for g in result.allocation.groups)
    assert pooled, ("high-tau passengers were not pooled, so the low-tau test "
                    "proves nothing about tau")
    


def test_mode_affects_welfare_and_therefore_price():
    """The same allocation priced under different tau distributions differs.

    Not a specific number, a direction: raising share tolerance across the
    population changes what pooling is worth, which changes the externality
    terms, which changes payments. If payments were invariant to tau then the
    mode would not really be in the allocation space and the whole contribution
    would be decorative.
    """
    def welfare_at(tau):
        ps = [pax(i, origin=0, destination=8, v_solo=40.0, tau=tau)
              for i in range(3)]
        d = drv(0, location=0, capacity=3)
        return clear(ps, [d]).welfare

    assert welfare_at(0.95) > welfare_at(0.30) + TOL


# --------------------------------------------------------------------------
# Budget balance. The number that is a result, not a pass condition.
# --------------------------------------------------------------------------

def test_budget_balance_is_computed_with_the_stated_sign():
    """Collected minus paid, so a deficit is negative.

    The sign convention is asserted here once so that every figure in the
    report inherits it. Getting this backwards would flip the headline number
    of the two-sided-IR section.
    """
    p = pax(0, origin=0, destination=8, v_solo=100.0)
    d = drv(0, location=0)
    result = clear([p], [d])

    manual = (sum(result.allocation.payments.values())
              - sum(result.allocation.driver_earnings.values()))
    assert budget_balance(result.allocation) == pytest.approx(manual, abs=TOL)


def test_utilities_are_quasilinear():
    """Utility is value minus payment, with nothing else folded in.

    Quasi-linearity is a precondition of the Clarke argument, so it is worth
    asserting that the utility functions actually implement it rather than
    silently including a term the theory does not have.
    """
    p = pax(0, origin=0, destination=8, v_solo=60.0)
    d = drv(0, location=0, reservation=3.0)
    result = clear([p], [d])

    g = result.allocation.groups[0]
    value = p.v_solo if g.mode is Mode.SOLO else p.v_share
    expected = value - result.allocation.payments.get(0, 0.0)
    assert passenger_utility(p, result.allocation) == pytest.approx(expected,
                                                                   abs=TOL)

    expected_d = (result.allocation.driver_earnings.get(0, 0.0)
                  - g.route_cost - d.reservation)
    assert driver_utility(d, result.allocation) == pytest.approx(expected_d,
                                                                abs=TOL)

# --------------------------------------------------------------------------
# Driver-side pivot, pinned to hand-computed numbers. The exclusion term for
# a driver is the asymmetric branch in _welfare_excluding_agent: it rebuilds
# the "others with him" figure from scratch instead of starting from
# group_welfare, which is exactly where a sign error on route_cost would hide.
# test_driver_earnings_cover_route_cost only checks an inequality (>= cost),
# so it cannot catch a wrong sign that still lands above the cost. This does.
# --------------------------------------------------------------------------

def test_driver_earnings_equal_hand_computed_pivot():
    """Two drivers competing, cost_per_km != 0, every number checked on paper.

    Layout of the 3x3 unit grid, row-major:
        0 1 2
        3 4 5
        6 7 8

    Both passengers want 0 -> 2 (a 2 km trip). d0 sits on zone 0 (pickup 0),
    d1 sits on zone 6 (2 km from zone 0). Capacity 1 each, so no pooling: this
    is the clean two-sided Vickrey case.

    Optimal allocation: d0 serves p0 (route 2 km, cost 2), d1 serves p1
    (route 4 km, cost 4). Welfare = 50 + 30 - 2 - 4 = 74.

    Driver d0's earnings = W(others | d0 present) - W(others | d0 absent).
      others with d0    : p0's value 50 kept, d0's cost dropped; plus d1's
                          edge 30 - 4 = 26. Total 76.
      others without d0 : only d1 remains, capacity 1, serves the higher-value
                          p0 at 4 km. Welfare 50 - 4 = 46.
      earnings[d0] = 76 - 46 = 30. utility = 30 - 2 - 0 = 28.

    Driver d1's earnings, symmetric reasoning:
      others with d1    : p1's value 30 kept, d1's cost dropped; plus d0's
                          edge 50 - 2 = 48. Total 78.
      others without d1 : only d0, serves p0 at 2 km. Welfare 48.
      earnings[d1] = 78 - 48 = 30. utility = 30 - 4 - 0 = 26.

    A sign flip on the driver's route_cost in the exclusion term moves the 76
    or the 78 and breaks these exact equalities, while leaving the >= cost
    inequality test still passing. That is the whole reason this test exists.
    """
    network = grid_network(3, 3, cell_km=1.0)
    p0 = pax(0, origin=0, destination=2, v_solo=50.0)
    p1 = pax(1, origin=0, destination=2, v_solo=30.0)
    d0 = drv(0, location=0, cost_per_km=1.0, capacity=1)
    d1 = drv(1, location=6, cost_per_km=1.0, capacity=1)
    result = clear([p0, p1], [d0, d1], network)

    assert result.welfare == pytest.approx(74.0, abs=TOL)
    assert result.allocation.driver_earnings[0] == pytest.approx(30.0, abs=TOL)
    assert result.allocation.driver_earnings[1] == pytest.approx(30.0, abs=TOL)
    assert driver_utility(d0, result.allocation) == pytest.approx(28.0, abs=TOL)
    assert driver_utility(d1, result.allocation) == pytest.approx(26.0, abs=TOL)