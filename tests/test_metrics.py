"""Tests for the measurement layer.

Two kinds here. The index tests use distributions whose Gini and Jain values
are known in closed form, so a wrong formula fails immediately rather than
producing a plausible number nobody checks. The metrics tests run on real
cleared markets and assert the properties that must hold whatever the market
looks like.

The index tests matter more than they look. Gini and Jain are the headline
fairness numbers in the report, and a subtly wrong implementation would produce
figures that look reasonable, plot smoothly, and be wrong. Closed-form cases
are the only defence.
"""
import numpy as np
import pytest

from ora.metrics import gini, jain, compute_metrics, Metrics
from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.types import Passenger, Driver, Mode
from ora.payments import clear_market, driver_utility

TOL = 1e-9


# --------------------------------------------------------------------------
# Gini, against distributions with known values.
# --------------------------------------------------------------------------

def test_gini_perfect_equality_is_zero():
    """Everyone identical means no inequality, whatever the level."""
    assert gini([5.0] * 10) == pytest.approx(0.0, abs=TOL)
    assert gini([0.3] * 4) == pytest.approx(0.0, abs=TOL)


def test_gini_maximal_concentration():
    """One agent holds everything: Gini is (n-1)/n, not 1.

    Worth pinning because the textbook statement "Gini is 1 at maximal
    inequality" is the continuous limit. For a finite population the maximum
    is (n-1)/n, and an implementation that normalises to exactly 1 is using a
    different formula than the one I cite.
    """
    for n in (2, 5, 10):
        x = [0.0] * (n - 1) + [100.0]
        assert gini(x) == pytest.approx((n - 1) / n, abs=1e-9)


def test_gini_known_small_case():
    """Values 1,2,3,4 have Gini 0.25. Worked out by hand.

    Mean is 2.5, and the mean absolute difference over all ordered pairs is
    1.25, so G = MAD / (2 * mean) = 1.25 / 5 = 0.25.
    """
    assert gini([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25, abs=1e-9)


def test_gini_is_scale_invariant():
    """Doubling every value leaves inequality unchanged.

    A relative measure must not move when the units do. If this fails the
    implementation is measuring spread rather than inequality.
    """
    x = [1.0, 4.0, 7.0, 9.0]
    assert gini(x) == pytest.approx(gini([10 * v for v in x]), abs=1e-9)


def test_gini_degenerate_populations():
    """Empty and all-zero both return zero rather than NaN.

    NaN here would propagate silently through every aggregate in the sweep.
    """
    assert gini([]) == 0.0
    assert gini([0.0, 0.0, 0.0]) == 0.0


def test_gini_rejects_negative_values():
    """Negative inputs are a caller error, not something to average over."""
    with pytest.raises(ValueError):
        gini([1.0, -2.0, 3.0])


# --------------------------------------------------------------------------
# Jain, same treatment.
# --------------------------------------------------------------------------

def test_jain_perfect_equality_is_one():
    assert jain([7.0] * 6) == pytest.approx(1.0, abs=TOL)


def test_jain_maximal_concentration_is_one_over_n():
    """One agent holds everything: Jain hits its lower bound 1/n.

    This is the property that distinguishes Jain from Gini in the report: the
    floor depends on population size, so Jain values are not comparable across
    arms with different numbers of agents unless that is stated.
    """
    for n in (2, 5, 10):
        x = [0.0] * (n - 1) + [100.0]
        assert jain(x) == pytest.approx(1.0 / n, abs=1e-12)


def test_jain_known_small_case():
    """Values 1,2,3,4: (10)^2 / (4 * 30) = 100/120 = 0.8333..."""
    assert jain([1.0, 2.0, 3.0, 4.0]) == pytest.approx(100.0 / 120.0, abs=1e-12)


def test_jain_is_scale_invariant():
    x = [1.0, 4.0, 7.0, 9.0]
    assert jain(x) == pytest.approx(jain([10 * v for v in x]), abs=1e-12)


def test_gini_and_jain_agree_on_direction():
    """More concentrated means higher Gini and lower Jain.

    They are opposite-signed measures of the same thing, so a bug that flips
    one shows up as the two moving together instead of apart.
    """
    equal = [10.0, 10.0, 10.0, 10.0]
    skewed = [1.0, 1.0, 1.0, 37.0]
    assert gini(skewed) > gini(equal)
    assert jain(skewed) < jain(equal)


# --------------------------------------------------------------------------
# Metrics on real cleared markets.
# --------------------------------------------------------------------------

def cleared(seed=0, n_passengers=12, n_drivers=4):
    net = grid_network(6, 6)
    params = DemandParams(n_passengers=n_passengers, n_drivers=n_drivers,
                          capacity=3)
    rng = np.random.default_rng(seed)
    pax, drv = generate(params, net, rng)
    groups, _ = enumerate_groups(drv, pax, net)
    result = clear_market(groups, {p.id: p for p in pax},
                          {d.id: d for d in drv})
    return result, pax, drv, net


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_metrics_are_finite_and_in_range(seed):
    """Every reported number is finite and rates lie in [0,1].

    The sweep aggregates hundreds of these, so one NaN or one rate above 1
    corrupts a whole figure. Cheaper to catch here.
    """
    result, pax, drv, net = cleared(seed)
    m = compute_metrics(result.allocation, pax, drv, net)

    for name, value in m.as_row().items():
        assert np.isfinite(value), f"{name} is not finite"

    for name in ("match_rate", "pooling_rate", "idle_ratio",
                 "gini_wait_all", "gini_wait_served", "gini_earnings"):
        v = getattr(m, name)
        assert 0.0 <= v <= 1.0, f"{name} out of range: {v}"

    for name in ("jain_wait_all", "jain_wait_served", "jain_earnings"):
        v = getattr(m, name)
        assert 0.0 < v <= 1.0 + 1e-12, f"{name} out of range: {v}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_counts_are_consistent_with_the_allocation(seed):
    """Matched and pooled counts match what is actually in the groups."""
    result, pax, drv, net = cleared(seed)
    m = compute_metrics(result.allocation, pax, drv, net)

    expected_matched = sum(len(g.passenger_ids)
                           for g in result.allocation.groups)
    expected_pooled = sum(len(g.passenger_ids)
                          for g in result.allocation.groups
                          if g.mode is Mode.SHARED)
    busy = {g.driver_id for g in result.allocation.groups}

    assert m.matched_passengers == expected_matched
    assert m.pooled_passengers == expected_pooled
    assert m.idle_drivers == len(drv) - len(busy)
    assert m.match_rate == pytest.approx(expected_matched / len(pax))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_population_is_never_fairer_than_served_only(seed):
    """The all-population Gini is at least the served-only Gini.

    This is the property the whole two-population design exists to enforce. If
    it ever failed, the `all` figure would be gameable by matching less, which
    is exactly the trap the module docstring describes.
    """
    result, pax, drv, net = cleared(seed)
    m = compute_metrics(result.allocation, pax, drv, net)
    assert m.gini_wait_all >= m.gini_wait_served - 1e-12


def test_serving_fewer_is_visible_in_the_metrics():
    """Abandoning passengers must show up somewhere. It is not in the Gini.

    I wrote this test first asserting that an arm serving one passenger would
    score a WORSE waiting-time Gini than an arm serving three. It failed, and
    the failure was informative rather than a bug.

    Gini measures dispersion. An arm that serves almost nobody produces a
    nearly flat distribution — one served passenger and three charged the same
    penalty — and a flat distribution is an equal one. It scored 0.05 against
    0.44 for the arm that served three with a spread of waits. No choice of
    penalty value repairs this, because uniform misery genuinely is equal.

    So the property that actually holds, and the one the report relies on, is
    about the level rather than the spread: the mean wait over everyone who
    asked is worse for the arm that served fewer. That is why no fairness index
    in this project is ever quoted without the match rate and this mean beside
    it. Inequality and adequacy are separate questions.
    """
    net = grid_network(3, 3, cell_km=1.0)

    def mk_p(pid, origin):
        return Passenger(id=pid, origin=origin, destination=8, v_solo=50.0,
                         v_share=40.0, r_max=10.0, delta_max=10.0)

    # Same four passengers in both markets; only the number of drivers differs,
    # so the difference in outcome is purely how many got served.
    pax = [mk_p(0, 0), mk_p(1, 1), mk_p(2, 2), mk_p(3, 5)]

    def clear_with(n_drivers):
        drv = [Driver(id=i, location=0, capacity=1, cost_per_km=1.0,
                      reservation=0.0) for i in range(n_drivers)]
        groups, _ = enumerate_groups(drv, pax, net)
        res = clear_market(groups, {p.id: p for p in pax},
                           {d.id: d for d in drv})
        return compute_metrics(res.allocation, pax, drv, net)

    few = clear_with(1)    # serves one passenger
    many = clear_with(3)   # serves three

    assert few.matched_passengers < many.matched_passengers

    # The level catches it: serving fewer means a worse expected wait for
    # someone who requested a ride.
    assert few.mean_wait_all_min > many.mean_wait_all_min

    # And the match rate catches it directly, which is why it is reported.
    assert few.match_rate < many.match_rate

    # The Gini does NOT catch it, and this assertion documents that rather
    # than pretending otherwise. If a future change made the fairness index
    # sensitive to abandonment, this line would fail and I would want to know.
    assert few.gini_wait_all < many.gini_wait_all


def test_deficit_is_a_percentage_and_signed_as_a_deficit():
    """Two-sided Clarke is expected to run a deficit, reported as a percentage.

    The percentage form is the one comparable to Kamar & Horvitz's 55-79%.
    Asserting the sign here means every figure downstream inherits it.
    """
    result, pax, drv, net = cleared(0)
    m = compute_metrics(result.allocation, pax, drv, net)

    manual = (100.0 * (m.total_payments - m.total_driver_earnings)
              / m.realised_welfare)
    assert m.deficit_pct_of_welfare == pytest.approx(manual, abs=1e-9)
    assert m.budget_balance < 0, "two-sided Clarke should run a deficit here"


def test_empty_allocation_does_not_raise():
    """A window where nothing cleared must still produce a full row.

    The radius sweep will hit this at small radii, where no group is feasible.
    If it raised, the sweep would die at exactly the left-hand end of the
    headline figure.
    """
    from ora.types import Allocation
    net = grid_network(3, 3)
    pax = [Passenger(id=0, origin=0, destination=8, v_solo=10.0, v_share=8.0,
                     r_max=0.01, delta_max=0.01)]
    drv = [Driver(id=0, location=8, capacity=1, cost_per_km=1.0,
                  reservation=0.0)]
    empty = Allocation(groups=(), payments={}, driver_earnings={})

    m = compute_metrics(empty, pax, drv, net)
    assert m.matched_passengers == 0
    assert m.match_rate == 0.0
    assert m.idle_ratio == 1.0
    assert np.isfinite(m.deficit_pct_of_welfare)
    
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_driver_utility_metrics_match_manual_calculation(seed):
    """Driver utility metrics use allocated drivers and match direct utility.

    Idle drivers are excluded because they receive the outside option of zero.
    The violation rate counts utilities strictly below the numerical tolerance.
    """
    result, pax, drv, net = cleared(seed)
    metrics = compute_metrics(result.allocation, pax, drv, net)

    allocated_driver_ids = {
        group.driver_id
        for group in result.allocation.groups
    }

    utilities = [
        driver_utility(driver, result.allocation)
        for driver in drv
        if driver.id in allocated_driver_ids
    ]

    if utilities:
        expected_violation_rate = (
            sum(utility < -1e-9 for utility in utilities)
            / len(utilities)
        )

        assert metrics.driver_ir_violation_rate == pytest.approx(
            expected_violation_rate
        )
        assert metrics.mean_allocated_driver_utility == pytest.approx(
            np.mean(utilities)
        )
        assert metrics.min_allocated_driver_utility == pytest.approx(
            np.min(utilities)
        )
    else:
        assert metrics.driver_ir_violation_rate == 0.0
        assert metrics.mean_allocated_driver_utility == 0.0
        assert metrics.min_allocated_driver_utility == 0.0