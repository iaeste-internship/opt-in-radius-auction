"""Tests for the comparison arms.

These validate the claims the results table makes, not just that the code runs.
Three of them correspond to sentences I intend to write in the report, and if
one fails the sentence is false:

  - ORA never collects a passenger beyond her reported radius.
  - An arm using a global radius does, when that radius exceeds what
    passengers reported.
  - Allocating on assumed types and measuring on true ones gives a different
    welfare figure than measuring on the assumed ones, which is why the
    measurement convention in baselines.py is load-bearing rather than
    cosmetic.

The last one matters most. If measuring with assumed types gave the same
answer, the ablation would be scoring itself against its own fiction and the
comparison would be empty.
"""
import numpy as np
import pytest

from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.metrics import compute_metrics
from ora.baselines import (ablation_passengers, run_ablation,
                           radius_violations, DEFAULT_RADIUS_KM,
                           DEFAULT_SHARE_DISCOUNT)
from ora.types import Passenger, Driver, Mode

SEEDS = [0, 1, 2, 3, 4]


def market(seed, n_passengers=12, n_drivers=4):
    net = grid_network(6, 6)
    rng = np.random.default_rng(seed)
    pax, drv = generate(DemandParams(n_passengers=n_passengers,
                                     n_drivers=n_drivers, capacity=3),
                        net, rng)
    return net, pax, drv


def run_ora(net, pax, drv):
    groups, _ = enumerate_groups(drv, pax, net)
    return clear_market(groups, {p.id: p for p in pax},
                        {d.id: d for d in drv})


# --------------------------------------------------------------------------
# The type substitution itself.
# --------------------------------------------------------------------------

def test_ablation_overwrites_only_the_two_elicited_components():
    """r_max and v_share are replaced; everything else survives untouched.

    delta_max in particular must survive, because the ablation is meant to
    isolate the pooling decision and the search radius. If it also overwrote
    the detour tolerance I could not attribute any difference to the two
    components the contribution is about.
    """
    _, pax, _ = market(0)
    assumed = ablation_passengers(pax, radius_km=2.0, share_discount=0.75)

    assert len(assumed) == len(pax)
    for original, a in zip(pax, assumed):
        assert a.r_max == 2.0
        assert a.v_share == pytest.approx(original.v_solo * 0.75)
        # untouched
        assert a.id == original.id
        assert a.origin == original.origin
        assert a.destination == original.destination
        assert a.v_solo == original.v_solo
        assert a.delta_max == original.delta_max


def test_ablation_does_not_mutate_the_real_passengers():
    """The assumed types must not leak back into the true population.

    Passenger is frozen and dataclasses.replace returns a copy, so this holds
    by construction. Tested anyway because if it ever stopped holding, every
    later measurement in the same run would silently use the wrong valuations.
    """
    _, pax, _ = market(0)
    before = [(p.r_max, p.v_share) for p in pax]
    ablation_passengers(pax)
    after = [(p.r_max, p.v_share) for p in pax]
    assert before == after


# --------------------------------------------------------------------------
# The radius claim. This is the headline of the comparison.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_ora_never_violates_a_reported_radius(seed):
    """The claim the report makes about ORA, tested rather than asserted.

    Zero by construction: radius_ok runs inside the router against the
    realised boarding distance. If this ever fails, the elicitation delivers
    nothing and the contribution is empty.
    """
    net, pax, drv = market(seed)
    result = run_ora(net, pax, drv)
    count, rate = radius_violations(result.allocation, pax)
    assert count == 0, f"ORA violated {count} reported radii on seed {seed}"
    assert rate == 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_global_radius_violates_when_it_exceeds_what_was_reported(seed):
    """A generous global radius collects people beyond their stated bound.

    This is the harm the elicitation prevents, and it must be demonstrated
    rather than argued. The radius here is deliberately well above the demand
    generator's mean r_max so the effect is unambiguous.

    Asserting "greater than zero somewhere across the seeds" rather than on
    every individual seed: on a small market a particular seed can happen to
    produce no violation, and requiring one per seed would make the test
    flaky for reasons that say nothing about the mechanism.
    """
    net, pax, drv = market(seed)
    result = run_ablation(pax, drv, net, radius_km=4.0, share_discount=0.75)
    count, _ = radius_violations(result.allocation, pax)
    # Collected for the aggregate assertion below.
    test_global_radius_violates_when_it_exceeds_what_was_reported.counts.append(count)


test_global_radius_violates_when_it_exceeds_what_was_reported.counts = []


def test_global_radius_violates_somewhere_across_the_seeds():
    """Aggregate of the per-seed run above.

    Runs after the parametrised test has populated the counter. If the whole
    set came out clean, the ablation is not actually reaching past anyone's
    limit and the comparison in the results table has no content.
    """
    counts = test_global_radius_violates_when_it_exceeds_what_was_reported.counts
    assert counts, "parametrised test did not run first"
    assert sum(counts) > 0, ("a 4 km global radius violated nobody's reported "
                             "r_max, so the ablation is not demonstrating the "
                             "effect the results table claims")


def test_a_tight_global_radius_violates_nobody():
    """The control. A radius below every reported bound cannot violate one.

    Without this, the previous test could be passing because of a bug in
    radius_violations rather than because of the radius, and the whole
    comparison would rest on a broken counter.
    """
    net, pax, drv = market(0)
    tight = min(p.r_max for p in pax)
    result = run_ablation(pax, drv, net, radius_km=tight,
                          share_discount=0.75)
    count, _ = radius_violations(result.allocation, pax)
    assert count == 0


def test_radius_violations_counts_a_hand_built_violation():
    """Direct test of the counter on a market built to violate.

    One passenger with a very small reported radius, one driver far away, and
    a global radius large enough to reach her. She should be served and
    counted as violated. If this fails the counter is broken and every
    violation figure in the report is wrong.
    """
    net = grid_network(3, 3, cell_km=1.0)
    p = Passenger(id=0, origin=8, destination=2, v_solo=100.0, v_share=80.0,
                  r_max=0.5, delta_max=20.0)
    d = Driver(id=0, location=0, capacity=1, cost_per_km=0.1, reservation=0.0)

    # Under ORA she is unreachable: 4 km away, radius 0.5.
    ora = run_ora(net, [p], [d])
    assert radius_violations(ora.allocation, [p])[0] == 0
    assert ora.allocation.groups == ()

    # Under a 10 km global radius she is collected, in violation of her report.
    abl = run_ablation([p], [d], net, radius_km=10.0, share_discount=0.75)
    assert len(abl.allocation.groups) == 1
    count, rate = radius_violations(abl.allocation, [p])
    assert count == 1
    assert rate == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The measurement convention. Allocate on assumed, measure on true.
# --------------------------------------------------------------------------

def test_measuring_on_true_types_differs_from_measuring_on_assumed():
    """The convention that makes the ablation meaningful.

    The ablation assumes one discount for everyone. Measured against its own
    assumption it looks like it delivered what it planned. Measured against
    what passengers actually valued, it does not. If these two numbers were
    equal the convention would be cosmetic and the ablation would be scoring
    itself against its own fiction.

    Getting the setup right took two attempts, and the first failure was
    instructive in the same way the low-tau test was. I first assumed a
    discount of 0.30, far BELOW the generator's mean tau of 0.75. That made
    pooling look so bad that the optimiser chose SOLO for every group, and a
    SOLO group's welfare uses v_solo, which is identical in both populations.
    True and assumed welfare came out exactly equal and the test failed for a
    reason that had nothing to do with the convention.

    So the discount has to sit ABOVE the true mean: the platform overestimates
    how well passengers tolerate sharing, pools them accordingly, and the two
    welfare figures then diverge because v_share is actually being used. That
    is also the more realistic direction of error, and it gives the ablation
    the flattering-itself behaviour the comparison is designed to expose: the
    platform believes it created more value than it did.

    The pooling assertion below is what stops this test passing vacuously a
    second time.
    """
    net, pax, drv = market(0)
    assumed_discount = 0.95          # generator's tau_mean is 0.75
    result = run_ablation(pax, drv, net, radius_km=DEFAULT_RADIUS_KM,
                          share_discount=assumed_discount)

    pooled = any(g.mode is Mode.SHARED for g in result.allocation.groups)
    assert pooled, ("nothing was pooled, so v_share never entered the welfare "
                    "sum and this test proves nothing about the convention")

    assumed = ablation_passengers(pax, DEFAULT_RADIUS_KM, assumed_discount)

    true_welfare = compute_metrics(result.allocation, pax, drv,
                                   net).realised_welfare
    assumed_welfare = compute_metrics(result.allocation, assumed, drv,
                                      net).realised_welfare

    assert true_welfare != pytest.approx(assumed_welfare), (
        "measuring on true and assumed types gave the same welfare, so the "
        "measurement convention in baselines.py is doing nothing")
    # The platform assumed passengers tolerate sharing better than they do, so
    # it believes it created more welfare than it actually did.
    assert assumed_welfare > true_welfare


@pytest.mark.parametrize("seed", SEEDS)
def test_ablation_allocation_uses_real_passenger_ids(seed):
    """The assumed types must not escape the arm.

    Everything downstream looks passengers up by id, so an allocation carrying
    ids from a substituted list would either raise or, worse, silently match
    the wrong person.
    """
    net, pax, drv = market(seed)
    result = run_ablation(pax, drv, net)
    real_ids = {p.id for p in pax}
    for g in result.allocation.groups:
        for pid in g.passenger_ids:
            assert pid in real_ids


@pytest.mark.parametrize("seed", SEEDS)
def test_ablation_output_is_measurable_by_the_metrics_layer(seed):
    """Both arms must go through identical measurement code.

    That is the whole reason Allocation exists as a shared type. If the metrics
    layer choked on one arm's output, the comparison would need arm-specific
    handling and would stop being honest.
    """
    net, pax, drv = market(seed)
    for allocation in (run_ora(net, pax, drv).allocation,
                       run_ablation(pax, drv, net).allocation):
        m = compute_metrics(allocation, pax, drv, net)
        for name, value in m.as_row().items():
            assert np.isfinite(value), f"{name} not finite on seed {seed}"


def test_default_radius_is_above_the_generator_mean():
    """A guard on the comparison's premise, not on the code.

    The results table reports that the ablation reaches past reported bounds.
    That only happens because the literature's default radius (2.0 km) exceeds
    the mean r_max the demand generator produces (1.5 km). If either constant
    is ever changed so that no longer holds, the violation column would go to
    zero and the table's story would quietly stop being true. This test fails
    loudly at that moment instead.
    """
    assert DEFAULT_RADIUS_KM > DemandParams(n_passengers=1,
                                            n_drivers=1).r_max_mean, (
        "the ablation radius no longer exceeds the mean reported r_max, so "
        "the violation result in the table needs rechecking")
    assert 0.0 < DEFAULT_SHARE_DISCOUNT <= 1.0
