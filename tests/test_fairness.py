"""Unit tests for ora.fairness.

These tests cover only the pure building blocks of the passenger-side
historical-fairness extension. They deliberately do not modify or test the WDP
or Groves payments yet.
"""

import pytest

from ora.fairness import (
    DEFAULT_RHO,
    allocation_fairness_bonus,
    augmented_group_weight,
    group_fairness_bonus,
    update_priority,
)
from ora.grouping import group_welfare
from ora.types import Allocation, Driver, Group, Mode, Passenger


def passenger(passenger_id: int, value: float = 20.0) -> Passenger:
    return Passenger(
        id=passenger_id,
        origin=0,
        destination=1,
        v_solo=value,
        v_share=0.75 * value,
        r_max=5.0,
        delta_max=5.0,
    )


def driver(driver_id: int = 0) -> Driver:
    return Driver(
        id=driver_id,
        location=0,
        capacity=3,
        cost_per_km=1.0,
        reservation=2.0,
    )


def group(
    passenger_ids: tuple[int, ...],
    *,
    driver_id: int = 0,
    mode: Mode | None = None,
    route_cost: float = 3.0,
) -> Group:
    if mode is None:
        mode = Mode.SOLO if len(passenger_ids) == 1 else Mode.SHARED

    return Group(
        driver_id=driver_id,
        passenger_ids=passenger_ids,
        mode=mode,
        route_km=route_cost,
        route_cost=route_cost,
        pickup_km=tuple(0.5 for _ in passenger_ids),
        detour_km=tuple(0.0 for _ in passenger_ids),
    )


# ---------------------------------------------------------------------------
# Historical-state update
# ---------------------------------------------------------------------------

def test_new_passenger_can_start_at_zero():
    assert update_priority(
        0.0,
        requested=False,
        served=False,
    ) == pytest.approx(0.0)


def test_unserved_request_increases_priority_from_zero():
    assert update_priority(
        0.0,
        requested=True,
        served=False,
    ) == pytest.approx(0.2)


def test_unserved_request_updates_priority_with_memory():
    assert update_priority(
        0.5,
        requested=True,
        served=False,
    ) == pytest.approx(0.6)


def test_served_request_reduces_priority():
    assert update_priority(
        0.5,
        requested=True,
        served=True,
    ) == pytest.approx(0.4)


def test_inactive_passenger_keeps_priority():
    assert update_priority(
        0.5,
        requested=False,
        served=False,
    ) == pytest.approx(0.5)


def test_default_memory_parameter_is_the_design_value():
    assert DEFAULT_RHO == pytest.approx(0.8)


def test_priority_remains_in_unit_interval_over_repeated_updates():
    q = 0.0

    for _ in range(100):
        q = update_priority(q, requested=True, served=False)
        assert 0.0 <= q <= 1.0

    for _ in range(100):
        q = update_priority(q, requested=True, served=True)
        assert 0.0 <= q <= 1.0


@pytest.mark.parametrize(
    ("current_priority", "rho"),
    [
        (-0.01, 0.8),
        (1.01, 0.8),
        (0.5, -0.01),
        (0.5, 1.01),
    ],
)
def test_invalid_priority_parameters_raise(current_priority, rho):
    with pytest.raises(ValueError):
        update_priority(
            current_priority,
            requested=True,
            served=False,
            rho=rho,
        )


def test_served_without_requesting_is_rejected():
    with pytest.raises(ValueError):
        update_priority(
            0.5,
            requested=False,
            served=True,
        )


# ---------------------------------------------------------------------------
# Fairness bonus
# ---------------------------------------------------------------------------

def test_group_bonus_is_sum_of_passenger_priorities():
    candidate = group((0, 1))
    priorities = {0: 0.2, 1: 0.7, 2: 0.4}

    assert group_fairness_bonus(
        candidate,
        priorities,
    ) == pytest.approx(0.9)


def test_group_bonus_requires_every_priority_to_be_explicit():
    candidate = group((0, 1))

    with pytest.raises(KeyError):
        group_fairness_bonus(candidate, {0: 0.2})


def test_allocation_bonus_sums_selected_groups():
    first = group((0, 1), driver_id=0)
    second = group((2,), driver_id=1)
    allocation = Allocation(
        groups=(first, second),
        payments={},
        driver_earnings={},
    )
    priorities = {0: 0.2, 1: 0.7, 2: 0.4}

    assert allocation_fairness_bonus(
        allocation,
        priorities,
    ) == pytest.approx(1.3)


def test_empty_allocation_has_zero_fairness_bonus():
    allocation = Allocation(
        groups=(),
        payments={},
        driver_earnings={},
    )

    assert allocation_fairness_bonus(
        allocation,
        {},
    ) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Augmented group weight
# ---------------------------------------------------------------------------

def test_lambda_zero_returns_exact_economic_welfare():
    p0 = passenger(0)
    d0 = driver(0)
    candidate = group((0,), route_cost=3.0)

    passengers = {p0.id: p0}
    drivers = {d0.id: d0}
    priorities = {p0.id: 0.8}

    expected = group_welfare(
        candidate,
        passengers,
        drivers,
    )
    actual = augmented_group_weight(
        candidate,
        passengers,
        drivers,
        priorities,
        fairness_lambda=0.0,
    )

    assert actual == pytest.approx(expected)


def test_augmented_weight_adds_lambda_times_bonus():
    p0 = passenger(0)
    p1 = passenger(1)
    d0 = driver(0)
    candidate = group((0, 1), route_cost=3.0)

    passengers = {p0.id: p0, p1.id: p1}
    drivers = {d0.id: d0}
    priorities = {0: 0.2, 1: 0.7}

    economic = group_welfare(
        candidate,
        passengers,
        drivers,
    )
    actual = augmented_group_weight(
        candidate,
        passengers,
        drivers,
        priorities,
        fairness_lambda=5.0,
    )

    assert actual == pytest.approx(
        economic + 5.0 * 0.9
    )


def test_negative_lambda_is_rejected():
    p0 = passenger(0)
    d0 = driver(0)
    candidate = group((0,))

    with pytest.raises(ValueError):
        augmented_group_weight(
            candidate,
            {p0.id: p0},
            {d0.id: d0},
            {p0.id: 0.5},
            fairness_lambda=-1.0,
        )
