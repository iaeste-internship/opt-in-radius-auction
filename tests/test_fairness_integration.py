"""Integration tests for fairness-aware WDP and Groves payments."""

import pytest

from ora.payments import clear_market, driver_utility, passenger_utility
from ora.types import Driver, Group, Mode, Passenger
from ora.wdp import solve_wdp

TOL = 1e-8


def passenger(passenger_id: int, value: float) -> Passenger:
    return Passenger(
        id=passenger_id,
        origin=0,
        destination=1,
        v_solo=value,
        v_share=value,
        r_max=10.0,
        delta_max=10.0,
    )


def driver(driver_id: int = 0) -> Driver:
    return Driver(
        id=driver_id,
        location=0,
        capacity=1,
        cost_per_km=0.0,
        reservation=0.0,
    )


def solo_group(passenger_id: int, driver_id: int = 0) -> Group:
    return Group(
        driver_id=driver_id,
        passenger_ids=(passenger_id,),
        mode=Mode.SOLO,
        route_km=0.0,
        route_cost=0.0,
        pickup_km=(0.0,),
        detour_km=(0.0,),
    )


def test_lambda_zero_reproduces_original_wdp_exactly():
    passengers = {
        0: passenger(0, 10.0),
        1: passenger(1, 8.0),
    }
    drivers = {0: driver()}
    groups = [solo_group(0), solo_group(1)]

    original_winners, original_objective = solve_wdp(
        groups,
        passengers,
        drivers,
    )
    fair_winners, fair_objective = solve_wdp(
        groups,
        passengers,
        drivers,
        priorities={0: 0.0, 1: 1.0},
        fairness_lambda=0.0,
    )

    assert fair_winners == original_winners
    assert fair_objective == pytest.approx(original_objective, abs=TOL)


def test_priority_changes_winner_above_theoretical_threshold():
    # Economic scores: A=10 and B=8. Priorities: q_A=0 and q_B=1.
    # B wins exactly above lambda=(10-8)/(1-0)=2.
    passengers = {
        0: passenger(0, 10.0),
        1: passenger(1, 8.0),
    }
    drivers = {0: driver()}
    groups = [solo_group(0), solo_group(1)]
    priorities = {0: 0.0, 1: 1.0}

    below, _ = solve_wdp(
        groups,
        passengers,
        drivers,
        priorities=priorities,
        fairness_lambda=1.9,
    )
    above, _ = solve_wdp(
        groups,
        passengers,
        drivers,
        priorities=priorities,
        fairness_lambda=2.1,
    )

    assert below[0].passenger_ids == (0,)
    assert above[0].passenger_ids == (1,)


def test_clearing_result_separates_economic_and_augmented_objectives():
    passengers = {
        0: passenger(0, 10.0),
        1: passenger(1, 8.0),
    }
    drivers = {0: driver()}
    groups = [solo_group(0), solo_group(1)]

    result = clear_market(
        groups,
        passengers,
        drivers,
        priorities={0: 0.0, 1: 1.0},
        fairness_lambda=3.0,
    )

    assert result.allocation.groups[0].passenger_ids == (1,)
    assert result.welfare == pytest.approx(8.0, abs=TOL)
    assert result.fairness_bonus == pytest.approx(1.0, abs=TOL)
    assert result.augmented_objective == pytest.approx(11.0, abs=TOL)


def test_augmented_passenger_payment_matches_hand_calculation():
    # With lambda=3 passenger 1 wins: 8 + 3 = 11 > 10.
    # Without passenger 1, passenger 0 creates augmented objective 10.
    # Others with passenger 1 retain the public fairness term 3, so p_1=10-3=7.
    passengers = {
        0: passenger(0, 10.0),
        1: passenger(1, 8.0),
    }
    drivers = {0: driver()}
    groups = [solo_group(0), solo_group(1)]

    result = clear_market(
        groups,
        passengers,
        drivers,
        priorities={0: 0.0, 1: 1.0},
        fairness_lambda=3.0,
    )

    assert result.allocation.payments[1] == pytest.approx(7.0, abs=TOL)
    assert passenger_utility(
        passengers[1],
        result.allocation,
    ) == pytest.approx(1.0, abs=TOL)


def test_augmented_driver_earnings_match_hand_calculation():
    # The winning passenger contributes value 8 and public credit 3 to the
    # others-with-driver term. With no driver the objective is zero.
    passengers = {
        0: passenger(0, 10.0),
        1: passenger(1, 8.0),
    }
    drivers = {0: driver()}
    groups = [solo_group(0), solo_group(1)]

    result = clear_market(
        groups,
        passengers,
        drivers,
        priorities={0: 0.0, 1: 1.0},
        fairness_lambda=3.0,
    )

    assert result.allocation.driver_earnings[0] == pytest.approx(
        11.0,
        abs=TOL,
    )
    assert driver_utility(
        drivers[0],
        result.allocation,
    ) == pytest.approx(11.0, abs=TOL)


def test_negative_passenger_payment_is_preserved():
    # No alternative allocation exists. The passenger's public priority credit
    # remains in the others-with term, producing a subsidy of lambda*q = 3.
    passengers = {0: passenger(0, 10.0)}
    drivers = {0: driver()}
    groups = [solo_group(0)]

    result = clear_market(
        groups,
        passengers,
        drivers,
        priorities={0: 1.0},
        fairness_lambda=3.0,
    )

    assert result.allocation.payments[0] == pytest.approx(-3.0, abs=TOL)
    assert passenger_utility(
        passengers[0],
        result.allocation,
    ) == pytest.approx(13.0, abs=TOL)


def test_positive_lambda_requires_priorities():
    passengers = {0: passenger(0, 10.0)}
    drivers = {0: driver()}

    with pytest.raises(ValueError):
        solve_wdp(
            [solo_group(0)],
            passengers,
            drivers,
            fairness_lambda=1.0,
        )


def test_lambda_zero_reproduces_original_payments():
    passengers = {
        0: passenger(0, 10.0),
        1: passenger(1, 8.0),
    }
    drivers = {0: driver()}
    groups = [solo_group(0), solo_group(1)]

    original = clear_market(groups, passengers, drivers)
    fair_zero = clear_market(
        groups,
        passengers,
        drivers,
        priorities={0: 0.3, 1: 0.9},
        fairness_lambda=0.0,
    )

    assert fair_zero.allocation == original.allocation
    assert fair_zero.welfare == pytest.approx(original.welfare, abs=TOL)
    assert fair_zero.augmented_objective == pytest.approx(
        original.augmented_objective,
        abs=TOL,
    )
