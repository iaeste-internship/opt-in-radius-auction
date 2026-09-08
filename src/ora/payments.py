"""Clarke pivot payments on both sides of the ORA market.

The payment layer must use exactly the same objective as the allocation layer.
When historical passenger fairness is disabled, this is the original ORA
reported economic welfare. When fairness is enabled, the main solve and every
agent-exclusion solve maximise

    economic welfare + fairness_lambda * served public priority.

The fairness term is public and allocation-dependent. It therefore remains in
the Groves ``others`` term; only the excluded agent's own economic contribution
is removed. In particular, a served passenger's own public priority bonus is
not subtracted from that term. This can produce a negative passenger payment
(a subsidy), which must not be clipped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ora.fairness import group_fairness_bonus
from ora.grouping import group_welfare
from ora.types import Allocation, Driver, Group, Mode, Passenger
from ora.wdp import solve_wdp


@dataclass(frozen=True)
class ClearingResult:
    """Everything produced by one clearing window.

    ``welfare`` remains the economic welfare of the selected allocation for
    backward compatibility with the current metrics and experiments.
    ``augmented_objective`` is the objective actually maximised by the WDP.
    They coincide when fairness_lambda is zero.
    """

    allocation: Allocation
    welfare: float
    n_exclusion_solves: int
    fairness_bonus: float = 0.0
    augmented_objective: float = 0.0


def _served_passengers(winners: list[Group]) -> set[int]:
    return {
        passenger_id
        for group in winners
        for passenger_id in group.passenger_ids
    }


def _served_drivers(winners: list[Group]) -> set[int]:
    return {group.driver_id for group in winners}


def _economic_welfare_of(
    winners: list[Group],
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
) -> float:
    """Economic welfare of the selected groups, excluding fairness credits."""
    return float(
        sum(
            group_welfare(group, passengers, drivers)
            for group in winners
        )
    )


def _fairness_bonus_of(
    winners: list[Group],
    priorities: Mapping[int, float] | None,
) -> float:
    """Public fairness credit served by the selected groups."""
    if priorities is None:
        return 0.0
    return float(
        sum(
            group_fairness_bonus(group, priorities)
            for group in winners
        )
    )


def _others_objective_with_agent(
    winners: list[Group],
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
    priorities: Mapping[int, float] | None,
    fairness_lambda: float,
    passenger_id: int | None = None,
    driver_id: int | None = None,
) -> float:
    """Augmented objective of the actual allocation, excluding one agent's
    own economic contribution.

    The public fairness term is retained in full. For a passenger this means
    subtracting her reported mode value but retaining every selected priority
    credit, including her own. For a driver, the route cost and reservation are
    the driver's economic contribution, so the relevant edge contributes the
    passengers' values plus its public fairness bonus.
    """
    total = 0.0

    for group in winners:
        public_term = 0.0
        if fairness_lambda != 0.0:
            if priorities is None:
                raise ValueError(
                    "priorities are required when fairness_lambda is positive"
                )
            public_term = (
                fairness_lambda
                * group_fairness_bonus(group, priorities)
            )

        if driver_id is not None and group.driver_id == driver_id:
            # Keep all passenger values and the public fairness credit, while
            # removing the selected driver's route cost and reservation.
            for current_passenger_id in group.passenger_ids:
                passenger = passengers[current_passenger_id]
                total += (
                    passenger.v_solo
                    if group.mode is Mode.SOLO
                    else passenger.v_share
                )
            total += public_term
            continue

        edge = group_welfare(group, passengers, drivers) + public_term

        if (
            passenger_id is not None
            and passenger_id in group.passenger_ids
        ):
            # Remove only the passenger's private economic value. The public
            # fairness term is not the passenger's valuation and stays in the
            # Groves others term.
            passenger = passengers[passenger_id]
            edge -= (
                passenger.v_solo
                if group.mode is Mode.SOLO
                else passenger.v_share
            )

        total += edge

    return float(total)


def clear_market(
    groups: list[Group],
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
    msg: bool = False,
    priorities: Mapping[int, float] | None = None,
    fairness_lambda: float = 0.0,
) -> ClearingResult:
    """Allocate and price one ORA clearing window.

    The exact same ``priorities`` mapping and ``fairness_lambda`` are passed to
    the main solve and every Clarke counterfactual. The caller must treat the
    mapping as immutable for the duration of this function.
    """
    winners, augmented_objective = solve_wdp(
        groups,
        passengers,
        drivers,
        msg=msg,
        priorities=priorities,
        fairness_lambda=fairness_lambda,
    )
    solves = 1

    payments: dict[int, float] = {}
    driver_earnings: dict[int, float] = {}

    # Passenger side: payment to the platform equals the augmented objective
    # available to others without the passenger minus the augmented objective
    # received by others in the selected allocation.
    for passenger_id in _served_passengers(winners):
        objective_without, _ = _objective_without(
            groups,
            passengers,
            drivers,
            exclude_passenger=passenger_id,
            msg=msg,
            priorities=priorities,
            fairness_lambda=fairness_lambda,
        )
        solves += 1

        others_with = _others_objective_with_agent(
            winners,
            passengers,
            drivers,
            priorities,
            fairness_lambda,
            passenger_id=passenger_id,
        )
        payments[passenger_id] = objective_without - others_with

    # Driver side: same pivot, with the project's opposite money-flow sign.
    for driver_id in _served_drivers(winners):
        objective_without, _ = _objective_without(
            groups,
            passengers,
            drivers,
            exclude_driver=driver_id,
            msg=msg,
            priorities=priorities,
            fairness_lambda=fairness_lambda,
        )
        solves += 1

        others_with = _others_objective_with_agent(
            winners,
            passengers,
            drivers,
            priorities,
            fairness_lambda,
            driver_id=driver_id,
        )
        driver_earnings[driver_id] = others_with - objective_without

    allocation = Allocation(
        groups=tuple(winners),
        payments=payments,
        driver_earnings=driver_earnings,
    )

    economic_welfare = _economic_welfare_of(
        winners,
        passengers,
        drivers,
    )
    fairness_bonus = _fairness_bonus_of(winners, priorities)

    return ClearingResult(
        allocation=allocation,
        welfare=economic_welfare,
        n_exclusion_solves=solves,
        fairness_bonus=fairness_bonus,
        augmented_objective=augmented_objective,
    )


def _objective_without(
    groups,
    passengers,
    drivers,
    exclude_passenger=None,
    exclude_driver=None,
    msg=False,
    priorities=None,
    fairness_lambda=0.0,
):
    """Solve one agent-exclusion counterfactual with the same objective."""
    winners, objective = solve_wdp(
        groups,
        passengers,
        drivers,
        exclude_passenger=exclude_passenger,
        exclude_driver=exclude_driver,
        msg=msg,
        priorities=priorities,
        fairness_lambda=fairness_lambda,
    )
    return objective, winners


def passenger_utility(
    passenger: Passenger,
    allocation: Allocation,
) -> float:
    """Passenger's quasi-linear utility: allocated value minus payment."""
    for group in allocation.groups:
        if passenger.id in group.passenger_ids:
            value = (
                passenger.v_solo
                if group.mode is Mode.SOLO
                else passenger.v_share
            )
            return value - allocation.payments.get(passenger.id, 0.0)
    return 0.0


def driver_utility(driver: Driver, allocation: Allocation) -> float:
    """Driver earnings minus route cost and true reservation value."""
    for group in allocation.groups:
        if group.driver_id == driver.id:
            return (
                allocation.driver_earnings.get(driver.id, 0.0)
                - group.route_cost
                - driver.reservation
            )
    return 0.0


def budget_balance(allocation: Allocation) -> float:
    """Platform surplus: passenger payments minus driver earnings."""
    return (
        sum(allocation.payments.values())
        - sum(allocation.driver_earnings.values())
    )
