"""Winner determination: exact set packing over feasible hyperedges.

Phase 1 of the mechanism. Given the allocation space that ``grouping`` built,
this module selects the set of non-overlapping hyperedges that maximises the
declared objective exactly.

With ``fairness_lambda == 0`` the objective is the original ORA economic
welfare. With passenger-side historical fairness enabled, each group receives
the public additive bonus

    economic welfare + fairness_lambda * sum(passenger priorities).

The priority vector and fairness weight are fixed before the clearing window
and must be reused unchanged in every Clarke exclusion solve. Feasibility is
not altered: fairness changes group weights only.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import pulp

from ora.fairness import augmented_group_weight
from ora.grouping import group_welfare
from ora.types import Driver, Group, Passenger


def _group_objective_weight(
    group: Group,
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
    priorities: Mapping[int, float] | None,
    fairness_lambda: float,
) -> float:
    """Return the fixed objective coefficient of one feasible group."""
    fairness_lambda = float(fairness_lambda)
    if not math.isfinite(fairness_lambda):
        raise ValueError("fairness_lambda must be finite")
    if fairness_lambda < 0.0:
        raise ValueError("fairness_lambda must be non-negative")

    # This branch preserves the original ORA path exactly. It also means that
    # callers running lambda=0 do not need to manufacture a priority mapping.
    if fairness_lambda == 0.0:
        return group_welfare(group, passengers, drivers)

    if priorities is None:
        raise ValueError(
            "priorities are required when fairness_lambda is positive"
        )

    return augmented_group_weight(
        group,
        passengers,
        drivers,
        priorities,
        fairness_lambda,
    )


def solve_wdp(
    groups: list[Group],
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
    exclude_passenger: int | None = None,
    exclude_driver: int | None = None,
    msg: bool = False,
    priorities: Mapping[int, float] | None = None,
    fairness_lambda: float = 0.0,
) -> tuple[list[Group], float]:
    """Maximise the ORA objective over non-overlapping hyperedges.

    Args:
        groups:
            Exhaustively enumerated feasible groups.
        passengers, drivers:
            Current reported agent lookups.
        exclude_passenger, exclude_driver:
            Optional Clarke counterfactual removal. At most one is normally
            supplied by the payment layer.
        msg:
            Whether CBC should print solver output.
        priorities:
            Public passenger-priority mapping fixed before the current window.
            Required only when ``fairness_lambda`` is positive.
        fairness_lambda:
            Non-negative weight of the public historical-fairness bonus.

    Returns:
        ``(winning_groups, optimal_objective)``. The second value is economic
        welfare when lambda is zero and the augmented objective otherwise.
    """
    fairness_lambda = float(fairness_lambda)
    if not math.isfinite(fairness_lambda):
        raise ValueError("fairness_lambda must be finite")
    if fairness_lambda < 0.0:
        raise ValueError("fairness_lambda must be non-negative")
    if fairness_lambda > 0.0 and priorities is None:
        raise ValueError(
            "priorities are required when fairness_lambda is positive"
        )

    # Remove an excluded agent from the market by removing every edge that
    # contains that agent. This is the same operation in the main and fairness
    # variants; only edge weights differ.
    active = [
        group
        for group in groups
        if exclude_passenger not in group.passenger_ids
        and group.driver_id != exclude_driver
    ]

    # The empty allocation is always feasible and has objective zero.
    if not active:
        return [], 0.0

    problem = pulp.LpProblem("wdp", pulp.LpMaximize)
    selected = {
        index: pulp.LpVariable(f"x_{index}", cat="Binary")
        for index in range(len(active))
    }

    problem += pulp.lpSum(
        _group_objective_weight(
            group,
            passengers,
            drivers,
            priorities,
            fairness_lambda,
        )
        * selected[index]
        for index, group in enumerate(active)
    )

    # Each passenger appears in at most one selected group.
    for passenger_id in passengers:
        if passenger_id == exclude_passenger:
            continue
        member_indices = [
            index
            for index, group in enumerate(active)
            if passenger_id in group.passenger_ids
        ]
        if member_indices:
            problem += pulp.lpSum(
                selected[index] for index in member_indices
            ) <= 1

    # Each driver serves at most one selected group.
    for driver_id in drivers:
        if driver_id == exclude_driver:
            continue
        member_indices = [
            index
            for index, group in enumerate(active)
            if group.driver_id == driver_id
        ]
        if member_indices:
            problem += pulp.lpSum(
                selected[index] for index in member_indices
            ) <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=msg))

    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(
            "WDP not solved to optimality: "
            f"{pulp.LpStatus[problem.status]}"
        )

    winners = [
        group
        for index, group in enumerate(active)
        if selected[index].value() > 0.5
    ]

    objective = pulp.value(problem.objective)
    return winners, float(objective if objective is not None else 0.0)
