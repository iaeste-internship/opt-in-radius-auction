"""Passenger-side historical fairness for ORA.

This module contains only pure functions. It does not store state globally and
does not modify passengers, groups, allocations, or the economic-welfare
definition.

The public priority of passenger i before clearing window t is q_i^t in [0, 1].
Priorities must be fixed before current private reports are processed. After
the window clears, the caller updates each active passenger once:

    served:    q_i^(t+1) = rho * q_i^t
    unserved:  q_i^(t+1) = rho * q_i^t + (1 - rho)
    inactive:  q_i^(t+1) = q_i^t

The fairness bonus is additive across passengers. For a feasible group g,

    B_t(g) = sum(q_i^t for i in P(g))

and its augmented weight is

    W_tilde(g) = W_econ(g) + lambda * B_t(g).

Economic welfare remains defined by ora.grouping.group_welfare.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

from ora.grouping import group_welfare
from ora.types import Allocation, Driver, Group, Passenger


DEFAULT_RHO = 0.8


def _validate_unit_interval(value: float, name: str) -> float:
    """Return value as float after checking that it lies in [0, 1]."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


def update_priority(
    current_priority: float,
    *,
    requested: bool,
    served: bool,
    rho: float = DEFAULT_RHO,
) -> float:
    """Update one passenger's public priority after a clearing window.

    Args:
        current_priority:
            Passenger priority q_i^t before the window.
        requested:
            Whether the passenger submitted a valid request in the window.
        served:
            Whether the passenger was allocated. A served passenger must also
            have requested.
        rho:
            Memory parameter in [0, 1]. The initial design uses 0.8.

    Returns:
        The next-window priority q_i^(t+1), guaranteed to lie in [0, 1].

    Raises:
        ValueError:
            If a priority or rho is outside [0, 1], is not finite, or the
            logically inconsistent combination served=True, requested=False is
            supplied.
    """
    q = _validate_unit_interval(current_priority, "current_priority")
    rho = _validate_unit_interval(rho, "rho")

    if served and not requested:
        raise ValueError("a passenger cannot be served without requesting")

    if not requested:
        return q

    if served:
        updated = rho * q
    else:
        updated = rho * q + (1.0 - rho)

    # The formula is mathematically bounded. The clamp only removes possible
    # floating-point noise at the endpoints.
    return min(1.0, max(0.0, float(updated)))


def group_fairness_bonus(
    group: Group,
    priorities: Mapping[int, float],
) -> float:
    """Return the sum of public priorities of passengers in one group.

    Every passenger appearing in the group must have an explicit priority
    entry. New passengers should be inserted in the mapping with priority 0.0
    rather than being silently omitted.
    """
    bonus = 0.0

    for passenger_id in group.passenger_ids:
        if passenger_id not in priorities:
            raise KeyError(
                f"missing public priority for passenger {passenger_id}"
            )
        bonus += _validate_unit_interval(
            priorities[passenger_id],
            f"priority[{passenger_id}]",
        )

    return float(bonus)


def allocation_fairness_bonus(
    allocation: Allocation,
    priorities: Mapping[int, float],
) -> float:
    """Return the total fairness bonus of a selected allocation."""
    return float(
        sum(
            group_fairness_bonus(group, priorities)
            for group in allocation.groups
        )
    )


def augmented_group_weight(
    group: Group,
    passengers: Mapping[int, Passenger],
    drivers: Mapping[int, Driver],
    priorities: Mapping[int, float],
    fairness_lambda: float,
) -> float:
    """Return economic welfare plus the public fairness bonus.

    This function does not alter ora.grouping.group_welfare. It combines that
    economic quantity with the public allocation-dependent term used by the
    fairness-aware WDP.

    Args:
        group:
            A feasible candidate hyperedge.
        passengers:
            Passenger lookup used by group_welfare.
        drivers:
            Driver lookup used by group_welfare.
        priorities:
            Public priority q_i^t for each passenger in the current window.
        fairness_lambda:
            Non-negative weight applied to the fairness bonus.

    Returns:
        W_econ(g) + fairness_lambda * B_t(g).
    """
    fairness_lambda = float(fairness_lambda)

    if not math.isfinite(fairness_lambda):
        raise ValueError("fairness_lambda must be finite")
    if fairness_lambda < 0.0:
        raise ValueError(
            f"fairness_lambda must be non-negative, got {fairness_lambda}"
        )

    economic_welfare = group_welfare(
        group,
        dict(passengers),
        dict(drivers),
    )
    fairness_bonus = group_fairness_bonus(group, priorities)

    return float(
        economic_welfare
        + fairness_lambda * fairness_bonus
    )
