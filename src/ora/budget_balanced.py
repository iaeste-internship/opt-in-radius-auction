"""Budget-balanced comparators for ORA (Experiment E).

ORA keeps exact efficiency, dominant-strategy incentive compatibility and
ex-post individual rationality on both sides, and pays for it with a platform
deficit. By Myerson-Satterthwaite all four properties cannot hold together, so
the honest comparison is against mechanisms that give up something else to
reach budget balance. Two are implemented here.

Threshold rule (TB)
-------------------
Parkes, Kalagnanam and Eso (IJCAI 2001). The allocation is ORA's, so welfare is
unchanged. Under Clarke payments every allocated agent k keeps her Vickrey
discount, Delta_k = W* - H_k, which is also her utility. Because total surplus
is split between agents and platform,

    B = W* - sum_k Delta_k,

so a deficit means the discounts add up to more than the welfare created. The
threshold rule caps every discount at a common C, choosing C so that the
capped discounts add up to W* exactly. The result is budget balanced and still
individually rational (every capped discount is non-negative), but an agent
whose discount is capped can now gain by shading her report. That lost
incentive compatibility is the price of budget balance under this rule, and it
is measured in the experiment rather than assumed.

Posted price (PP)
-----------------
Fixed prices instead of an auction. A passenger is offered a flat fare for a
solo ride and a common shared fare; a driver is offered a fixed participation
payment plus his observable per-kilometre cost. A group is admissible only if
every member accepts the price on offer and the group's own margin (fares
minus driver payment) is non-negative. Among admissible groups the platform
serves as many passengers as possible, breaking ties by margin. Feasibility
still uses the reported pickup and detour bounds, exactly as in ORA.

PP is budget balanced group by group and individually rational by
construction. It gives up allocative efficiency, because reported values only
decide acceptance and do not rank allocations. It is also not exactly
strategy-proof in the mode dimension: a passenger who accepts both modes can
sometimes obtain a solo ride by declining the shared fare. That gain is
measured in the experiment as well.

Both functions return the ordinary Allocation type, so the metrics layer
measures them exactly as it measures every other arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from ora.payments import ClearingResult
from ora.types import Allocation, Driver, Group, Mode, Passenger


# --------------------------------------------------------------------------
# Threshold rule
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdInfo:
    """Diagnostics of one threshold-rule clearing."""

    vcg_budget: float          # B under Clarke payments
    threshold: float           # the cap C; +inf when no cap was needed
    sum_vickrey_discounts: float
    sum_capped_discounts: float
    n_capped_agents: int       # agents whose discount was actually reduced


def _passenger_value(passenger: Passenger, mode: Mode) -> float:
    return passenger.v_solo if mode is Mode.SOLO else passenger.v_share


def vickrey_discounts(
    result: ClearingResult,
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
) -> tuple[dict[int, float], dict[int, float]]:
    """Clarke utility of every allocated agent, using reported types.

    Returns two dicts: passenger id -> discount and driver id -> discount.
    Tiny negative values from floating point are clamped to zero; a genuinely
    negative discount would be an IR failure of the Clarke layer and is
    rejected loudly.
    """
    allocation = result.allocation
    pax: dict[int, float] = {}
    drv: dict[int, float] = {}

    for group in allocation.groups:
        for pid in group.passenger_ids:
            value = _passenger_value(passengers[pid], group.mode)
            pax[pid] = value - allocation.payments[pid]
        driver = drivers[group.driver_id]
        drv[group.driver_id] = (
            allocation.driver_earnings[group.driver_id]
            - group.route_cost
            - driver.reservation
        )

    for book in (pax, drv):
        for key, value in book.items():
            if value < -1e-6:
                raise ValueError(
                    f"negative Vickrey discount {value} for agent {key}; "
                    "the Clarke layer should never produce this"
                )
            book[key] = max(0.0, value)

    return pax, drv


def _threshold_for(discounts: list[float], target: float) -> float:
    """Smallest C with sum(min(d, C)) == target, assuming sum(d) > target."""
    values = sorted(discounts)
    n = len(values)
    below = 0.0
    for k, value in enumerate(values):
        remaining = n - k
        # If every remaining discount were capped at `value`, the total would
        # be below + remaining * value. Once that reaches the target, the cap
        # lies between the previous value and this one.
        if below + remaining * value >= target:
            return (target - below) / remaining
        below += value
    # Unreachable when sum(values) > target.
    return values[-1]


def threshold_rule(
    result: ClearingResult,
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
) -> tuple[Allocation, ThresholdInfo]:
    """Re-price ORA's allocation with the Parkes-Kalagnanam-Eso threshold rule.

    The allocation is left untouched. If the Clarke budget is already
    non-negative the Clarke prices are returned unchanged.
    """
    allocation = result.allocation
    pax, drv = vickrey_discounts(result, passengers, drivers)
    welfare = result.welfare

    total = sum(pax.values()) + sum(drv.values())
    vcg_budget = (
        sum(allocation.payments.values())
        - sum(allocation.driver_earnings.values())
    )

    if total <= welfare + 1e-9:
        info = ThresholdInfo(
            vcg_budget=vcg_budget,
            threshold=float("inf"),
            sum_vickrey_discounts=total,
            sum_capped_discounts=total,
            n_capped_agents=0,
        )
        return allocation, info

    # Welfare of a selected allocation is non-negative because the empty
    # allocation is always available, so a non-negative cap always exists.
    cap = _threshold_for(list(pax.values()) + list(drv.values()),
                         max(0.0, welfare))

    payments: dict[int, float] = {}
    earnings: dict[int, float] = {}
    n_capped = 0

    for group in allocation.groups:
        for pid in group.passenger_ids:
            discount = min(pax[pid], cap)
            n_capped += pax[pid] > cap + 1e-12
            payments[pid] = (
                _passenger_value(passengers[pid], group.mode) - discount
            )
        driver = drivers[group.driver_id]
        discount = min(drv[driver.id], cap)
        n_capped += drv[driver.id] > cap + 1e-12
        earnings[driver.id] = (
            group.route_cost + driver.reservation + discount
        )

    new_allocation = Allocation(
        groups=allocation.groups,
        payments=payments,
        driver_earnings=earnings,
    )
    info = ThresholdInfo(
        vcg_budget=vcg_budget,
        threshold=cap,
        sum_vickrey_discounts=total,
        sum_capped_discounts=sum(
            min(d, cap) for d in list(pax.values()) + list(drv.values())
        ),
        n_capped_agents=int(n_capped),
    )
    return new_allocation, info


# --------------------------------------------------------------------------
# Posted price
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PostedPrices:
    """One posted-price menu.

    fare_solo:      flat fare for a solo trip
    share_ratio:    shared fare as a fraction of the solo fare
    base_pay:       fixed participation payment to an assigned driver
    per_km_pay:     payment per route kilometre (set to the observable
                    operating cost, so the fixed part is what a driver's
                    reservation value is compared against)
    """

    fare_solo: float
    share_ratio: float
    base_pay: float
    per_km_pay: float

    def fare(self, mode: Mode) -> float:
        if mode is Mode.SOLO:
            return self.fare_solo
        return self.share_ratio * self.fare_solo

    def driver_pay(self, group: Group) -> float:
        return self.base_pay + self.per_km_pay * group.route_km


def _admissible(
    group: Group,
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
    prices: PostedPrices,
) -> bool:
    fare = prices.fare(group.mode)
    for pid in group.passenger_ids:
        if _passenger_value(passengers[pid], group.mode) < fare:
            return False
    driver = drivers[group.driver_id]
    pay = prices.driver_pay(group)
    if pay < group.route_cost + driver.reservation:
        return False
    margin = fare * len(group.passenger_ids) - pay
    return margin >= -1e-9


def posted_price_clear(
    groups: list[Group],
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
    prices: PostedPrices,
    msg: bool = False,
) -> Allocation:
    """Clear one window at fixed prices.

    `groups` must already respect the reported pickup and detour bounds (the
    output of grouping.enumerate_groups). Reported values are used only to
    decide whether each agent accepts the price on offer.
    """
    admissible = [
        group for group in groups
        if _admissible(group, passengers, drivers, prices)
    ]
    if not admissible:
        return Allocation(groups=(), payments={}, driver_earnings={})

    margins = [
        prices.fare(g.mode) * len(g.passenger_ids) - prices.driver_pay(g)
        for g in admissible
    ]
    # Lexicographic objective: passengers served first, margin second. The
    # margin weight is small enough that one extra passenger always wins.
    scale = 1.0 / (1.0 + sum(abs(m) for m in margins))

    problem = pulp.LpProblem("posted_price", pulp.LpMaximize)
    x = {k: pulp.LpVariable(f"x_{k}", cat="Binary")
         for k in range(len(admissible))}
    problem += pulp.lpSum(
        (len(g.passenger_ids) + scale * margins[k]) * x[k]
        for k, g in enumerate(admissible)
    )
    for pid in passengers:
        terms = [x[k] for k, g in enumerate(admissible)
                 if pid in g.passenger_ids]
        if len(terms) > 1:
            problem += pulp.lpSum(terms) <= 1
    for did in drivers:
        terms = [x[k] for k, g in enumerate(admissible)
                 if g.driver_id == did]
        if len(terms) > 1:
            problem += pulp.lpSum(terms) <= 1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=msg))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"posted-price IP not solved to optimality: "
            f"{pulp.LpStatus[status]}"
        )

    chosen = [g for k, g in enumerate(admissible)
              if (x[k].value() or 0.0) > 0.5]
    payments = {
        pid: prices.fare(g.mode)
        for g in chosen for pid in g.passenger_ids
    }
    earnings = {g.driver_id: prices.driver_pay(g) for g in chosen}
    return Allocation(
        groups=tuple(chosen),
        payments=payments,
        driver_earnings=earnings,
    )
