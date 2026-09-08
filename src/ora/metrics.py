"""Measurement layer: everything the report needs, computed from an Allocation.

Phase 3. This module is deliberately blind to which mechanism produced its
input. It takes an Allocation, the agents, and the network, and returns numbers.
ORA, the VCG ablation and the greedy baseline all return the same Allocation
type, so the same measurement code runs on every arm and a comparison cannot be
contaminated by measuring the arms differently. That is the whole reason
Allocation exists as a type rather than each mechanism returning its own bundle.

Distances become time here and nowhere else, through network.minutes and the
single declared avg_speed_kmh. Change the speed and the allocation does not
move, only the reported times do.

The one modelling decision in this module, stated loudly because it is the one
that can quietly destroy the fairness claim.

Fairness indices need a population. The obvious choice is served passengers,
because an unserved passenger has no waiting time to measure. But that choice
is a trap: a mechanism that serves five passengers out of forty and serves them
perfectly scores a Gini near zero and looks maximally fair, while a mechanism
that serves thirty-five with some spread looks worse. Measuring only the served
rewards refusing service, which is the opposite of what a fairness-aware thesis
should reward.

So both populations are computed, and every reported figure says which it uses:

  served-only : the distribution among people who actually got a ride. Answers
                "given that I was served, was I treated comparably to others?"
  all         : unserved passengers enter at a penalty wait fixed by the
                network rather than by the arm's own performance, so refusing
                service is penalised rather than rewarded. Answers "across
                everyone who asked, how unequal was the outcome?"

I report `all` as the headline and `served_only` alongside it, because the
match rate is itself one of my metrics and the two together are what make the
comparison honest. Reporting only the served-only figure would let any arm buy
fairness by matching less.

The limitation that survives both populations, and that I want stated in the
report rather than discovered by a reader. No penalty scheme makes a Gini
non-gameable, because Gini measures dispersion and a population that is
uniformly badly off has very little dispersion. An arm that serves one
passenger and abandons the rest produces a nearly flat distribution and scores
as almost perfectly equal. I found this by writing a test asserting the
opposite and watching it fail.

The consequence is a reporting rule, not a formula: a fairness index is never
quoted on its own. Every Gini and Jain figure in the report appears next to the
match rate and next to mean_wait_all_min, which is the mean over everyone who
asked rather than everyone who was served. That mean does move the right way
when an arm abandons passengers, because it measures the level rather than the
spread. Inequality and adequacy are different questions and need different
numbers; conflating them is the mistake this module is arranged to prevent.
"""
from dataclasses import dataclass, asdict

import numpy as np

from ora.types import Allocation, Passenger, Driver, Mode
from ora.network import ZoneNetwork
from ora.grouping import group_welfare
from ora.network import ZoneNetwork
from ora.grouping import group_welfare
from ora.payments import driver_utility
from ora.acceptability import excess_penalty, nominal_value

# --------------------------------------------------------------------------
# Inequality indices. Both take a plain sequence and return a scalar.
# --------------------------------------------------------------------------

def gini(values) -> float:
    """Gini coefficient. 0 is perfect equality, 1 is maximal inequality.

    Computed from the sorted-values formula rather than the double sum over all
    pairs, which is O(n^2) and pointless here. Both give the same number.

    Defined as 0 for an empty population and for an all-zero population: with
    nothing to distribute there is no inequality, and returning NaN would
    propagate into every aggregate downstream. Note this makes Gini blind to
    the difference between "everyone waited zero" and "nobody was served",
    which is exactly why match rate is reported alongside it and never on its
    own.

    Assumes non-negative values. Waiting times and earnings are non-negative by
    construction; passenger utilities are not, so do not pass those here
    without thinking about what a negative share would mean.
    """
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        raise ValueError("Gini is not defined for negative values here; "
                         "check what population was passed in")
    total = x.sum()
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) - (n + 1) * total) / (n * total))


def jain(values) -> float:
    """Jain's fairness index. 1 is perfect equality, 1/n is maximal inequality.

    (sum x)^2 / (n * sum x^2). Reported alongside Gini rather than instead of
    it because the two disagree in useful ways: Jain is bounded below by 1/n
    and so is sensitive to population size, while Gini is not. Where they
    disagree the disagreement is informative, and the problem statement asks
    for Jain specifically.

    Returns 1.0 for an empty or all-zero population, matching the Gini
    convention: no distribution means no unfairness detected.
    """
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return 1.0
    denom = x.size * float(np.sum(x ** 2))
    if denom == 0:
        return 1.0
    return float(float(np.sum(x)) ** 2 / denom)


def _p95(values) -> float:
    """95th percentile, or 0.0 for an empty population.

    The problem statement asks for 95th-percentile waiting time specifically,
    which is the number that catches the tail the mean hides: an arm can have a
    good mean wait and still leave a handful of passengers waiting far too long.
    """
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 95))


# --------------------------------------------------------------------------
# The result bundle.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Metrics:
    """Every number one cleared window produced.

    Flat and frozen so that a list of these goes straight into a pandas
    DataFrame with one row per window, which is what the sweep consumes.
    """
    # Population
    n_passengers: int
    n_drivers: int

    # Efficiency
    realised_welfare: float
    matched_passengers: int
    match_rate: float
    pooled_passengers: int
    pooling_rate: float

    # Service quality, minutes and km
    mean_wait_min: float          # served passengers only
    mean_wait_all_min: float      # unserved charged the penalty wait
    p95_wait_min: float
    mean_detour_km: float
    idle_drivers: int
    idle_ratio: float

    # Fairness over waiting time, both populations
    gini_wait_all: float
    jain_wait_all: float
    gini_wait_served: float
    jain_wait_served: float

    # Fairness over driver earnings. Idle drivers earn zero and are included:
    # a driver who got no work is exactly the inequality being measured, so
    # excluding him would hide the thing the index is for.
    gini_earnings: float
    jain_earnings: float

    # Money
    total_payments: float
    total_driver_earnings: float
    budget_balance: float
    deficit_pct_of_welfare: float
    
    # Driver utility and individual rationality.
    # These metrics use allocated drivers only, since idle drivers receive their
    # outside option of zero and do not participate in the cleared allocation.
    driver_ir_violation_rate: float
    mean_allocated_driver_utility: float
    min_allocated_driver_utility: float

    # Passenger utility, under two accounting conventions. The nominal figures
    # book a served passenger at the flat valuation every mechanism here
    # already assumes for itself. The adjusted figure additionally charges for
    # the part of the route that exceeded the bounds she reported, at the rate
    # derived from those same bounds. The gap between them, reported as
    # total_acceptability_cost, is the welfare a mechanism books but a
    # passenger would not have accepted. Served passengers only: an unserved
    # passenger takes her outside option of zero and cannot violate IR.
    passenger_ir_violation_rate: float
    mean_allocated_passenger_utility: float
    min_allocated_passenger_utility: float
    mean_adjusted_passenger_utility: float
    total_acceptability_cost: float

    def as_row(self) -> dict:
        """One flat dict, for building a DataFrame across windows."""
        return asdict(self)


# --------------------------------------------------------------------------
# The computation.
# --------------------------------------------------------------------------

def _wait_minutes_by_passenger(allocation: Allocation,
                               network: ZoneNetwork) -> dict[int, float]:
    """Waiting time per served passenger, in minutes.

    Waiting is derived from pickup_km, which the router stored as the
    cumulative distance travelled before that passenger boards. That is the
    r_max quantity, so the waiting time reported here is measured against
    exactly the bound the passenger reported, not against some other notion of
    wait. Conversion to minutes happens through the network's single declared
    average speed.
    """
    out: dict[int, float] = {}
    for g in allocation.groups:
        for pid, km in zip(g.passenger_ids, g.pickup_km):
            out[pid] = network.minutes(km)
    return out


def _passenger_utilities(allocation: Allocation,
                         by_id: dict[int, Passenger],
                         waits: dict[int, float],
                         network: ZoneNetwork
                         ) -> tuple[list[float], list[float], float]:
    """Nominal utility, adjusted utility and total acceptability cost.

    Nominal utility is the passenger's flat valuation for the mode she got,
    minus what she paid. Adjusted utility subtracts, in addition, the cost of
    whatever part of her route exceeded the bounds she reported. The third
    return value is the sum of those excess costs across served passengers.

    The same computation is applied to every arm. A mechanism whose feasibility
    respects the reported bounds pays no acceptability cost, which is a measured
    outcome rather than an exemption granted to it.
    """
    nominal: list[float] = []
    adjusted: list[float] = []
    total_cost = 0.0

    for group in allocation.groups:
        for pid, detour_km in zip(group.passenger_ids, group.detour_km):
            passenger = by_id[pid]
            payment = allocation.payments.get(pid, 0.0)

            value = nominal_value(passenger, group.mode)
            nominal_utility = value - payment

            penalty = excess_penalty(
                passenger,
                detour_km,
                waits.get(pid, 0.0),
                network,
            )

            nominal.append(nominal_utility)
            adjusted.append(nominal_utility - penalty)
            total_cost += penalty

    return nominal, adjusted, total_cost


def compute_metrics(allocation: Allocation,
                    passengers: list[Passenger],
                    drivers: list[Driver],
                    network: ZoneNetwork,
                    unserved_wait_min: float | None = None) -> Metrics:
    """Measure one cleared window."""
    n_pax = len(passengers)
    n_drv = len(drivers)

    by_id = {p.id: p for p in passengers}
    drivers_by_id = {d.id: d for d in drivers}

    # ---- Efficiency ----------------------------------------------------
    realised_welfare = sum(
        group_welfare(g, by_id, drivers_by_id)
        for g in allocation.groups
    )

    served_ids = [
        pid
        for g in allocation.groups
        for pid in g.passenger_ids
    ]
    matched = len(served_ids)
    pooled = sum(len(g.passenger_ids) for g in allocation.groups
                 if g.mode is Mode.SHARED)

    match_rate = matched / n_pax if n_pax else 0.0
    # Pooling rate is over SERVED passengers, not over all of them. Pooling is
    # a property of how someone was served, so an unserved passenger is not a
    # failure to pool, she is a failure to match, and that is a different
    # number reported separately.
    pooling_rate = pooled / matched if matched else 0.0

    # ---- Service quality ------------------------------------------------
    waits = _wait_minutes_by_passenger(allocation, network)
    served_waits = list(waits.values())

    detours = [d for g in allocation.groups for d in g.detour_km]
    mean_detour = float(np.mean(detours)) if detours else 0.0

    busy_drivers = {g.driver_id for g in allocation.groups}
    idle = n_drv - len(busy_drivers)
    idle_ratio = idle / n_drv if n_drv else 0.0

    # ---- Fairness over waiting -----------------------------------------
    # The `all` population charges every unserved passenger a penalty wait.
    # Choosing that penalty is the subtle part, and my first attempt was wrong
    # in a way the tests caught, so the reasoning is written down here.
    #
    # I first used the worst wait observed in this window. That is gameable:
    # an arm that serves one passenger has one observed wait, so charging the
    # unserved that same value makes the whole population identical and the
    # Gini comes out at zero. The arm that served almost nobody scored as
    # perfectly fair, which is precisely the trap this design exists to stop.
    # The penalty scaled with how badly the arm performed.
    #
    # So the penalty has to be external to the arm. It is the network
    # diameter converted to minutes: the longest wait this geography can
    # produce, identical for every arm measured on the same network. An
    # unserved passenger is charged as though she waited the worst the city
    # allows, which is a conservative stand-in for never being collected.
    #
    # The max() guard keeps the served-only figure a lower bound on the `all`
    # figure even in the pathological case where a cumulative pickup distance
    # along a multi-stop route exceeds the straight-line diameter.
    penalty_wait = unserved_wait_min
    if penalty_wait is None:
        penalty_wait = network.minutes(float(np.max(network.dist_km)))
    if served_waits:
        penalty_wait = max(penalty_wait, max(served_waits))
    all_waits = served_waits + [penalty_wait] * (n_pax - matched)

    # ---- Fairness over driver earnings ----------------------------------
    # Every driver appears, idle ones at zero. A mechanism that concentrates
    # all work on two drivers and leaves eight idle is unequal, and that is
    # precisely what Miao et al. found surge pricing doing to real drivers.
    earnings = [allocation.driver_earnings.get(d.id, 0.0) for d in drivers]
    # Clamp tiny negatives from floating point so gini does not raise. A
    # genuinely negative earning would be a bug in the payment rule, not a
    # rounding artefact, so the tolerance is deliberately tight.
    earnings = [0.0 if -1e-9 < e < 0 else e for e in earnings]
    
        # ---- Driver utility and individual rationality -------------------------
    # IR is evaluated over allocated drivers. Idle drivers receive their outside
    # option of zero and therefore cannot constitute an IR violation.
    allocated_driver_ids = {g.driver_id for g in allocation.groups}

    allocated_driver_utilities = [
        driver_utility(driver, allocation)
        for driver in drivers
        if driver.id in allocated_driver_ids
    ]

    if allocated_driver_utilities:
        driver_ir_violation_rate = (
            sum(utility < -1e-9 for utility in allocated_driver_utilities)
            / len(allocated_driver_utilities)
        )
        mean_allocated_driver_utility = float(
            np.mean(allocated_driver_utilities)
        )
        min_allocated_driver_utility = float(
            np.min(allocated_driver_utilities)
        )
    else:
        driver_ir_violation_rate = 0.0
        mean_allocated_driver_utility = 0.0
        min_allocated_driver_utility = 0.0

    # ---- Passenger utility and acceptability -----------------------------
    # Mirrors the driver-side block above: evaluated over served passengers,
    # since an unserved passenger takes her outside option of zero.
    nominal_utils, adjusted_utils, acceptability_cost = _passenger_utilities(
        allocation,
        by_id,
        waits,
        network,
    )

    if nominal_utils:
        # IR is judged on the adjusted figure: a passenger whose ride overshot
        # the bounds she reported by more than the ride was worth to her has
        # been made worse off by being served, however the mechanism books it.
        passenger_ir_violation_rate = (
            sum(utility < -1e-9 for utility in adjusted_utils)
            / len(adjusted_utils)
        )
        mean_allocated_passenger_utility = float(np.mean(nominal_utils))
        min_allocated_passenger_utility = float(np.min(nominal_utils))
        mean_adjusted_passenger_utility = float(np.mean(adjusted_utils))
    else:
        passenger_ir_violation_rate = 0.0
        mean_allocated_passenger_utility = 0.0
        min_allocated_passenger_utility = 0.0
        mean_adjusted_passenger_utility = 0.0

    # ---- Money -----------------------------------------------------------
    total_payments = float(sum(allocation.payments.values()))
    total_earnings = float(sum(allocation.driver_earnings.values()))
    bb = total_payments - total_earnings

    # Reported as a percentage of welfare created, never as an absolute. The
    # absolute figure is meaningless across arms with different welfare, and
    # the percentage is the form directly comparable to the 55-79% deficit
    # Kamar & Horvitz measured in 2009. Negative means deficit, matching the
    # sign convention in payments.py.
    deficit_pct = (100.0 * bb / realised_welfare) if realised_welfare else 0.0

    return Metrics(
        n_passengers=n_pax,
        n_drivers=n_drv,
        realised_welfare=realised_welfare,
        matched_passengers=matched,
        match_rate=match_rate,
        pooled_passengers=pooled,
        pooling_rate=pooling_rate,
        mean_wait_min=float(np.mean(served_waits)) if served_waits else 0.0,
        mean_wait_all_min=float(np.mean(all_waits)) if all_waits else 0.0,
        p95_wait_min=_p95(served_waits),
        mean_detour_km=mean_detour,
        idle_drivers=idle,
        idle_ratio=idle_ratio,
        gini_wait_all=gini(all_waits),
        jain_wait_all=jain(all_waits),
        gini_wait_served=gini(served_waits),
        jain_wait_served=jain(served_waits),
        gini_earnings=gini(earnings),
        jain_earnings=jain(earnings),
        driver_ir_violation_rate=driver_ir_violation_rate,
        mean_allocated_driver_utility=mean_allocated_driver_utility,
        min_allocated_driver_utility=min_allocated_driver_utility,
        passenger_ir_violation_rate=passenger_ir_violation_rate,
        mean_allocated_passenger_utility=mean_allocated_passenger_utility,
        min_allocated_passenger_utility=min_allocated_passenger_utility,
        mean_adjusted_passenger_utility=mean_adjusted_passenger_utility,
        total_acceptability_cost=acceptability_cost,
        total_payments=total_payments,
        total_driver_earnings=total_earnings,
        budget_balance=bb,
        deficit_pct_of_welfare=deficit_pct,
    )