"""The Someya & Namerikawa (2026) arm: their mechanism on our instance.

This module reimplements the mechanism of Someya & Namerikawa, "Mechanism
design-based matching and pricing for multi-passenger ride-sharing" (SICE JCMSI
2026), and runs it on the exact same demand, network and passenger population
ORA is measured on. It is a third comparison arm alongside ORA and the VCG
ablation, and it exists to answer a different question from the ablation.

The ablation asks what ORA's own elicitation of pickup radius and pooling buys
inside ORA's architecture. This arm asks how the closest published state of the
art performs on our instance when measured against the passenger constraints it
structurally cannot represent. Someya's feasibility is seat capacity alone: it
has no per-passenger pickup radius (r_max) and no per-passenger detour tolerance
(delta_max). It therefore cannot avoid collecting a passenger beyond the radius
she would have accepted, and cannot avoid a detour past what she would have
tolerated. Those events are not defects in the reimplementation; they are the
consequence of the model, and counting them is the point of the arm.

Isolation contract.

Nothing in the ORA path is imported in a way that could change its output, and
no ORA module is modified. In particular this arm does NOT call ora.routing.route
or ora.grouping.enumerate_groups, because both enforce r_max and delta_max as
hard cuts, which Someya's model does not have. The pickup-then-dropoff route
search is duplicated here without those two cuts. The duplication is deliberate:
a shared router with a "skip the cuts" flag would touch the ORA path, and the
requirement is that ORA and the ablation stay bit-identical.

What is faithful to the paper, and what is a declared evaluation choice.

Faithful: seat-capacity-only feasibility (their Eq. 1); the soft valuation
V_i = C_i^self - b_i with b_i = omega_det * detour + omega_wait * wait
(their Eqs. 8-9); exact max-weight set packing (their Eq. 18); passenger-side
Clarke payments by exclusion (their Eq. 24); drivers as fixed-rate executors
with no private reservation and no payment (their Section 2.1).

Declared evaluation choice: Someya's passengers report omega_det and omega_wait,
which do not exist in our Passenger type. Rather than invent them, we DERIVE them
per passenger from the flexibility bounds this population already carries, so the
two arms face the same population. The derivation is stated in `someya_valuation`
and is the one parameter a reviewer will scrutinise, so it is kept in one place
and justified there.

Measurement.

The winners this arm produces are scored by the SAME compute_metrics and the
SAME group_welfare as ORA and the ablation. The arm changes only how the
allocation is decided, never how it is scored. Someya's soft valuation is used
INSIDE this module to choose the allocation; the realised welfare reported for
the arm is ORA's true economic welfare of that allocation, exactly as for the
other two arms. That is what keeps the three-way comparison honest.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations, permutations

import pulp

from ora.types import Passenger, Driver, Group, Mode, Allocation
from ora.network import ZoneNetwork
from ora.feasibility import solo_km, capacity_ok
from ora.grouping import group_welfare
from ora.payments import ClearingResult
from ora.acceptability import omega_for


# --------------------------------------------------------------------------
# Valuation: Someya's soft disutility, with omega derived from our bounds.
# --------------------------------------------------------------------------

# The omega derivation used to live here as a private helper. It now lives in
# ora.acceptability, because the passenger-utility metrics need the same rates
# and a general metrics layer should not import them from one comparison arm.
# The rule itself is unchanged; see acceptability.omega_for for the reasoning.
_omega = omega_for


def someya_passenger_value(group: Group,
                           pid: int,
                           passengers: dict[int, Passenger],
                           network: ZoneNetwork) -> float:
    """Passenger i's Someya valuation V_i(S_k) for one candidate group.

    The source mechanism defines V_i(S_k) = C_i^self - b_i, where b_i is the
    passenger-specific detour/wait disadvantage.  Keeping this term in a helper
    is important because the Clarke payment must remove the *full* V_i(S_k)
    from social welfare when computing the welfare of the other participants.
    """
    passenger = passengers[pid]
    idx = group.passenger_ids.index(pid)
    detour_km = group.detour_km[idx]
    omega_det, omega_wait = _omega(passenger, network)

    if group.mode is Mode.SOLO:
        # In the relative-utility formulation used by the reimplementation, a
        # singleton is the solo outside-service reference and carries no
        # sharing disadvantage.
        b_i = 0.0
    else:
        wait_min = _wait_min_for(group, pid, network)
        det_term = 0.0 if detour_km == 0.0 else omega_det * detour_km
        wait_term = 0.0 if wait_min == 0.0 else omega_wait * wait_min
        b_i = det_term + wait_term

    if b_i == float("inf"):
        return float("-inf")

    return passenger.v_solo - b_i


def someya_valuation(group: Group,
                     passengers: dict[int, Passenger],
                     drivers: dict[int, Driver],
                     network: ZoneNetwork) -> float:
    """Someya's group weight W = sum_i V_i(S_k) - C_j(S_k)."""
    total = 0.0
    for pid in group.passenger_ids:
        value = someya_passenger_value(group, pid, passengers, network)
        if value == float("-inf"):
            return float("-inf")
        total += value
    return total - group.route_cost


def _wait_min_for(group: Group, pid: int, network: ZoneNetwork) -> float:
    """Waiting time in minutes for one passenger in a group.

    The pickup distance stored on the group is the cumulative driver distance at
    the moment this passenger boards, which is exactly what Someya's waiting
    disutility is measured on. Converted to minutes through the network speed.
    """
    idx = group.passenger_ids.index(pid)
    return network.minutes(group.pickup_km[idx])


# --------------------------------------------------------------------------
# Routing without r_max / delta_max. Duplicated from ora.routing on purpose:
# Someya's model has neither cut, and importing route() would apply both.
# --------------------------------------------------------------------------

def _someya_route(driver: Driver, passengers: tuple[Passenger, ...],
                  mode: Mode, network: ZoneNetwork):
    """Cheapest pickup-then-dropoff route, with NO radius or detour cut.

    Returns (total_km, pickup_km tuple, detour_km tuple). Never None for a
    non-empty group, because Someya's feasibility is capacity only: any subset
    that fits the vehicle is routable. This is the single behavioural difference
    from ora.routing.shared_route, and it is the whole point of the arm.
    """
    if mode is Mode.SOLO:
        p = passengers[0]
        pickup = network.km(driver.location, p.origin)
        ride = network.km(p.origin, p.destination)
        return pickup + ride, (pickup,), (0.0,)

    best = None
    for pick_order in permutations(passengers):
        for drop_order in permutations(passengers):
            total = 0.0
            pos = driver.location
            pickup_at = {}
            for p in pick_order:
                total += network.km(pos, p.origin)
                pickup_at[p.id] = total
                pos = p.origin
            dropoff_at = {}
            for p in drop_order:
                total += network.km(pos, p.destination)
                dropoff_at[p.id] = total
                pos = p.destination
            in_vehicle = {p.id: dropoff_at[p.id] - pickup_at[p.id]
                          for p in passengers}
            if any(in_vehicle[p.id] < 0 for p in passengers):
                continue
            detours = tuple(in_vehicle[p.id] - solo_km(p, network)
                            for p in passengers)
            pickups = tuple(pickup_at[p.id] for p in passengers)
            if best is None or total < best[0]:
                best = (total, pickups, detours)
    return best


def enumerate_someya_groups(drivers: list[Driver],
                            passengers: list[Passenger],
                            network: ZoneNetwork) -> list[Group]:
    """All capacity-feasible groups, ignoring r_max and delta_max.

    Mirrors ora.grouping.enumerate_groups in structure but uses the unrestricted
    router and does not pre-filter passengers by radius. Every subset up to
    vehicle capacity becomes a group, which is Someya's Eq. (1).
    """
    groups: list[Group] = []
    for driver in drivers:
        for size in range(1, driver.capacity + 1):
            for subset in combinations(passengers, size):
                mode = Mode.SOLO if size == 1 else Mode.SHARED
                result = _someya_route(driver, subset, mode, network)
                if result is None:
                    continue
                total_km, pickups, detours = result
                groups.append(Group(
                    driver_id=driver.id,
                    passenger_ids=tuple(p.id for p in subset),
                    mode=mode,
                    route_km=total_km,
                    route_cost=total_km * driver.cost_per_km,
                    pickup_km=pickups,
                    detour_km=detours,
                ))
    return groups


# --------------------------------------------------------------------------
# Winner determination on Someya's weight. Same set packing as ORA's WDP,
# duplicated so ORA's solve_wdp is untouched.
# --------------------------------------------------------------------------

def _solve_someya_wdp(groups: list[Group],
                      passengers: dict[int, Passenger],
                      drivers: dict[int, Driver],
                      network: ZoneNetwork,
                      exclude_passenger: int | None = None,
                      msg: bool = False) -> tuple[list[Group], float]:
    """Max-weight set packing under Someya's valuation.

    Identical constraints to ora.wdp.solve_wdp (each passenger in <=1 group,
    each driver in <=1 group, empty allocation feasible with objective 0). Only
    the weight differs, and there is no driver-exclusion path because Someya
    prices passengers only.
    """
    active = [g for g in groups
              if exclude_passenger not in g.passenger_ids
              and someya_valuation(g, passengers, drivers, network) > float("-inf")]
    if not active:
        return [], 0.0

    problem = pulp.LpProblem("someya_wdp", pulp.LpMaximize)
    selected = {i: pulp.LpVariable(f"x_{i}", cat="Binary")
                for i in range(len(active))}

    problem += pulp.lpSum(
        someya_valuation(g, passengers, drivers, network) * selected[i]
        for i, g in enumerate(active))

    for pid in passengers:
        if pid == exclude_passenger:
            continue
        members = [i for i, g in enumerate(active) if pid in g.passenger_ids]
        if members:
            problem += pulp.lpSum(selected[i] for i in members) <= 1

    for did in drivers:
        members = [i for i, g in enumerate(active) if g.driver_id == did]
        if members:
            problem += pulp.lpSum(selected[i] for i in members) <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=msg))
    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(f"Someya WDP not optimal: {pulp.LpStatus[problem.status]}")

    winners = [g for i, g in enumerate(active) if selected[i].value() > 0.5]
    obj = pulp.value(problem.objective)
    return winners, float(obj if obj is not None else 0.0)


# --------------------------------------------------------------------------
# Clearing: allocate on Someya's weight, price passengers by Clarke exclusion.
# --------------------------------------------------------------------------

def run_someya(passengers: list[Passenger],
               drivers: list[Driver],
               network: ZoneNetwork,
               msg: bool = False) -> ClearingResult:
    """Clear one window under Someya & Namerikawa (2026).

    Returns the same ClearingResult type as ORA so the metrics layer cannot tell
    the arms apart. Driver earnings are empty: Someya does not pay drivers.
    `welfare` is ORA's true economic welfare of the selected allocation, so it is
    directly comparable to the other arms; `augmented_objective` carries Someya's
    own optimised weight for reference.
    """
    pax = {p.id: p for p in passengers}
    drv = {d.id: d for d in drivers}

    groups = enumerate_someya_groups(drivers, passengers, network)
    winners, someya_objective = _solve_someya_wdp(groups, pax, drv, network, msg=msg)
    solves = 1

    served = {pid for g in winners for pid in g.passenger_ids}
    payments: dict[int, float] = {}

    for pid in served:
        _, without = _solve_someya_wdp(groups, pax, drv, network,
                                       exclude_passenger=pid, msg=msg)
        solves += 1
        # Welfare of all *other* participants in the chosen allocation.
        # Eq. (24)'s Clarke externality removes passenger i's complete
        # valuation V_i(S_k) = C_i^self - b_i, not only C_i^self.  Leaving the
        # -b_i term in others' welfare would overcharge passengers who incur
        # detour/wait disadvantage and can spuriously violate IR.
        others_with = 0.0
        for g in winners:
            v = someya_valuation(g, pax, drv, network)
            if pid in g.passenger_ids:
                own_value = someya_passenger_value(
                    g, pid, pax, network
                )
                v -= own_value
            others_with += v
        payments[pid] = without - others_with

    # True economic welfare of the chosen allocation, scored with ORA's rule so
    # the number is comparable across arms.
    economic_welfare = float(sum(group_welfare(g, pax, drv) for g in winners))

    allocation = Allocation(
        groups=tuple(winners),
        payments=payments,
        driver_earnings={},
    )
    return ClearingResult(
        allocation=allocation,
        welfare=economic_welfare,
        n_exclusion_solves=solves,
        fairness_bonus=0.0,
        augmented_objective=someya_objective,
    )


def detour_violations(allocation: Allocation,
                      passengers: list[Passenger]) -> tuple[int, float]:
    """How many served passengers were routed past their true delta_max.

    The detour-side twin of baselines.radius_violations. Zero for ORA by
    construction; non-zero for this arm, because Someya's feasibility never
    checked delta_max. Together the two counts are the acceptability cost the
    state of the art cannot see.
    """
    by_id = {p.id: p for p in passengers}
    served = 0
    violated = 0
    for g in allocation.groups:
        for pid, det in zip(g.passenger_ids, g.detour_km):
            served += 1
            if det > by_id[pid].delta_max + 1e-9:
                violated += 1
    return violated, (violated / served if served else 0.0)
