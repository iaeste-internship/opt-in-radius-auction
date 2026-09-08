"""Enumeration of feasible hyperedges (driver, passenger subset, mode).

This is the module that builds the allocation space. Everything downstream, the
winner determination problem and the Clarke payments, operates over the list
this produces, so if a group is missing here it simply cannot be allocated and
cannot be priced. That makes the enumeration part of the mechanism definition
rather than an implementation detail: the space I enumerate over IS the space
the VCG argument quantifies over.

Two things follow from that.

First, the enumeration has to be exhaustive over what is feasible. Dropping
groups heuristically to save time would mean the optimiser no longer maximises
welfare over the true allocation space, and strategy-proofness is a statement
about that maximisation. Every subset that survives the reported constraints
gets generated, however unpromising it looks.

Second, the mode enters here. A group is a (driver, subset, mode) triple, not
just (driver, subset). Right now the mode is determined by size, since a single
passenger is always SOLO and a multi-passenger group is always SHARED, so the
loop over modes has one element every time. I have kept the loop structure
anyway because the whole point of ORA is that the mode lives in the allocation
space, and if I later allow a driver to carry two passengers who each ride
alone in sequence, the change is a longer tuple here and nothing else. Writing
it as a loop now costs nothing and marks where that extension goes.

The cost is combinatorial and I have measured it, after the radius fix that
moved the boarding-distance check inside the router. At capacity 3 the
enumeration produces 87 groups at n=40, 249 at n=60 and 543 at n=80, which is
growth of roughly n^2.6 rather than the n^4 I estimated before the fix. The
solve itself scales much better than the enumeration does, so this module, not
the IP, is what will bound my experiment sizes. Someya & Namerikawa hit the
same wall at 120 requests and suggested spatio-temporal pruning without
implementing it; that is the optimisation I would reach for first if the radius
sweep turns out not to fit in the calendar.
"""
from dataclasses import dataclass
from itertools import combinations

from ora.types import Passenger, Driver, Group, Mode
from ora.network import ZoneNetwork
from ora.feasibility import compatible_passengers
from ora.routing import route


@dataclass
class GroupingStats:
    """Instrumentation: how big did the search actually get?
    Not decoration. The gap between candidates_considered and groups_generated
    is the share of subsets killed at the routing stage. In my runs that share
    is 79% at n=40, 86% at n=60 and 91% at n=80: it grows with market size,
    because larger markets admit more subsets that pass the cheap prior filters
    and only fail once a route is attempted. That trend is the argument for a
    cheap detour bound applied before routing, and it says the case for one
    gets stronger, not weaker, as I scale up.

    Mutable, unlike everything else in this project, because it is a counter
    updated in place through the loop. It carries no modelling content, so
    nothing about the mechanism depends on it being frozen.
    """
    candidates_considered: int = 0
    groups_generated: int = 0 
    rejected_by_routing: int = 0

    def __str__(self) -> str:
        return (f"considered={self.candidates_considered} "
                f"generated={self.groups_generated} "
                f"rejected_routing={self.rejected_by_routing}")


def enumerate_groups(drivers: list[Driver], passengers: list[Passenger],
                     network: ZoneNetwork,
                     max_group_size: int | None = None
                     ) -> tuple[list[Group], GroupingStats]:
    """All feasible (driver, subset, mode) triples.

    Pruning order is the whole performance story, and it runs cheapest first.
    The radius filter is a single distance lookup per passenger and it shrinks
    the ground set, which matters more than anything else because the number of
    subsets is exponential in the size of that set. Capacity then bounds the
    subset sizes. Routing, which is factorial in group size, runs last and only
    on subsets that survived both.

    max_group_size is a testing and profiling knob, not a modelling parameter.
    Capping it below driver capacity is what lets me hold the enumeration cost
    fixed while varying n, and it is how I produced the scaling measurements.
    It must be left at None for any run whose numbers go in the report:
    capping it silently removes feasible allocations, so the mechanism would no
    longer be maximising over the true space and the efficiency figures would
    be against a restricted benchmark.
    """
    stats = GroupingStats()
    groups: list[Group] = []

    for driver in drivers:
        # Cheap prior filter, once per driver. Everything outside `reachable`
        # is unreachable by this driver by construction, so it never enters the
        # combinatorial part at all.
        reachable = compatible_passengers(driver, passengers, network)
        cap = driver.capacity
        if max_group_size is not None:
            cap = min(cap, max_group_size)

        for size in range(1, cap + 1):
            for subset in combinations(reachable, size):
                stats.candidates_considered += 1

                # Mode is determined by size for now: one passenger is SOLO, 
                # more than one is SHARED. Kept as a tuple and a loop because
                # the mode belongs to the allocation space, and this is the
                # single line that changes if I ever allow a richer set.
                modes = (Mode.SOLO,) if size == 1 else (Mode.SHARED,)
                for mode in modes:
                    result = route(driver, subset, mode, network)

                    # None means no ordering of pickups and dropoffs satisfied
                    # everyone's detour tolerance. Not an error: this is the
                    # posterior filter doing its job, and it is where most
                    # candidates die.
                    if result is None:
                        stats.rejected_by_routing += 1
                        continue

                    total_km, pickups, detours = result
                    groups.append(Group(
                        driver_id=driver.id,
                        passenger_ids=tuple(p.id for p in subset),
                        mode=mode,
                        route_km=total_km,
                        # Cost is stored on the hyperedge rather than computed
                        # later, which is exactly the hypergraph move: once the
                        # weight is fixed inside the edge, matching becomes a
                        # max-weight set packing problem and Clarke payments
                        # apply to it directly.
                        route_cost=total_km * driver.cost_per_km,
                        pickup_km=pickups,
                        detour_km=detours,
                    ))
                    stats.groups_generated += 1

    return groups, stats


def group_welfare(
    group: Group,
    passengers: dict[int, Passenger],
    drivers: dict[int, Driver],
) -> float:
    """Reported social welfare of one feasible hyperedge.

    Passenger valuations contribute positively. The selected driver's route
    cost and reported reservation value contribute negatively.

    Although a driver may appear in multiple candidate hyperedges, the WDP
    allows at most one winning hyperedge per driver. Therefore, the reservation
    value is subtracted exactly once whenever the driver participates.
    """
    value = 0.0

    for pid in group.passenger_ids:
        passenger = passengers[pid]

        if group.mode is Mode.SOLO:
            value += passenger.v_solo
        else:
            value += passenger.v_share

    driver = drivers[group.driver_id]

    return value - group.route_cost - driver.reservation