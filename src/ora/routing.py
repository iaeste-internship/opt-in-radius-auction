"""Route cost for a driver serving a set of passengers.

This is the expensive half of feasibility. Given a driver and a candidate
subset, it answers two questions at once: is there any ordering of pickups and
dropoffs that respects everyone's detour tolerance, and if so what is the
cheapest one. The answer becomes the weight stored inside the hyperedge.

Precomputing the route here is what makes the whole VCG argument work. Once
each feasible group carries its own optimal routing cost, the winner
determination problem is a clean max-weight set packing problem over
independent hyperedges, and Clarke payments apply to it exactly. If the routing
were solved jointly with the matching, the allocation space would no longer be
finite and enumerable and the standard VCG argument would not transfer.

The search is exhaustive over orderings, which means the cost is factorial in
group size: 2!x2! = 4 orderings for a pair, 4!x4! = 576 for a group of four.
That is fine for the capacity-3 experiments I have measured and it is exact,
which matters more than speed here. An approximate router would make the stored
weights approximate, and approximate weights break strategy-proofness, which is
my main theorem. Kamar & Horvitz (2009) hit exactly this: their group
assignment was solved greedily and they concede that exact truthfulness is
already lost as a result.

If capacity 4 turns out to be necessary and too slow, the standard fix is
incremental insertion with pickup-before-dropoff enforced by construction,
which cuts the search space by a large constant without giving up exactness.
Not doing it until profiling says I have to.
"""
from itertools import permutations


from ora.feasibility import solo_km, radius_ok
from ora.types import Passenger, Driver, Mode
from ora.network import ZoneNetwork

def solo_route(driver: Driver, passenger: Passenger,
               network: ZoneNetwork) -> tuple[float, float, float]:
    """Driver serves one passenger directly, no sharing.

    Returns (total_km, pickup_km, detour_km). The detour is zero by
    definition: this is the trip the passenger would take alone, which is the
    baseline every shared detour is measured against.
    """
    pickup = network.km(driver.location, passenger.origin)
    ride = network.km(passenger.origin, passenger.destination)
    return pickup + ride, pickup, 0.0


def shared_route(driver: Driver, passengers: tuple[Passenger, ...],
                 network: ZoneNetwork) -> tuple[float, tuple[float, ...],
                                                tuple[float, ...]] | None:
    """Cheapest feasible pickup-then-dropoff sequence for a shared group.

    Returns (total_km, pickup_km per passenger, detour_km per passenger), or
    None if no ordering satisfies everyone's detour tolerance. Returning None
    is not an error: it is how a subset that passed the radius and capacity
    filters gets rejected, and it accounts for most of the pruning in practice.

    The structure is all pickups first, then all dropoffs. That is a modelling
    restriction, not the fully general vehicle routing problem, where pickups
    and dropoffs can interleave. Interleaving would allow cheaper routes in
    some cases but multiplies the search space further, and the all-pickups-
    first structure matches the event scenario I am designing for, where
    passengers are concentrated at an origin area and dispersing.
    """
    best = None

    for pick_order in permutations(passengers):
        for drop_order in permutations(passengers):
            total = 0.0
            pos = driver.location
            pickup_at = {}

            # Leg 1: collect everyone. Cumulative distance at the moment each
            # passenger boards, which is what her waiting cost is measured on.
            for p in pick_order:
                total += network.km(pos, p.origin)
                pickup_at[p.id] = total
                pos = p.origin

            # The reported radius, enforced against the realised ordering. The
            # prior filter in feasibility only bounds the driver-to-origin leg;
            # this is the quantity the passenger actually reported a limit on.
            # Checked here rather than after the dropoff loop so that a doomed
            # ordering is abandoned before paying for the rest of it.
            if any(not radius_ok(p, pickup_at[p.id]) for p in passengers):
                continue
            dropoff_at = {}
            # Leg 2: deliver everyone, continuing from the last pickup point.
            for p in drop_order:
                total += network.km(pos, p.destination)
                dropoff_at[p.id] = total
                pos = p.destination

            # What each passenger actually rides: the stretch between boarding
            # and alighting. Time spent waiting for pickup is excluded, since
            # that is governed by r_max rather than delta_max.
            in_vehicle = {p.id: dropoff_at[p.id] - pickup_at[p.id]
                          for p in passengers}

            # Cannot happen with pickups strictly before dropoffs, but kept as
            # a guard: if the leg structure is ever relaxed, this is the
            # invariant that would break first and it should fail loudly rather
            # than produce negative detours.
            if any(in_vehicle[p.id] < 0 for p in passengers):
                continue

            detours = tuple(in_vehicle[p.id] - solo_km(p, network)
                            for p in passengers)

            # The reported constraint, enforced. One passenger over her limit
            # kills the whole ordering: the group is only feasible if everyone
            # in it accepted the route.
            if any(d > p.delta_max for d, p in zip(detours, passengers)):
                continue

            pickups = tuple(pickup_at[p.id] for p in passengers)

            # Minimising total distance, which is the driver's cost. Passenger
            # disutility is priced separately through the valuations, not
            # folded into the routing objective.
            if best is None or total < best[0]:
                best = (total, pickups, detours)

    return best


def route(driver: Driver, passengers: tuple[Passenger, ...], mode: Mode,
          network: ZoneNetwork):
    """Route a candidate group under a given mode.

    The mode is an argument rather than something inferred from group size,
    because in ORA the mode is part of the allocation space: the optimiser
    chooses it. This function just refuses the combination that makes no sense,
    a SOLO group with more than one passenger.
    """
    if mode is Mode.SOLO:
        if len(passengers) != 1:
            return None
        total, pickup, detour = solo_route(driver, passengers[0], network)
        return total, (pickup,), (detour,)
    return shared_route(driver, passengers, network)