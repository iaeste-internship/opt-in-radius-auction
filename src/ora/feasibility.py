"""Feasibility filters: which passenger subsets a driver may serve.

These are the constraints the passenger reported, enforced. r_max and
delta_max are components of her private type, so a group that violates either
is not a group the mechanism is allowed to consider at all. That is the point
of eliciting them: the platform no longer decides how far it will search on
her behalf or how much detour she will tolerate.

The functions here split into two kinds, and the split is what makes the
enumeration tractable.

Cheap and prior: within_radius and capacity_ok depend only on the reported
type and the driver's position. They can be applied before any subset is
generated, which shrinks the ground set the combinatorial enumeration runs
over.

Expensive and posterior: detour_ok cannot be evaluated until a route exists,
because the realised in-vehicle distance depends on the pickup and dropoff
ordering. So it is applied inside the router, once per candidate ordering, not
here.

Getting that order wrong is what makes hypergraph enumeration blow up. Someya
& Namerikawa tested on 120 requests and admitted the combinatorial explosion,
suggesting spatio-temporal pruning without implementing it. My measurements
show 79% to 91% of candidate subsets dying at the routing stage, rising with
market size, which says the prior filters are doing real work but that a
cheaper detour bound applied before routing would help more the larger the
market gets. Noted as an optimisation to make only if profiling justifies it.
"""
from ora.types import Passenger, Driver
from ora.network import ZoneNetwork


def pickup_km(driver: Driver, passenger: Passenger,
              network: ZoneNetwork) -> float:
    """Distance from where the driver is now to where the passenger is."""
    return network.km(driver.location, passenger.origin)


def solo_km(passenger: Passenger, network: ZoneNetwork) -> float:
    """Direct origin-to-destination distance, the no-detour baseline.

    This is the reference every detour is measured against, so it is defined
    once here rather than recomputed inline wherever it is needed.
    """
    return network.km(passenger.origin, passenger.destination)


def within_radius(driver: Driver, passenger: Passenger,
                  network: ZoneNetwork) -> bool:
    """Whether this passenger accepts being searched over by this driver.

    The cheap prior filter. Applied to the full passenger list once per driver
    before any subset enumeration begins.
    """
    return pickup_km(driver, passenger, network) <= passenger.r_max


def detour_ok(passenger: Passenger, in_vehicle_km: float,
              network: ZoneNetwork) -> bool:
    """Whether a realised route respects the detour this passenger accepted.

    in_vehicle_km is her dropoff distance minus her pickup distance along the
    driver's route, so it counts only the part of the trip she is actually in
    the car for. Waiting for pickup is a separate cost and is measured by
    r_max, not by this.

    Called from the router, once per candidate ordering, because it needs a
    route to exist first.
    """
    return in_vehicle_km - solo_km(passenger, network) <= passenger.delta_max

def radius_ok(passenger: Passenger, pickup_km_along_route: float) -> bool:
    """Whether a realised route picks this passenger up within her radius.

    The posterior half of the radius check, and the one that actually binds.
    r_max is the distance travelled before she boards, counted from the start
    of the driver's route, so it includes any earlier pickups. That is the
    quantity she cares about: it is what her waiting time is proportional to,
    and it is what the metrics layer converts to minutes.

    within_radius above is the cheap prior approximation of this, using the
    straight driver-to-origin distance. It is a valid lower bound, since
    collecting other passengers first can only add distance, so it prunes
    soundly. But it is not sufficient: a passenger second in a pickup order can
    pass the prior filter and still be reached late. This function is what
    catches that, and it runs inside the router once per candidate ordering.

    I had this wrong initially. The prior filter alone let groups through whose
    stored pickup distance exceeded the reported radius, which the invariant
    test caught across four seeds.
    """
    return pickup_km_along_route <= passenger.r_max

def capacity_ok(driver: Driver, group_size: int) -> bool:
    """Vehicle capacity. The other cheap prior filter."""
    return group_size <= driver.capacity


def compatible_passengers(driver: Driver, passengers: list[Passenger],
                          network: ZoneNetwork) -> list[Passenger]:
    """The ground set for this driver's subset enumeration.

    Everything outside this list is unreachable by construction, so restricting
    to it before enumerating subsets is where most of the pruning happens: the
    number of subsets is exponential in the size of this list, so cutting the
    list is worth far more than cutting subsets afterwards.
    """
    return [p for p in passengers if within_radius(driver, p, network)]