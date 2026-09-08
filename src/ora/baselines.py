"""Comparison arms: the mechanisms ORA is measured against.

Every arm here returns the same Allocation type ORA returns, so the metrics
layer cannot tell which one produced its input. That is what keeps the
comparison honest: identical measurement code on every arm.

The ablation is the scientifically load-bearing one and the reason this module
exists. Comparing ORA to a greedy matcher measures exact-versus-approximate,
which is Someya's result rather than mine. Comparing ORA to the same exact VCG
with the elicitation removed measures what the elicitation buys, which is my
contribution and nothing else.

The measurement convention that makes the ablation mean anything.

The ablation platform does not know each passenger's share tolerance or pickup
radius, so it assumes: one discount fraction for everybody, one radius for
everybody. It then allocates on those assumptions. But the passengers are still
the passengers they were, with the valuations and limits they actually hold.

So the arm ALLOCATES with assumed types and is MEASURED with true types. Any
other convention hides the effect. Measuring with the assumed types would
score the platform against its own fiction: it would look like it delivered
exactly the welfare it thought it was delivering, and the cost of guessing
wrong would vanish from the results. Passing the true passengers to
compute_metrics is what makes that cost visible, and it is the whole point.

The same applies to r_max. The ablation uses a global radius, so it can collect
a passenger further out than she would actually have accepted. ORA cannot do
that by construction. `radius_violations` below counts it, and that count is a
result rather than a diagnostic: it is the harm the elicitation prevents,
measured directly rather than argued for.
"""
from dataclasses import dataclass, replace

from ora.types import Passenger, Driver, Allocation
from ora.network import ZoneNetwork
from ora.grouping import enumerate_groups
from ora.payments import clear_market, ClearingResult


# Literature default. The reviewed simulation papers fix a matching radius,
# usually two or three kilometres, and usually without comment. Two is the
# lower end of that range and is what my headline figure sweeps away from.
DEFAULT_RADIUS_KM = 2.0

# Literature default for the pooling discount. Standard pooling models assume a
# shared seat is worth some fixed fraction of the solo fare. The fraction has
# no economic justification, which is exactly the free parameter ORA removes.
DEFAULT_SHARE_DISCOUNT = 0.75


def ablation_passengers(passengers: list[Passenger],
                        radius_km: float = DEFAULT_RADIUS_KM,
                        share_discount: float = DEFAULT_SHARE_DISCOUNT
                        ) -> list[Passenger]:
    """Replace the elicited components with platform-wide constants.

    This is the whole ablation: same passengers, same geography, same exact
    solver, same Clarke payments. Only the two things ORA elicits are taken
    back off the passenger and set by the platform.

    delta_max is deliberately left alone. It is a detour tolerance, and while
    ORA elicits it too, it is not one of the two decisions the contribution is
    about. Overwriting it as well would blur the comparison: I would no longer
    be able to say which of the differences came from the pooling decision and
    the search radius specifically.
    """
    return [replace(p,
                    r_max=radius_km,
                    v_share=p.v_solo * share_discount)
            for p in passengers]


def run_ablation(passengers: list[Passenger],
                 drivers: list[Driver],
                 network: ZoneNetwork,
                 radius_km: float = DEFAULT_RADIUS_KM,
                 share_discount: float = DEFAULT_SHARE_DISCOUNT
                 ) -> ClearingResult:
    """Clear one window under the ablation.

    Returns a ClearingResult whose Allocation refers to the real passenger ids,
    so it can be measured against the true passenger list directly. The assumed
    types exist only inside this function and never escape it.
    """
    assumed = ablation_passengers(passengers, radius_km, share_discount)
    groups, _ = enumerate_groups(drivers, assumed, network)
    return clear_market(groups,
                        {p.id: p for p in assumed},
                        {d.id: d for d in drivers})


def radius_violations(allocation: Allocation,
                      passengers: list[Passenger]) -> tuple[int, float]:
    """How many served passengers were collected beyond their true r_max.

    Returns (count, fraction of served). Zero for ORA by construction, because
    the reported radius is enforced inside the router. Non-zero for any arm
    that uses a global radius, and that gap is the elicitation made visible.

    A violation here is not a bug in the ablation. It is the ablation working
    exactly as the literature specifies: the platform picked a radius, and some
    passengers fell outside what they would have accepted. Nobody measures this
    because nobody elicits the bound to measure against.
    """
    by_id = {p.id: p for p in passengers}
    served = 0
    violated = 0
    for g in allocation.groups:
        for pid, km in zip(g.passenger_ids, g.pickup_km):
            served += 1
            if km > by_id[pid].r_max + 1e-9:
                violated += 1
    return violated, (violated / served if served else 0.0)


def radius_only_ablation_passengers(
    passengers: list[Passenger],
    radius_km: float = DEFAULT_RADIUS_KM,
) -> list[Passenger]:
    """Replace only passenger-specific pickup radii with one global radius.

    Share valuations remain passenger-specific. This isolates the contribution
    of eliciting r_max while leaving the pooling-preference information intact.
    """
    return [replace(p, r_max=radius_km) for p in passengers]


def share_only_ablation_passengers(
    passengers: list[Passenger],
    share_discount: float = DEFAULT_SHARE_DISCOUNT,
) -> list[Passenger]:
    """Replace only passenger-specific shared-ride values with one discount.

    Pickup radii remain passenger-specific. This isolates the contribution of
    eliciting v_share / tau while keeping individual pickup acceptability.
    """
    return [
        replace(p, v_share=p.v_solo * share_discount)
        for p in passengers
    ]


def run_radius_only_ablation(
    passengers: list[Passenger],
    drivers: list[Driver],
    network: ZoneNetwork,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> ClearingResult:
    """Clear with a global pickup radius but true passenger share valuations."""
    assumed = radius_only_ablation_passengers(passengers, radius_km)
    groups, _ = enumerate_groups(drivers, assumed, network)
    return clear_market(
        groups,
        {p.id: p for p in assumed},
        {d.id: d for d in drivers},
    )


def run_share_only_ablation(
    passengers: list[Passenger],
    drivers: list[Driver],
    network: ZoneNetwork,
    share_discount: float = DEFAULT_SHARE_DISCOUNT,
) -> ClearingResult:
    """Clear with a common share discount but true passenger pickup radii."""
    assumed = share_only_ablation_passengers(passengers, share_discount)
    groups, _ = enumerate_groups(drivers, assumed, network)
    return clear_market(
        groups,
        {p.id: p for p in assumed},
        {d.id: d for d in drivers},
    )
