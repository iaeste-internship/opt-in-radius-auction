"""Passenger acceptability: what a served ride is worth once bounds are checked.

Two accounting conventions live here, and the gap between them is the point.

The nominal convention is the one every mechanism in this study already assumes
about itself: a served passenger is worth v_solo when she rides alone and
v_share when she is pooled, whatever route she actually got. Under this
convention a passenger collected two kilometres past the radius she reported is
booked at exactly the same value as one collected next door, because the
valuation is flat in the route.

The adjusted convention charges for the part of the route that exceeded what the
passenger said she would accept. Only the excess is charged: a ride inside the
reported bounds costs nothing, and a ride that overshoots is penalised in
proportion to the overshoot. The per-unit rate is not a new parameter. It is
derived from the flexibility bounds the passenger already reports, by the same
rule closed for the Someya arm, so this module owns the derivation and someya.py
imports it rather than keeping a private copy.

Why the adjusted convention rather than treating any breach as an outright
refusal. Declaring a ride worthless the moment it exceeds a bound by one metre
applies ORA's own hard-constraint reading to mechanisms that never claimed it,
and it treats a trivial overshoot and an egregious one identically. Charging in
proportion uses the soft-penalty logic that the compared mechanism itself uses,
which makes the yardstick one it cannot object to, and it grades the damage.

The same formula is applied to every arm. ORA's penalty comes out at zero
because its feasibility never admits a group that breaches a bound, but that is
a measured result rather than an exemption: if ORA ever did breach one, this
metric would show it, which makes it a sanity check on the feasibility layer as
well as a comparison instrument.
"""
from __future__ import annotations

from ora.types import Passenger, Mode
from ora.network import ZoneNetwork


def omega_for(passenger: Passenger,
              network: ZoneNetwork) -> tuple[float, float]:
    """Per-unit disutility of detour distance and of waiting time.

    A passenger reports the bounds delta_max and r_max rather than the monetary
    rates a soft-penalty model would ask for, so the rates are derived from the
    bounds. delta_max is the detour at which she stops accepting the pooled
    ride: the point where the nuisance of the detour cancels the surplus that
    sharing gives her over riding alone, v_solo - v_share. Setting

        omega_det = (v_solo - v_share) / delta_max

    makes the penalty reach exactly that surplus at exactly her stated
    tolerance, so the hard bound and the soft rate agree at the boundary and
    differ only past it. r_max plays the same role for waiting, converted to
    minutes through the network's declared speed so the rate is per unit time.

    A bound of zero means no tolerance at all, and the rate is infinite. Callers
    must therefore skip the multiplication when the excess is zero, since
    0 * inf is nan rather than 0; `excess_penalty` below does exactly that.
    """
    surplus = passenger.v_solo - passenger.v_share

    omega_det = (
        float("inf") if passenger.delta_max <= 0.0
        else surplus / passenger.delta_max
    )

    r_max_min = network.minutes(passenger.r_max)
    omega_wait = (
        float("inf") if r_max_min <= 0.0
        else surplus / r_max_min
    )

    return omega_det, omega_wait


def excess_penalty(passenger: Passenger,
                   detour_km: float,
                   wait_min: float,
                   network: ZoneNetwork) -> float:
    """What the parts of this ride beyond the passenger's bounds cost her.

    Only the overshoot is charged. A ride within both bounds returns zero, so
    a mechanism whose feasibility respects the reported bounds pays nothing
    here and one that ignores them pays in proportion to how far it went.

    Returns infinity when a passenger with zero tolerance is made to travel any
    excess at all, which callers should read as a ride she would not have taken
    under any price.
    """
    omega_det, omega_wait = omega_for(passenger, network)

    detour_excess = max(0.0, detour_km - passenger.delta_max)
    wait_excess = max(0.0, wait_min - network.minutes(passenger.r_max))

    # Skip the multiplication when there is no excess: an infinite rate times a
    # zero excess is nan, and the honest answer there is no penalty at all.
    penalty = 0.0
    if detour_excess > 0.0:
        penalty += omega_det * detour_excess
    if wait_excess > 0.0:
        penalty += omega_wait * wait_excess

    return penalty


def nominal_value(passenger: Passenger, mode: Mode) -> float:
    """The passenger's valuation as every mechanism here already books it.

    Flat in the route: v_solo riding alone, v_share pooled, regardless of how
    far the route strayed from what she said she would accept. Matches
    grouping.group_welfare so the per-passenger figures reconcile with the
    aggregate welfare the arms are scored on.
    """
    return passenger.v_solo if mode is Mode.SOLO else passenger.v_share
