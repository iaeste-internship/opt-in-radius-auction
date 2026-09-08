"""Synthetic demand generation for ORA experiments.

Every passenger and driver in the simulation is drawn here. That makes this
module the place where my modelling assumptions about the population live, and
where a reviewer should look first if a result seems too good.

The whole thing is driven by an injected `rng`, never by module-level random
state. That is what makes a run reproducible from its config file alone: one
seed in the YAML, one Generator built from it, passed down. If I ever call
np.random directly anywhere in this project, reproducibility is gone and I
will not notice until someone asks me to regenerate a figure.

Uniform sampling of origins and destinations over zones is the base regime. The
event regime, where demand concentrates in space and time and raises the
density of feasible groupings, needs a non-uniform distribution and is not
implemented yet. That is deliberate ordering: I want the mechanism verified
against a regime I fully understand before adding the one my thesis depends on.
"""
from dataclasses import dataclass
import numpy as np

from ora.types import Passenger, Driver
from ora.network import ZoneNetwork


@dataclass(frozen=True)
class DemandParams:
    """Population parameters for one clearing window.

    Defaults are placeholders chosen to give a working toy market, not
    calibrated values. Before any result goes in the report these need
    grounding: valuations against real fare levels, cost_per_km against fuel
    and vehicle cost, and the flexibility bounds against something defensible.
    Right now they are round numbers that produce a market with matches in it.
    """
    n_passengers: int
    n_drivers: int
    capacity: int = 4

    # Passenger valuations. v_solo is drawn and v_share is derived from it via
    # tau, never drawn independently. Drawing both would produce passengers
    # with v_share > v_solo, which is incoherent: nobody values a shared ride
    # above the same ride taken alone. Deriving guarantees tau <= 1 by
    # construction rather than by rejection sampling.
    v_solo_mean: float = 20.0        # currency units per trip
    v_solo_sd: float = 5.0
    tau_mean: float = 0.75           # share tolerance v_share / v_solo
    tau_sd: float = 0.15

    # Passenger flexibility. r_max is the parameter my headline figure sweeps.
    # In the reviewed simulation papers this is a global constant, usually two
    # or three kilometres and usually set without comment, which silently
    # determines the efficiency they report. Here it is per-passenger and
    # elicited, so the sweep is over the distribution rather than over a
    # platform setting.
    r_max_mean: float = 1.5          # km
    r_max_sd: float = 0.5
    delta_max_mean: float = 2.0      # km
    delta_max_sd: float = 0.7

    # Drivers. cost_per_km is shared across drivers because the base model
    # treats it as observable and vehicle-determined; reservation is drawn
    # per-driver because it is the private component that makes driver-side IR
    # meaningful.
    cost_per_km: float = 0.8
    reservation_mean: float = 5.0
    reservation_sd: float = 1.5


def generate(params: DemandParams, network: ZoneNetwork,
             rng: np.random.Generator) -> tuple[list[Passenger], list[Driver]]:
    """Draw one clearing window of passengers and drivers.

    The rng is passed in, not created here, so that the whole simulation runs
    off a single seeded Generator and every window is reproducible in sequence.
    """
    n_zones = network.n_zones

    passengers = []
    for i in range(params.n_passengers):
        # replace=False so nobody requests a trip to the zone they are already
        # in, which would have zero value and clutter the allocation.
        origin, destination = rng.choice(n_zones, size=2, replace=False)

        # Clamped at 1.0: a non-positive valuation would make tau undefined and
        # would mean a passenger who does not want the ride she requested.
        v_solo = max(1.0, rng.normal(params.v_solo_mean, params.v_solo_sd))

        # Clipped to [0.1, 1.0]. The upper bound is the coherence constraint
        # above. The lower bound stops the normal tail producing passengers who
        # value sharing at essentially nothing, which are numerically awkward
        # and not the interesting case: a passenger with very low tau simply
        # never gets pooled, which the mechanism handles without needing her to
        # be an extreme outlier.
        tau = float(np.clip(rng.normal(params.tau_mean, params.tau_sd), 0.1, 1.0))

        passengers.append(Passenger(
            id=i,
            origin=int(origin),
            destination=int(destination),
            v_solo=v_solo,
            v_share=v_solo * tau,
            # Floored at 0.1 km so that a draw from the left tail does not
            # produce a passenger who is unreachable by construction.
            r_max=max(0.1, rng.normal(params.r_max_mean, params.r_max_sd)),
            delta_max=max(0.1, rng.normal(params.delta_max_mean, params.delta_max_sd)),
        ))

    drivers = []
    for j in range(params.n_drivers):
        drivers.append(Driver(
            id=j,
            # Uniform over zones. Real fleets are not uniformly distributed,
            # they cluster where demand was recently. That matters for the
            # event regime and is a limitation to state rather than to fix now.
            location=int(rng.integers(n_zones)),
            capacity=params.capacity,
            cost_per_km=params.cost_per_km,
            reservation=max(0.0, rng.normal(params.reservation_mean,
                                            params.reservation_sd)),
        ))

    return passengers, drivers