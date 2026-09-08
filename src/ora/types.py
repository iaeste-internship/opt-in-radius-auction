"""Core types for the Opt-In Radius Auction (ORA).

This module defines the vocabulary of the mechanism. Nothing here computes
anything: it fixes what an agent is, what a feasible assignment looks like,
and what a cleared market produces. Every other module speaks in these terms.

Two modelling choices are encoded here, and both are deliberate.

First, ORA moves two decisions from the planner to the agent. In the papers I
reviewed, the platform decides whether a passenger gets pooled and how far it
will search for a driver on her behalf. Here both are private type components
she reports (v_share and r_max). That is what makes sharing an allocated good
with a derived price instead of a discount the platform applies.

Second, everything is in kilometres, never minutes. The mechanism works in
distance from start to finish so that no traffic assumption is baked into the
allocation. Conversion to time happens only in the metrics layer, through one
declared average speed (see network.ZoneNetwork.minutes). If someone asks
where the waiting time numbers come from, the answer is one value in a config
file rather than a constant buried in the code.
"""
from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    """Travel mode of an allocated group.

    The mode is part of the allocation space, not a label attached afterwards.
    A feasible allocation assigns each passenger to a group AND to a mode, and
    the optimiser maximises reported welfare over that joint space.

    This is the one structural change I make to hypergraph VCG, and it is what
    makes the price gap between private and shared rides fall out of the
    payment rule instead of being assumed as a fixed discount fraction. Since
    the mode enters only through the valuation inside the welfare objective,
    Clarke payments stay strategy-proof over the enlarged type space without
    any new proof work.
    """
    SOLO = "solo"
    SHARED = "shared"


@dataclass(frozen=True)
class Passenger:
    """A passenger and her reported type theta_i.

    The type has four components: (v_solo, v_share, r_max, delta_max). The
    first two price the mode, the last two constrain which groups are feasible
    for her. Prior work elicits at most the first.

    Frozen because a reported type must not change during a clearing window.
    Mutation would break the strategy-proofness argument silently, since it
    assumes one fixed report per window.
    """
    id: int
    origin: int          # zone index
    destination: int     # zone index

    # Trip valuation under each mode. Both are per-trip and quasi-linear in
    # money, which is what VCG requires.
    v_solo: float        # value of a private ride
    v_share: float       # value of the same trip, shared

    # Flexibility bounds. These are the passenger's own limits, not global
    # engineering constants. Making r_max passenger-specific is what turns the
    # matching radius from a hidden parameter into an elicited one.
    r_max: float         # max pickup distance accepted (km)
    delta_max: float     # max detour tolerated (km)

    @property
    def tau(self) -> float:
        """Share tolerance: v_share / v_solo.

        The most informative single number in the type. A passenger with tau
        near one is almost indifferent between modes and is cheap to pool. A
        passenger with tau near 0.4 strongly prefers a private ride and will
        not be pooled unless the externality justifies it. The distribution of
        tau across the population is the experimental variable that shifts the
        turning point of the radius sweep, which is my headline empirical
        claim.

        Well defined because the demand generator enforces v_solo > 0.
        """
        return self.v_share / self.v_solo


@dataclass(frozen=True)
class Driver:
    """A driver with a private reservation value.

    Making the driver an agent rather than a fixed-rate executor is the second
    axis of my extension. It is also what puts the design inside
    Myerson-Satterthwaite: with two-sided individual rationality and private
    values, incentive compatibility, budget balance and efficiency cannot all
    hold at once. My planned response is to drop strict budget balance and
    report the resulting platform deficit as the measured price of two-sided
    IR.

    Only `reservation` is private. `cost_per_km` is treated as observable in
    the base model, since it follows from fuel and vehicle type, which the
    platform knows. Shi et al. (2024) make the marginal cost private instead;
    relaxing this is the natural extension if I have time.
    """
    id: int
    location: int        # zone index
    capacity: int        # max passengers carried at once
    cost_per_km: float   # marginal operating cost, assumed observable
    reservation: float   # private opportunity cost of participating


@dataclass(frozen=True)
class Group:
    """A feasible hyperedge: one driver, a set of passengers, one mode.

    This is the unit the winner determination problem selects over. Following
    the hypergraph formulation, the cost of routing the group is precomputed
    and stored here, which turns matching into a clean max-weight set packing
    problem that VCG applies to exactly.

    Constructing a Group asserts feasibility: capacity, pickup radius and
    detour tolerance have all been checked already. An infeasible combination
    is never represented, it is simply absent from the enumeration.
    """
    driver_id: int
    passenger_ids: tuple[int, ...]
    mode: Mode

    route_km: float                   # total distance driven to serve the group
    route_cost: float                 # route_km * driver.cost_per_km

    # Per-passenger outcomes, positionally aligned with `passenger_ids`.
    # Stored rather than recomputed because the metrics layer needs them and
    # re-deriving would mean solving the routing problem again.
    pickup_km: tuple[float, ...]      # driver-to-origin distance per passenger
    detour_km: tuple[float, ...]      # realised minus direct distance


@dataclass(frozen=True)
class Allocation:
    """The output of one clearing window.

    The baselines (greedy nearest-neighbour, static surge) return this same
    type, so the metrics layer cannot tell which mechanism produced it. That is
    what keeps the comparison honest: identical measurement code on every arm.
    """
    groups: tuple[Group, ...]
    payments: dict[int, float]        # passenger_id -> payment to the platform
    driver_earnings: dict[int, float] # driver_id -> payment from the platform