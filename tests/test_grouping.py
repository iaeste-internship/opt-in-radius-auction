from unicodedata import lookup

import numpy as np

from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups, group_welfare
from ora.network import grid_network
from ora.types import Passenger, Driver, Mode


def mk_p(pid, origin, destination, r_max=5.0, delta_max=10.0):
    return Passenger(id=pid, origin=origin, destination=destination,
                     v_solo=20.0, v_share=15.0, r_max=r_max, delta_max=delta_max)


def mk_d(did=0, location=0, capacity=2):
    return Driver(id=did, location=location, capacity=capacity,
                  cost_per_km=0.8, reservation=5.0)


def test_single_passenger_is_solo():
    net = grid_network(5, 5)
    groups, _ = enumerate_groups([mk_d()], [mk_p(0, 0, 6)], net)
    assert len(groups) == 1
    assert groups[0].mode is Mode.SOLO


def test_capacity_bounds_group_size():
    net = grid_network(5, 5)
    pax = [mk_p(i, i, i + 10) for i in range(4)]
    groups, _ = enumerate_groups([mk_d(capacity=2)], pax, net)
    assert all(len(g.passenger_ids) <= 2 for g in groups)


def test_radius_prunes_unreachable():
    net = grid_network(5, 5, cell_km=0.5)
    far = mk_p(0, 24, 20, r_max=0.1)
    groups, _ = enumerate_groups([mk_d(location=0)], [far], net)
    assert groups == []


def test_stats_are_populated():
    net = grid_network(5, 5)
    pax = [mk_p(i, i, i + 5) for i in range(3)]
    groups, stats = enumerate_groups([mk_d(capacity=3)], pax, net)
    assert stats.candidates_considered > 0
    assert stats.groups_generated == len(groups)


def test_welfare_uses_share_value_when_shared():
    net = grid_network(5, 5)
    a, b = mk_p(0, 0, 4), mk_p(1, 1, 3)
    groups, _ = enumerate_groups([mk_d(capacity=2)], [a, b], net)
    shared = [g for g in groups if g.mode is Mode.SHARED]
    assert shared
    lookup = {0: a, 1: b}
    driver_lookup = {0: mk_d()}
    w = group_welfare(
        shared[0],
        lookup,
        driver_lookup,
    )

    assert w == (
        a.v_share
        + b.v_share
        - shared[0].route_cost
        - driver_lookup[0].reservation
    )


def test_runs_on_generated_demand():
    net = grid_network(5, 5)
    rng = np.random.default_rng(0)
    pax, drv = generate(DemandParams(n_passengers=15, n_drivers=4,
                                     capacity=3), net, rng)
    groups, stats = enumerate_groups(drv, pax, net)
    assert stats.groups_generated == len(groups)