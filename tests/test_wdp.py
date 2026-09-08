import numpy as np

from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups, group_welfare
from ora.network import grid_network
from ora.types import Passenger, Driver, Mode
from ora.wdp import solve_wdp


def mk_p(pid, origin, destination, v_solo=20.0, tau=0.75):
    return Passenger(id=pid, origin=origin, destination=destination,
                     v_solo=v_solo, v_share=v_solo * tau,
                     r_max=5.0, delta_max=10.0)


def mk_d(did=0, location=0, capacity=2):
    return Driver(id=did, location=location, capacity=capacity,
                  cost_per_km=0.8, reservation=5.0)


def test_passenger_in_at_most_one_group():
    net = grid_network(5, 5)
    pax = [mk_p(0, 0, 4), mk_p(1, 1, 3)]
    drv = [mk_d(0, 0), mk_d(1, 2)]
    groups, _ = enumerate_groups(drv, pax, net)
    winners, _ = solve_wdp(groups, {p.id: p for p in pax},
                           {d.id: d for d in drv})
    served = [pid for g in winners for pid in g.passenger_ids]
    assert len(served) == len(set(served))


def test_driver_serves_at_most_one_group():
    net = grid_network(5, 5)
    pax = [mk_p(i, i, i + 5) for i in range(4)]
    drv = [mk_d(0, 0, capacity=2)]
    groups, _ = enumerate_groups(drv, pax, net)
    winners, _ = solve_wdp(groups, {p.id: p for p in pax},
                           {d.id: d for d in drv})
    assert len(winners) <= 1


def test_welfare_matches_winners():
    net = grid_network(5, 5)
    pax = [mk_p(i, i, i + 5) for i in range(3)]
    drv = [mk_d(0, 0), mk_d(1, 6)]
    lookup = {p.id: p for p in pax}
    groups, _ = enumerate_groups(drv, pax, net)
    winners, welfare = solve_wdp(groups, lookup, {d.id: d for d in drv})
    driver_lookup = {d.id: d for d in drv}
    assert abs(
        welfare
        - sum(
            group_welfare(g, lookup, driver_lookup)
            for g in winners
        )
    ) < 1e-6

def test_excluding_passenger_never_raises_welfare():
    net = grid_network(5, 5)
    pax = [mk_p(i, i, i + 5) for i in range(3)]
    drv = [mk_d(0, 0), mk_d(1, 6)]
    lookup = {p.id: p for p in pax}
    dlookup = {d.id: d for d in drv}
    groups, _ = enumerate_groups(drv, pax, net)
    _, full = solve_wdp(groups, lookup, dlookup)
    _, without = solve_wdp(groups, lookup, dlookup, exclude_passenger=0)
    assert without <= full + 1e-6


def test_empty_market():
    winners, welfare = solve_wdp([], {}, {})
    assert winners == [] and welfare == 0.0

def test_high_reservation_driver_is_not_selected():
    net = grid_network(3, 3, cell_km=1.0)

    passenger = Passenger(
        id=0,
        origin=0,
        destination=1,
        v_solo=20.0,
        v_share=15.0,
        r_max=10.0,
        delta_max=10.0,
    )

    driver = Driver(
        id=0,
        location=0,
        capacity=1,
        cost_per_km=1.0,
        reservation=100.0,
    )

    groups, _ = enumerate_groups(
        [driver],
        [passenger],
        net,
    )

    winners, welfare = solve_wdp(
        groups,
        {passenger.id: passenger},
        {driver.id: driver},
    )

    assert winners == []
    assert welfare == 0.0