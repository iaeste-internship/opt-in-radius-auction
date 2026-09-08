from ora.feasibility import (within_radius, capacity_ok, detour_ok,
                            compatible_passengers, solo_km)
from ora.network import grid_network
from ora.types import Passenger, Driver


def make_passenger(pid=0, origin=0, destination=1, r_max=1.0, delta_max=1.0):
    return Passenger(id=pid, origin=origin, destination=destination,
                     v_solo=20.0, v_share=15.0, r_max=r_max, delta_max=delta_max)


def make_driver(did=0, location=0, capacity=4):
    return Driver(id=did, location=location, capacity=capacity,
                  cost_per_km=0.8, reservation=5.0)


def test_radius_accepts_colocated():
    net = grid_network(5, 5)
    assert within_radius(make_driver(location=0), make_passenger(origin=0), net)


def test_radius_rejects_far():
    net = grid_network(5, 5, cell_km=0.5)
    # zone 0 to zone 24 is 4.0 km on the toy grid
    assert not within_radius(make_driver(location=0),
                             make_passenger(origin=24, r_max=1.0), net)


def test_capacity():
    d = make_driver(capacity=3)
    assert capacity_ok(d, 3)
    assert not capacity_ok(d, 4)


def test_detour_zero_is_ok():
    net = grid_network(5, 5)
    p = make_passenger(origin=0, destination=6, delta_max=0.5)
    direct = solo_km(p, net)
    assert detour_ok(p, direct, net)


def test_detour_beyond_tolerance():
    net = grid_network(5, 5)
    p = make_passenger(origin=0, destination=6, delta_max=0.5)
    direct = solo_km(p, net)
    assert not detour_ok(p, direct + 1.0, net)


def test_compatible_filters():
    net = grid_network(5, 5, cell_km=0.5)
    d = make_driver(location=0)
    near = make_passenger(pid=1, origin=1, r_max=2.0)
    far = make_passenger(pid=2, origin=24, r_max=0.5)
    result = compatible_passengers(d, [near, far], net)
    assert [p.id for p in result] == [1]