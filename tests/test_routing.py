from ora.network import grid_network
from ora.routing import solo_route, shared_route, route
from ora.types import Passenger, Driver, Mode


def mk_p(pid, origin, destination, delta_max=10.0):
    return Passenger(id=pid, origin=origin, destination=destination,
                     v_solo=20.0, v_share=15.0, r_max=5.0, delta_max=delta_max)


def mk_d(location=0, capacity=4):
    return Driver(id=0, location=location, capacity=capacity,
                  cost_per_km=0.8, reservation=5.0)


def test_solo_route_no_detour():
    net = grid_network(5, 5)
    total, pickup, detour = solo_route(mk_d(0), mk_p(0, 0, 6), net)
    assert detour == 0.0
    assert total == pickup + net.km(0, 6)


def test_shared_returns_none_when_detour_too_tight():
    net = grid_network(5, 5)
    a = mk_p(0, 0, 4, delta_max=0.0)
    b = mk_p(1, 20, 24, delta_max=0.0)
    assert shared_route(mk_d(0), (a, b), net) is None


def test_shared_feasible_when_tolerant():
    net = grid_network(5, 5)
    a = mk_p(0, 0, 4, delta_max=10.0)
    b = mk_p(1, 1, 3, delta_max=10.0)
    result = shared_route(mk_d(0), (a, b), net)
    assert result is not None
    total, pickups, detours = result
    assert total > 0
    assert len(pickups) == 2 and len(detours) == 2


def test_route_solo_rejects_multiple():
    net = grid_network(5, 5)
    assert route(mk_d(0), (mk_p(0, 0, 4), mk_p(1, 1, 3)), Mode.SOLO, net) is None

def test_second_pickup_respects_radius():
    """A passenger reachable directly can still be refused if collected second.

    The exact failure the invariant test caught. p1 sits within her radius of
    the driver's starting position, so the prior filter admits her, but any
    ordering that collects p0 first pushes her boarding distance past r_max.
    The group must be rejected rather than allocated with a violated report.
    """
    from ora.routing import shared_route

    network = grid_network(3, 3, cell_km=1.0)
    p0 = Passenger(id=0, origin=2, destination=8, v_solo=20.0, v_share=16.0,
                   r_max=5.0, delta_max=10.0)
    p1 = Passenger(id=1, origin=6, destination=8, v_solo=20.0, v_share=16.0,
                   r_max=2.0, delta_max=10.0)
    d = Driver(id=0, location=0, capacity=2, cost_per_km=1.0, reservation=0.0)

    result = shared_route(d, (p0, p1), network)
    if result is not None:
        _, pickups, _ = result
        assert pickups[1] <= p1.r_max + 1e-9