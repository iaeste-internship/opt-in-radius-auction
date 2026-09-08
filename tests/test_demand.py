import numpy as np
from ora.demand import DemandParams, generate
from ora.network import grid_network


def test_counts():
    net = grid_network(5, 5)
    rng = np.random.default_rng(0)
    pax, drv = generate(DemandParams(n_passengers=20, n_drivers=5), net, rng)
    assert len(pax) == 20
    assert len(drv) == 5


def test_share_never_exceeds_solo():
    net = grid_network(5, 5)
    rng = np.random.default_rng(0)
    pax, _ = generate(DemandParams(n_passengers=200, n_drivers=10), net, rng)
    assert all(p.v_share <= p.v_solo for p in pax)


def test_origin_differs_from_destination():
    net = grid_network(5, 5)
    rng = np.random.default_rng(1)
    pax, _ = generate(DemandParams(n_passengers=200, n_drivers=10), net, rng)
    assert all(p.origin != p.destination for p in pax)


def test_reproducible():
    net = grid_network(5, 5)
    a = generate(DemandParams(n_passengers=30, n_drivers=6), net,
                 np.random.default_rng(42))
    b = generate(DemandParams(n_passengers=30, n_drivers=6), net,
                 np.random.default_rng(42))
    assert a == b