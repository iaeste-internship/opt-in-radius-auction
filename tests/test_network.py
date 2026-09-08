from ora.network import grid_network


def test_grid_dimensions():
    n = grid_network(5, 5)
    assert n.n_zones == 25


def test_diagonal_is_zero():
    n = grid_network(5, 5)
    assert n.km(3, 3) == 0.0


def test_symmetry():
    n = grid_network(4, 4)
    assert n.km(2, 7) == n.km(7, 2)


def test_minutes_conversion():
    n = grid_network(3, 3, avg_speed_kmh=20.0)
    assert n.minutes(10.0) == 30.0