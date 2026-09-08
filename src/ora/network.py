"""Zone network: the distance matrix everything else queries.

The mechanism never touches a road graph directly. It asks one question over
and over, "how far is zone i from zone j", and that question has to be answered
in constant time because the routing step asks it inside a factorial loop over
pickup and dropoff orderings.

So the design is a precomputed zone-to-zone matrix. Partition the city into
zones once, compute all-pairs distances once, cache to disk, and from then on
every query is an array lookup. A 300-zone matrix is 90,000 floats, which is
nothing in memory and removes routing entirely from the inner loop. Contraction
hierarchies, which is what production platforms use, would be faster still but
are not worth implementing for this project: the science is in the mechanism,
not in the shortest-path engine.

The same class serves the toy grid I develop against and the real OSM network I
run experiments on. `grid_network` builds the first, and a future `osm_network`
will build the second by downloading a street graph, partitioning it into zones
and running all-pairs shortest paths. Both return a ZoneNetwork, so nothing
downstream knows or cares which one it got. That is deliberate: I want to debug
the mechanism on 25 zones with hand-checkable distances before adding the
uncertainty of real geography.
"""
import numpy as np


class ZoneNetwork:
    """Zone-to-zone distance matrix in km, plus the one time conversion.

    Distances are the currency of the whole mechanism. Time appears only here,
    and only through `minutes`, so that the average speed is a single declared
    parameter rather than an assumption scattered through the code.
    """

    def __init__(self, dist_km: np.ndarray, avg_speed_kmh: float = 20.0):
        # Square matrix is the only invariant worth asserting: everything
        # downstream indexes it symmetrically by zone id.
        assert dist_km.shape[0] == dist_km.shape[1]
        self.dist_km = dist_km
        self.avg_speed_kmh = avg_speed_kmh

    @property
    def n_zones(self) -> int:
        return self.dist_km.shape[0]

    def km(self, i: int, j: int) -> float:
        """Distance from zone i to zone j.

        Called from inside the routing permutation loop, so it stays a plain
        array lookup with no bounds checking or caching layer on top.
        """
        return float(self.dist_km[i, j])

    def minutes(self, km: float) -> float:
        """Convert a distance to travel time at the declared average speed.

        The only place in the project where distance becomes time. The problem
        statement asks for average and 95th-percentile waiting time and for
        Gini/Jain over waiting time, so the metrics layer needs this. Keeping
        it out of the mechanism means the allocation itself carries no traffic
        assumption: change the speed and the matching does not move, only the
        reported times do.
        """
        return 60.0 * km / self.avg_speed_kmh


def grid_network(rows: int, cols: int, cell_km: float = 0.5,
                 avg_speed_kmh: float = 20.0) -> ZoneNetwork:
    """Toy network: Manhattan distance on a rows x cols grid.

    Used for development and for every unit test, because distances are
    predictable by hand and a failing test points at the mechanism rather than
    at the geography.

    Manhattan distance rather than Euclidean, and that is not laziness.
    Euclidean distance on a grid systematically underestimates real street
    travel, and my whole argument about the search radius depends on how much
    it costs to divert a driver. An optimistic metric would produce curves that
    then fail to reproduce on the real network. Better for the toy case to be
    pessimistic in the same direction as reality.
    """
    coords = np.array([(r * cell_km, c * cell_km)
                       for r in range(rows) for c in range(cols)])
    # Broadcasting: |x_i - x_j| + |y_i - y_j| for every pair at once.
    dist = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    return ZoneNetwork(dist, avg_speed_kmh)