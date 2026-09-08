"""Real-geography network builders for ORA using NYC TLC taxi zones and HVFHV.

The synthetic experiments use ``grid_network``.  The real-data validation uses
exactly the same ``ZoneNetwork`` interface, but its zone-to-zone distances are
estimated from observed NYC TLC HVFHV trip distances.

The construction is deliberately two-stage:

1. ``build_real_network`` builds a geometric fallback matrix from Manhattan taxi
   zone centroids (projected EPSG:2263 coordinates, converted from feet to km)
   and a 1.3 street-detour factor.  It also returns the stable TLC LocationID ->
   ORA zone-index mapping needed to scan the trip data.
2. ``build_empirical_trip_distance_network`` replaces every sufficiently
   observed directed O/D cell with the median observed HVFHV ``trip_miles``
   distance for that ordered taxi-zone pair.  Cells with too few observations
   keep the geometric fallback distance.

This makes trip distance part of the real-data validation without changing any
mechanism code: routing, feasibility, payments, baselines, Someya and metrics all
continue to query ``network.km(i, j)`` exactly as before.

Why the median?  TLC trip distances contain route-choice and recording noise.
The median is a robust estimate of a typical road distance for one directed O/D
pair and is much less sensitive to occasional long detours than the mean.

Why keep a fallback?  Even several months of data can leave rare directed pairs
thin.  Using a declared minimum observation count avoids treating one unusual
trip as the canonical distance for an O/D pair while still retaining a complete
square matrix for the exact routing code.
"""

from __future__ import annotations

import os
import numpy as np

from ora.network import ZoneNetwork

# Feet to kilometres. EPSG:2263 is in US survey feet.
_FEET_TO_KM = 0.0003048

# Geometric fallback only. Empirically observed O/D cells replace this value.
_MANHATTAN_DETOUR_FACTOR = 1.3


def _read_lookup(lookup_csv: str, borough: str) -> set[int]:
    """Return TLC LocationIDs belonging to ``borough``."""
    import csv

    keep: set[int] = set()
    with open(lookup_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["Borough"].strip().lower() == borough.strip().lower():
                keep.add(int(row["LocationID"]))
    if not keep:
        raise ValueError(f"No zones found for borough {borough!r} in {lookup_csv}")
    return keep


def _polygon_centroid_xy(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Approximate one polygon centroid by the mean of its exterior vertices."""
    arr = np.asarray(points, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def zone_centroids_from_shapefile(
    shp_path: str,
    lookup_csv: str,
    borough: str = "Manhattan",
):
    """Return stable Manhattan taxi-zone centroids and the LocationID mapping.

    Returns
    -------
    location_ids:
        Sorted TLC LocationIDs kept in the network.
    xy_km:
        ``(n_zones, 2)`` centroid coordinates in km in the source projected CRS.
    loc_to_index:
        Mapping from TLC LocationID to ORA zone index.
    """
    import shapefile  # pyshp

    keep = _read_lookup(lookup_csv, borough)

    reader = shapefile.Reader(shp_path)
    fields = [f[0] for f in reader.fields[1:]]
    loc_field = "LocationID" if "LocationID" in fields else fields[0]
    loc_idx = fields.index(loc_field)

    centroid_by_loc: dict[int, tuple[float, float]] = {}
    for sr in reader.iterShapeRecords():
        loc_id = int(sr.record[loc_idx])
        if loc_id not in keep or loc_id in centroid_by_loc:
            continue
        pts = sr.shape.points
        if not pts:
            continue
        cx_ft, cy_ft = _polygon_centroid_xy(pts)
        centroid_by_loc[loc_id] = (
            cx_ft * _FEET_TO_KM,
            cy_ft * _FEET_TO_KM,
        )

    location_ids = sorted(centroid_by_loc)
    xy_km = np.array([centroid_by_loc[l] for l in location_ids], dtype=float)
    loc_to_index = {loc: idx for idx, loc in enumerate(location_ids)}
    return np.array(location_ids), xy_km, loc_to_index


def planar_distance_matrix(
    xy_km: np.ndarray,
    detour_factor: float = _MANHATTAN_DETOUR_FACTOR,
) -> np.ndarray:
    """Symmetric centroid-distance fallback matrix in km."""
    diff = xy_km[:, None, :] - xy_km[None, :, :]
    straight = np.sqrt((diff ** 2).sum(axis=2))
    return straight * detour_factor


def osm_shortest_path_matrix(*args, **kwargs) -> np.ndarray:
    """Placeholder for a future exact OSM street-network matrix."""
    raise NotImplementedError(
        "OSM shortest-path matrix is a possible future swap. The real-data "
        "validation currently uses observed HVFHV O/D medians with a geometric "
        "fallback for thin pairs."
    )


def build_real_network(
    shp_path: str,
    lookup_csv: str,
    borough: str = "Manhattan",
    avg_speed_kmh: float = 20.0,
    cache_path: str | None = None,
    detour_factor: float = _MANHATTAN_DETOUR_FACTOR,
):
    """Build/load the geometric taxi-zone fallback network and zone mapping.

    This stage exists because the HVFHV scan needs the TLC LocationID -> ORA zone
    mapping before an empirical distance matrix can be estimated.  The returned
    network is therefore a *fallback* network in the Option-B pipeline; the run
    script replaces its well-observed O/D cells with HVFHV medians before any
    mechanism arm is evaluated.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        dist_km = data["dist_km"]
        location_ids = data["location_ids"]
        loc_to_index = {int(l): i for i, l in enumerate(location_ids)}
        return ZoneNetwork(dist_km, avg_speed_kmh), location_ids, loc_to_index

    location_ids, xy_km, loc_to_index = zone_centroids_from_shapefile(
        shp_path,
        lookup_csv,
        borough,
    )
    dist_km = planar_distance_matrix(xy_km, detour_factor)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(
            cache_path,
            dist_km=dist_km,
            location_ids=location_ids,
        )

    return ZoneNetwork(dist_km, avg_speed_kmh), location_ids, loc_to_index


def _empirical_matrix_stats(
    *,
    pool: dict,
    empirical_dist_km: np.ndarray,
    empirical_mask: np.ndarray,
    od_counts: np.ndarray,
    min_observations: int,
) -> dict[str, float | int]:
    """Build paper-ready diagnostics for the empirical/fallback distance mix."""
    n_zones = empirical_dist_km.shape[0]
    off_diag = ~np.eye(n_zones, dtype=bool)
    n_pairs = int(off_diag.sum())
    n_empirical = int((empirical_mask & off_diag).sum())

    observed_counts = od_counts[empirical_mask & off_diag]
    empirical_values = empirical_dist_km[empirical_mask & off_diag]

    total_pool_trips = int(len(pool["direct_km"]))
    empirical_trip_count = int(observed_counts.sum()) if observed_counts.size else 0

    direct = np.asarray(pool["direct_km"], dtype=float)
    finite_direct = direct[np.isfinite(direct) & (direct > 0.0)]

    return {
        "n_zones": n_zones,
        "directed_od_pairs_excluding_diagonal": n_pairs,
        "empirical_od_pairs": n_empirical,
        "fallback_od_pairs": n_pairs - n_empirical,
        "empirical_od_pair_coverage": (
            n_empirical / n_pairs if n_pairs else 0.0
        ),
        "source_pool_trips": total_pool_trips,
        "trips_supporting_empirical_cells": empirical_trip_count,
        "empirical_trip_coverage": (
            empirical_trip_count / total_pool_trips if total_pool_trips else 0.0
        ),
        "min_observations_per_empirical_pair": int(min_observations),
        "median_observations_per_empirical_pair": (
            float(np.median(observed_counts)) if observed_counts.size else 0.0
        ),
        "median_empirical_od_km": (
            float(np.median(empirical_values)) if empirical_values.size else np.nan
        ),
        "source_trip_distance_mean_km": (
            float(finite_direct.mean()) if finite_direct.size else np.nan
        ),
        "source_trip_distance_median_km": (
            float(np.median(finite_direct)) if finite_direct.size else np.nan
        ),
        "source_trip_distance_p25_km": (
            float(np.percentile(finite_direct, 25)) if finite_direct.size else np.nan
        ),
        "source_trip_distance_p75_km": (
            float(np.percentile(finite_direct, 75)) if finite_direct.size else np.nan
        ),
    }


def build_empirical_trip_distance_network(
    pool: dict,
    fallback_network: ZoneNetwork,
    location_ids: np.ndarray,
    *,
    cache_path: str | None = None,
    min_observations: int = 20,
) -> tuple[ZoneNetwork, dict[str, float | int]]:
    """Replace geometric O/D cells with observed median HVFHV trip distances.

    Parameters
    ----------
    pool:
        Output of ``ora.real_demand.extract_pool``.  It must contain ``o_zone``,
        ``d_zone`` and ``direct_km`` arrays in the same zone-index system as
        ``fallback_network``.
    fallback_network:
        Complete geometric Manhattan network returned by ``build_real_network``.
    location_ids:
        TLC LocationIDs in network order.  Stored in the cache so an old matrix
        cannot silently be reused with a different zone ordering.
    cache_path:
        Optional ``.npz`` cache, preferably outside OneDrive.
    min_observations:
        Minimum number of observed trips required before one directed O/D cell
        is replaced by its empirical median.  Thin cells retain the declared
        geometric fallback.  The paper should report this threshold together
        with the resulting empirical-pair coverage.

    Returns
    -------
    network, diagnostics
        A complete ``ZoneNetwork`` plus descriptive coverage statistics.

    Notes
    -----
    The matrix is directed.  If TLC reports different median distances for
    ``i -> j`` and ``j -> i``, both values are retained.  The diagonal remains
    zero because same-zone movements are not part of the routing distance model.
    """
    min_observations = int(min_observations)
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")

    n_zones = fallback_network.n_zones
    location_ids = np.asarray(location_ids, dtype=np.int64)
    if location_ids.shape != (n_zones,):
        raise ValueError(
            "location_ids length must match fallback_network.n_zones"
        )

    required = {"o_zone", "d_zone", "direct_km"}
    missing = required.difference(pool)
    if missing:
        raise KeyError(
            "pool is missing required arrays: " + ", ".join(sorted(missing))
        )

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        cached_location_ids = data["location_ids"].astype(np.int64)
        cached_min = int(data["min_observations"])

        if np.array_equal(cached_location_ids, location_ids) and cached_min == min_observations:
            dist_km = data["dist_km"]
            empirical_mask = data["empirical_mask"].astype(bool)
            od_counts = data["od_counts"].astype(np.int64)
            stats = _empirical_matrix_stats(
                pool=pool,
                empirical_dist_km=dist_km,
                empirical_mask=empirical_mask,
                od_counts=od_counts,
                min_observations=min_observations,
            )
            return (
                ZoneNetwork(dist_km, fallback_network.avg_speed_kmh),
                stats,
            )

    o_zone = np.asarray(pool["o_zone"], dtype=np.int64)
    d_zone = np.asarray(pool["d_zone"], dtype=np.int64)
    direct_km = np.asarray(pool["direct_km"], dtype=float)

    if not (len(o_zone) == len(d_zone) == len(direct_km)):
        raise ValueError("pool O/D/distance arrays must have equal length")

    valid = (
        (o_zone >= 0)
        & (o_zone < n_zones)
        & (d_zone >= 0)
        & (d_zone < n_zones)
        & (o_zone != d_zone)
        & np.isfinite(direct_km)
        & (direct_km > 0.0)
    )

    o = o_zone[valid]
    d = d_zone[valid]
    km = direct_km[valid]

    # Start from the complete geometric fallback so every O/D pair is usable.
    dist_km = np.asarray(fallback_network.dist_km, dtype=float).copy()
    empirical_mask = np.zeros((n_zones, n_zones), dtype=bool)
    od_counts = np.zeros((n_zones, n_zones), dtype=np.int64)

    if km.size:
        # Encode each directed O/D pair as one integer key and sort once.  This
        # avoids a full-data boolean scan for every taxi-zone pair.
        key = o * n_zones + d
        order = np.argsort(key, kind="mergesort")
        key_sorted = key[order]
        km_sorted = km[order]

        starts = np.r_[0, np.flatnonzero(np.diff(key_sorted)) + 1]
        ends = np.r_[starts[1:], len(key_sorted)]

        for start, end in zip(starts, ends):
            pair_key = int(key_sorted[start])
            origin = pair_key // n_zones
            destination = pair_key % n_zones
            count = int(end - start)
            od_counts[origin, destination] = count

            if count < min_observations:
                continue

            median_km = float(np.median(km_sorted[start:end]))
            if not np.isfinite(median_km) or median_km <= 0.0:
                continue

            dist_km[origin, destination] = median_km
            empirical_mask[origin, destination] = True

    np.fill_diagonal(dist_km, 0.0)
    np.fill_diagonal(empirical_mask, False)
    np.fill_diagonal(od_counts, 0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            dist_km=dist_km,
            location_ids=location_ids,
            empirical_mask=empirical_mask,
            od_counts=od_counts,
            min_observations=np.array(min_observations, dtype=np.int64),
        )

    stats = _empirical_matrix_stats(
        pool=pool,
        empirical_dist_km=dist_km,
        empirical_mask=empirical_mask,
        od_counts=od_counts,
        min_observations=min_observations,
    )

    return ZoneNetwork(dist_km, fallback_network.avg_speed_kmh), stats
