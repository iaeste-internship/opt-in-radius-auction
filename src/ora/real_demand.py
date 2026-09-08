"""Real ride-hailing demand for ORA, sampled from the NYC TLC HVFHV records.

This is the demand side of the real-data validation. The synthetic `generate`
in demand.py draws origins and destinations uniformly over zones; this module
draws them from real High-Volume For-Hire Vehicle trips (Uber / Lyft / Via)
instead. Everything else about a passenger — the valuations and the pickup and
detour tolerances — is drawn from exactly the same distributions demand.py
uses. That is the whole point of the experiment: geography and demand become
real, and nothing else moves, so any change in the results is attributable to
the demand, not to a quietly retuned valuation model.

Two stages, split on purpose:

  extract_pool  runs once. An HVFHV month is ~20 million rows, far too large to
                load whole on the WDAC / OneDrive machine. So this pushes the
                zone filter down into the Parquet scan with pyarrow.dataset,
                keeps only the six columns that matter, and writes a slim cached
                pool (a few MB). Every later run reads that cache, never the raw
                file. A directory of month files works too, which is the easy
                way to add more variety: point it at several months.

  generate_from_trips  runs per market. It matches the (passengers, drivers)
                contract of demand.generate, so the run script swaps one sampler
                for the other and nothing else changes. Windows are spread
                across the full day by sampling a daypart in proportion to real
                demand mass, so off-peak markets appear as a real minority
                rather than being either absent or over-represented.

Decision recorded here: we keep ALL ride-hailing trips, not only the ones whose
shared_request_flag is set. The flag is carried through as observed data so the
paper can report what fraction of the real market actually opted into pooling,
which is more realistic and more varied than pre-filtering to shared requests.
Flip KEEP_ALL_TRIPS to False to get the shared-only pool instead.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from ora.types import Passenger, Driver

_MILES_TO_KM = 1.609344

KEEP_ALL_TRIPS = True

# Dayparts used to spread the sampled windows across the whole day. Coarse
# enough that even the thin hours carry real trips, fine enough that peak and
# off-peak are genuinely different strata. Each is a half-open [start, end) hour
# range. Crossed with weekday/weekend, these are the strata demand mass is
# measured over.
DEFAULT_DAYPARTS = (
    ("early", 0, 6),
    ("am_peak", 6, 10),
    ("midday", 10, 16),
    ("pm_peak", 16, 20),
    ("evening", 20, 24),
)

# HVFHV column names. Yellow taxi uses different names, which is exactly why
# this module is separate: the schema is part of the dataset choice.


def _source_signature(parquet_path: str) -> str:
    """Fingerprint the Parquet source set for safe cache reuse.

    The signature uses file names, sizes and nanosecond modification times. It
    is intentionally cheap: the point is to detect adding/removing/replacing
    monthly TLC files, not to hash tens of gigabytes of Parquet contents.
    """
    root = Path(parquet_path)
    if root.is_file():
        files = [root]
        base = root.parent
    elif root.is_dir():
        files = sorted(root.rglob("*.parquet"))
        base = root
    else:
        raise FileNotFoundError(f"HVFHV Parquet source not found: {parquet_path}")

    if not files:
        raise FileNotFoundError(
            f"No .parquet files found under HVFHV source: {parquet_path}"
        )

    lines = []
    for file in files:
        stat = file.stat()
        try:
            rel = file.relative_to(base)
        except ValueError:
            rel = file.name
        lines.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}")

    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

_COLS = [
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "request_datetime",
    "shared_request_flag",
    "hvfhs_license_num",
]


def extract_pool(parquet_path: str,
                 loc_to_index: dict[int, int],
                 cache_path: str | None = None,
                 keep_all_trips: bool = KEEP_ALL_TRIPS) -> dict:
    """Read the HVFHV file(s) once and return a slim in-memory trip pool.

    `parquet_path` may be a single .parquet file or a directory of them. Only
    trips whose pickup AND drop-off zones are in `loc_to_index` survive, so the
    pool is already restricted to the borough the network was built for.

    The returned pool is a dict of NumPy arrays, all the same length:
        o_zone, d_zone   passenger origin / destination as ZONE INDICES
        direct_km        real trip distance, converted from miles
        hour             pickup hour 0..23
        is_weekend       bool
        shared_request   bool, the observed pool opt-in flag
    Drop-off zone indices are also kept separately as `do_zone` for placing
    drivers from the real drop-off distribution.

    If `cache_path` exists it is loaded instead of re-reading the raw file.
    """
    source_signature = _source_signature(parquet_path)

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        cached_signature = (
            str(data["__source_signature__"].item())
            if "__source_signature__" in data.files
            else None
        )
        if cached_signature == source_signature:
            return {
                k: data[k]
                for k in data.files
                if not k.startswith("__")
            }
        print(
            "HVFHV source files changed; rebuilding cached Manhattan trip pool."
        )

    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    keep_ids = list(loc_to_index.keys())
    dataset = ds.dataset(parquet_path, format="parquet")

    # Push the zone membership filter down into the scan so rejected rows are
    # never materialised. This is what keeps a 20M-row month inside memory.
    in_manhattan = pc.field("PULocationID").isin(keep_ids) & \
        pc.field("DOLocationID").isin(keep_ids)
    if not keep_all_trips:
        in_manhattan = in_manhattan & (pc.field("shared_request_flag") == "Y")

    table = dataset.to_table(columns=_COLS, filter=in_manhattan)

    pu = table.column("PULocationID").to_numpy()
    do = table.column("DOLocationID").to_numpy()
    miles = table.column("trip_miles").to_numpy(zero_copy_only=False)
    req = table.column("request_datetime").to_pandas()  # datetime64 handling
    flag = table.column("shared_request_flag").to_numpy(zero_copy_only=False)

    # Map LocationIDs to the mechanism's zone indices. Membership was already
    # enforced in the scan, so every id is present in loc_to_index.
    o_zone = np.array([loc_to_index[int(x)] for x in pu], dtype=np.int64)
    d_zone = np.array([loc_to_index[int(x)] for x in do], dtype=np.int64)

    direct_km = np.asarray(miles, dtype=float) * _MILES_TO_KM
    hour = req.dt.hour.to_numpy().astype(np.int64)
    is_weekend = (req.dt.dayofweek.to_numpy() >= 5)
    shared_request = np.array([str(x).strip().upper() == "Y" for x in flag])

    # A handful of records carry zero or absurd distances; drop the clearly
    # broken ones so they do not poison detour feasibility. Keep it light: only
    # non-positive and the far tail, nothing that would bias the mix.
    good = (direct_km > 0.0) & (direct_km < 60.0) & (o_zone != d_zone)
    pool = {
        "o_zone": o_zone[good],
        "d_zone": d_zone[good],
        "do_zone": d_zone[good].copy(),
        "direct_km": direct_km[good],
        "hour": hour[good],
        "is_weekend": is_weekend[good],
        "shared_request": shared_request[good],
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(
            cache_path,
            **pool,
            __source_signature__=np.array(source_signature),
        )
    return pool


def _stratum_labels(hour: np.ndarray, is_weekend: np.ndarray, dayparts):
    """Assign each trip a stratum id: (daypart index, weekend flag)."""
    dp = np.full(hour.shape, -1, dtype=np.int64)
    for k, (_, lo, hi) in enumerate(dayparts):
        dp[(hour >= lo) & (hour < hi)] = k
    return list(zip(dp.tolist(), is_weekend.astype(int).tolist()))


def stratum_weights(pool: dict, dayparts=DEFAULT_DAYPARTS) -> dict:
    """Real demand mass per (daypart, weekend) stratum, as a probability dict.

    Exposed so the coverage of the sampled windows can be inspected: it says
    what share of real trips falls in each part of the day, which is the share
    of windows that will be drawn from there.
    """
    labels = _stratum_labels(pool["hour"], pool["is_weekend"], dayparts)
    total = len(labels)
    counts: dict = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    return {lab: c / total for lab, c in sorted(counts.items())}


def generate_from_trips(pool: dict,
                        params,
                        network,
                        loc_to_index: dict[int, int],
                        rng: np.random.Generator,
                        dayparts=DEFAULT_DAYPARTS):
    """One real-demand market: (passengers, drivers), matching demand.generate.

    A daypart-weekend stratum is drawn in proportion to real demand mass, then
    `params.n_passengers` trips are sampled from within it. Passenger origins,
    destinations and direct distances come from those real trips; valuations and
    tolerances are drawn from the SAME distributions as the synthetic sampler.
    Drivers are placed at zones drawn from the real drop-off distribution.
    """
    labels = _stratum_labels(pool["hour"], pool["is_weekend"], dayparts)

    # Pick a stratum by real mass, then restrict to that stratum's trips.
    weights = stratum_weights(pool, dayparts)
    strata = list(weights.keys())
    probs = np.array([weights[s] for s in strata])
    probs = probs / probs.sum()  # guard against float drift before rng.choice
    chosen = strata[rng.choice(len(strata), p=probs)]
    in_stratum = np.array([l == chosen for l in labels])
    idx_pool = np.flatnonzero(in_stratum)

    # Fall back to the whole pool if a stratum is too thin for one market.
    if idx_pool.size < params.n_passengers:
        idx_pool = np.arange(len(pool["o_zone"]))

    pick = rng.choice(idx_pool, size=params.n_passengers, replace=False)

    passengers = []
    for i, t in enumerate(pick):
        # --- identical to demand.generate: same clamps, same derivation ---
        v_solo = max(1.0, rng.normal(params.v_solo_mean, params.v_solo_sd))
        tau = float(np.clip(rng.normal(params.tau_mean, params.tau_sd), 0.1, 1.0))
        passengers.append(Passenger(
            id=i,
            origin=int(pool["o_zone"][t]),
            destination=int(pool["d_zone"][t]),
            v_solo=v_solo,
            v_share=v_solo * tau,
            r_max=max(0.1, rng.normal(params.r_max_mean, params.r_max_sd)),
            delta_max=max(0.1, rng.normal(params.delta_max_mean, params.delta_max_sd)),
        ))

    # Drivers start where cars really end up: sample idle locations from the
    # empirical drop-off distribution rather than uniformly over zones.
    do_zones = pool["do_zone"]
    drivers = []
    for j in range(params.n_drivers):
        loc = int(do_zones[rng.integers(do_zones.size)])
        drivers.append(Driver(
            id=j,
            location=loc,
            capacity=params.capacity,
            cost_per_km=params.cost_per_km,
            reservation=max(0.0, rng.normal(params.reservation_mean,
                                            params.reservation_sd)),
        ))

    return passengers, drivers
