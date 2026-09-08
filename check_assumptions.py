"""Sanity check: does this tree match the assumptions behind the results table?

Run this before trusting any number in the comparison. It does not test the
mechanism, it tests that the premises the table rests on still hold here.
"""
import numpy as np
from ora.demand import DemandParams
from ora.baselines import DEFAULT_RADIUS_KM, DEFAULT_SHARE_DISCOUNT

d = DemandParams(n_passengers=1, n_drivers=1)
print("DemandParams defaults in THIS tree")
print(f"  r_max_mean        {d.r_max_mean}      (table assumed 1.5)")
print(f"  tau_mean          {d.tau_mean}     (table assumed 0.75)")
print(f"  v_solo_mean       {d.v_solo_mean}    (table assumed 20.0)")
print(f"  reservation_mean  {d.reservation_mean}     (table assumed 5.0)")
print(f"  cost_per_km       {d.cost_per_km}")
print()
print("Ablation constants")
print(f"  DEFAULT_RADIUS_KM      {DEFAULT_RADIUS_KM}")
print(f"  DEFAULT_SHARE_DISCOUNT {DEFAULT_SHARE_DISCOUNT}")
print()
ok = DEFAULT_RADIUS_KM > d.r_max_mean
print(f"Radius exceeds mean reported r_max: {ok}")
if not ok:
    print("  -> the violation column in the table would be near zero here;")
    print("     the table's story does not hold on this tree.")
