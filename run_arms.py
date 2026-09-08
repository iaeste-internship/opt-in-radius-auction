"""Run ORA and the ablation over several seeds and report mean metrics."""
import numpy as np
from ora.network import grid_network
from ora.demand import DemandParams, generate
from ora.grouping import enumerate_groups
from ora.payments import clear_market
from ora.metrics import compute_metrics
from ora.baselines import run_ablation, radius_violations
from ora.someya import run_someya

SEEDS = [0, 1, 2, 3, 4]
PARAMS = DemandParams(n_passengers=12, n_drivers=4, capacity=3)

def run():
    rows = {"ORA": [], "VCG ablation": [], "Someya": []}
    viol = {"ORA": [], "VCG ablation": [], "Someya": []}
    for seed in SEEDS:
        net = grid_network(6, 6)
        rng = np.random.default_rng(seed)
        pax, drv = generate(PARAMS, net, rng)

        # ORA: elicited types, used as reported
        groups, _ = enumerate_groups(drv, pax, net)
        ora = clear_market(groups, {p.id: p for p in pax}, {d.id: d for d in drv})
        rows["ORA"].append(compute_metrics(ora.allocation, pax, drv, net))
        viol["ORA"].append(radius_violations(ora.allocation, pax)[1])

        # Ablation: allocate with assumed types, measure with true ones
        abl = run_ablation(pax, drv, net)
        rows["VCG ablation"].append(compute_metrics(abl.allocation, pax, drv, net))
        viol["VCG ablation"].append(radius_violations(abl.allocation, pax)[1])

        # Someya & Namerikawa (2026): capacity-only feasibility, soft detour and
        # wait penalties, drivers as fixed-rate executors. Allocated on their
        # weight, measured with our true types and the same metrics code.
        som = run_someya(pax, drv, net)
        rows["Someya"].append(compute_metrics(som.allocation, pax, drv, net))
        viol["Someya"].append(radius_violations(som.allocation, pax)[1])
    return rows, viol

if __name__ == "__main__":
    rows, viol = run()

    fields = [
        "realised_welfare",
        "match_rate",
        "pooling_rate",
        "mean_wait_min",
        "p95_wait_min",
        "mean_wait_all_min",
        "mean_detour_km",
        "idle_ratio",
        "gini_wait_all",
        "jain_wait_all",
        "gini_earnings",
        "jain_earnings",
        "driver_ir_violation_rate",
        "mean_allocated_driver_utility",
        "min_allocated_driver_utility",
        "deficit_pct_of_welfare",
    ]

    # Driver-side axes are out of scope for Someya, which models drivers as
    # fixed-rate executors with no reservation and no payment. Reporting a
    # number for these against our two-sided notion would compare Someya to a
    # criterion its model never claimed, so they are shown as N/A for that arm.
    driver_side = {
        "driver_ir_violation_rate",
        "mean_allocated_driver_utility",
        "min_allocated_driver_utility",
        "deficit_pct_of_welfare",
        # Earnings fairness is measured on driver earnings, which are all zero
        # under Someya (it pays no drivers). Zero earnings trivially score
        # Gini 0 / Jain 1, which reads as perfect equality but is an artifact
        # of paying nobody, not a property of the mechanism. Same reasoning as
        # the driver-IR rows above.
        "gini_earnings",
        "jain_earnings",
    }

    def cell(field, arm):
        if arm == "Someya" and field in driver_side:
            return f"{'N/A':>12s}"
        value = np.mean([getattr(m, field) for m in rows[arm]])
        return f"{value:12.4f}"

    print(f"{'metric':34s} {'ORA':>12s} {'ablation':>12s} {'Someya':>12s}")

    for field in fields:
        print(
            f"{field:34s} "
            f"{cell(field, 'ORA')} "
            f"{cell(field, 'VCG ablation')} "
            f"{cell(field, 'Someya')}"
        )

    print(
        f"{'r_max_violation_rate':34s} "
        f"{np.mean(viol['ORA']):12.4f} "
        f"{np.mean(viol['VCG ablation']):12.4f} "
        f"{np.mean(viol['Someya']):12.4f}"
    )