import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_excel('/mnt/project/ORA_market_pressure.xlsx', 'raw')
df['deficit'] = df.deficit_pct_of_welfare / 100 * df.realised_welfare * -1
df['served'] = df.match_rate * df.n_passengers
df['per_served'] = df.deficit / df.served

ARMS = [('ORA', 'ORA', 'tab:blue', 'o', '-'),
        ('Full ablation', 'VCG ablation', 'tab:orange', 's', '--')]
ratios = [2, 3, 4]

def stat(sub, col):
    v = sub[col]
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 5.2), sharex=True)
for arm, lab, c, mk, ls in ARMS:
    d = df[df.arm == arm]
    for ax, col in ((a1, 'deficit'), (a2, 'per_served')):
        m = [stat(d[d.passenger_driver_ratio == r], col) for r in ratios]
        ax.errorbar(range(3), [x[0] for x in m], yerr=[x[1] for x in m],
                    color=c, marker=mk, linestyle=ls, capsize=3, markersize=6,
                    linewidth=1.6, label=lab if ax is a1 else None)
        print(arm, col, [round(x[0], 2) for x in m], [round(x[1], 2) for x in m])
a1.set_ylabel('Mean deficit'); a1.legend(loc='upper left'); a1.grid(alpha=.3)
a2.set_ylabel('Deficit / served pax'); a2.grid(alpha=.3)
a2.set_xticks(range(3)); a2.set_xticklabels(['2:1', '3:1', '4:1'])
a2.set_xlabel('Passenger-driver ratio')
a1.set_title('Platform budget under market pressure')
fig.tight_layout()
fig.savefig('platform_budget_market_pressure.png', dpi=400)
