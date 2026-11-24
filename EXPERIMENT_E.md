# Experiment E: budget-balanced alternatives

Added in the revision for Discover Computing. Compares ORA with two
budget-balanced mechanisms on the market-pressure markets (4 drivers of
capacity 3; 8, 12 and 16 passengers; 50 paired seeds; 6 x 6 grid).

| File | Purpose |
|---|---|
| `src/ora/budget_balanced.py` | Threshold rule (Parkes, Kalagnanam and Eso, 2001) and posted-price mechanism |
| `tests/test_budget_balanced.py` | Budget balance, individual rationality and threshold arithmetic tests |
| `run_budget_balanced_comparison.py` | Runs the experiment, including the misreport grid |
| `make_budget_balanced_figure.py` | Fig. 9 of the revised paper |
| `make_prop1_figure.py` | Fig. 2 of the revised paper (schematic for Proposition 1) |

## Reproduce

From the repository root, in the `ora` environment:

```bash
python -m pytest -q                          # full suite, including the new tests
python run_budget_balanced_comparison.py     # about 15 min with 2 workers
python make_budget_balanced_figure.py
python make_prop1_figure.py
```

Set `ORA_WORKERS` to change the number of worker processes and `ORA_N_SEEDS`
for a quick smoke run (for example `ORA_N_SEEDS=2`).

## Outputs (`results/`)

- `budget_balanced_raw.csv`: one row per seed, pressure level and arm
- `budget_balanced_pp_grid.csv`: posted-price outcomes for all 56 price menus
- `budget_balanced_summary.csv`: mean, standard deviation and 95% CI per pressure level, arm and metric
- `budget_balanced_manipulation.csv`: best misreport of every allocated agent, per arm
- `budget_balanced_manip_summary.csv`: share of agents with a profitable misreport and mean gain

The posted-price menu reported in the paper is the one with the highest mean
welfare on the same seeds (solo fare 12, driver payment 8 at every pressure
level), so its welfare is an optimistic upper bound.
