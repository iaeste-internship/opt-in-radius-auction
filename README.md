# Opt-In Radius Auction (ORA)

Code, fixed seeds, results and figures for the paper

> M. Verdyguer Cuevas, S. K. Tiwari, and S. P. Dubey, "Eliciting Pickup Radius
> and Sharing Preferences in an Exact VCG Mechanism for Pooled Ridesharing,"
> Discover Computing (under review).

ORA applies the Vickrey–Clarke–Groves mechanism to exact two-sided hypergraph
allocation for multi-passenger ride-sharing. Passengers report solo and shared
valuations, a maximum pickup distance and a maximum detour; drivers report a
reservation value. Winner determination is an exact set-packing problem solved
with CBC through PuLP, and payments are bilateral Clarke transfers.

## Installation

```bash
conda env create -f environment.yml
conda activate ora
pip install -e .
python -m pytest -q          # full test suite
```

## Reproducing the paper

Run the scripts from the repository root. Each `run_*.py` script writes CSV
files to `results/`; the `make_*.py` scripts draw the figures in `figures/`.

| Paper | Script(s) |
|---|---|
| Synthetic radius sweep (Section 5.1) | `run_radius_sweep.py`, `make_radius_figures.py` |
| Someya benchmark validation (Section 5.8) | `validate_someya_benchmark.py` |
| Experiment A, factorial ablation | `run_factorial_ablation.py` |
| Experiment B, heterogeneity | `run_heterogeneity_sweep.py` |
| Experiment C, market pressure and budget | `run_market_pressure_sweep.py`, `budgetfig.py` |
| Experiment D, scalability | `run_scalability_benchmark.py`, `make_scalability_figure.py`, `make_scalability_workbook.py` |
| Experiment E, budget-balanced alternatives | `run_budget_balanced_comparison.py`, `make_budget_balanced_figure.py` (see [`EXPERIMENT_E.md`](EXPERIMENT_E.md)) |
| Fig. 2, Proposition 1 schematic | `make_prop1_figure.py` |
| Semi-synthetic NYC TLC experiment | `run_real_radius_sweep.py`, `make_real_radius_figures.py` |

The NYC TLC experiment needs the public High Volume For-Hire Vehicle trip
records, which are not redistributed here. Download the monthly Parquet files
from the NYC Taxi and Limousine Commission and place them in `data/hvfhv/`;
the taxi-zone shapefile is already in `data/taxi_zones/`. Extracted trip pools
are cached in `~/.ora_cache/`.

## License

This project is released under the MIT License. See `LICENSE` for the full text.
