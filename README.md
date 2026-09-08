# ORA — Opt-In Radius Auction

ORA is an exact hypergraph-VCG mechanism for multi-passenger ride-sharing in which passenger-specific pickup flexibility and willingness to share are elicited rather than imposed as global platform constants.

The current implementation supports:
- passenger-specific `r_max` pickup-radius constraints;
- passenger-specific shared-ride valuations `v_share`;
- passenger detour bounds `delta_max`;
- strategic driver reservation values;
- exact winner determination over feasible hyperedges;
- two-sided Clarke pivot payments;
- passenger- and driver-side utility and individual-rationality metrics;
- waiting-time and earnings fairness metrics;
- direct comparison with a global-radius VCG ablation;
- a reimplementation of Someya & Namerikawa (2026), plus an internal consistency validation of that reimplementation;
- a 2 x 2 factorial ablation separating radius elicitation and pooling-value elicitation;
- a mean-matched heterogeneity robustness experiment;
- a demand/supply market-pressure experiment;
- an empirical computational-scalability benchmark; and
- a semi-real validation on NYC TLC High-Volume For-Hire Vehicle (HVFHV) Manhattan trips.

## Setup

From the project root:

```bash
conda env create -f environment.yml
conda activate ora
pip install -e . --no-deps
```

If the environment already exists:

```bash
conda activate ora
pip install -e . --no-deps
```

## Main experimental setting

The main synthetic mechanism experiments use:

- 50 paired random seeds;
- a 6 x 6 Manhattan grid;
- 12 passengers at the baseline pressure level;
- 4 drivers;
- vehicle capacity 3;
- mean passenger pickup radius `r_max = 1.5 km`;
- mean share tolerance `tau = v_share / v_solo = 0.75`;
- observable driver operating cost and private driver reservation value.

The synthetic experiments are complemented by a semi-real validation (below) in which geography and demand come from real NYC TLC HVFHV trips while valuations and flexibility parameters keep the same synthetic distributions.

The scalability benchmark is separate: it uses a 10 x 10 Manhattan grid, 20 seeds per market size, capacity 3, and a fixed passenger/driver ratio of 3:1.

## Current evidence status

The following evidence blocks are complete:

1. **External comparison:** ORA vs global-radius VCG ablation vs Someya & Namerikawa reimplementation.
2. **Experiment A — Factorial ablation:** identifies the separate effects of passenger-specific pickup radius and passenger-specific sharing value.
3. **Experiment B — Mean-matched heterogeneity:** tests whether individualized elicitation matters because passenger types are heterogeneous.
4. **Experiment C — Market pressure:** tests how the factorial effects change as the passenger/driver ratio increases.
5. **Experiment D — Scalability/runtime:** measures where exact ORA becomes computationally expensive.
6. **Semi-real NYC TLC validation:** repeats the main radius comparison on real Manhattan geography and demand.
7. **Someya reimplementation validation:** within-mechanism consistency check of the benchmark arm.

## Reproducing the main radius comparison

Run:

```bash
python run_radius_sweep.py
```

Fresh runs write:

```text
results/radius_sweep_raw.csv
results/radius_sweep_summary.csv
```

The VCG ablation sweeps the platform-wide pickup radius from 0.5 km to 4.0 km in 0.25 km increments. ORA and Someya are radius-independent reference arms.

### Main figures

Run:

```bash
python make_radius_figures.py
```

`make_radius_figures.py` follows this source priority:

1. use `results/radius_sweep_summary.csv` when it exists, so figures automatically reflect a newly generated sweep;
2. otherwise fall back to the archived consolidated workbook `ORA_radius_sweep.xlsx`, sheet `summary`.

This avoids the previous situation in which a fresh CSV sweep could be generated while the figure script still read an older workbook.

Figures are written to:

```text
figures/
```

The main figures report:

- nominal welfare W^nom and match rate;
- pickup-radius violation rate;
- nominal welfare versus pickup-radius violations;
- adjusted passenger utility versus pickup-radius violations.

Only the first two are used in the paper; the other two are supporting variants and are regenerated on demand rather than stored.

## Fixed 2.0 km comparison

At a 2.0 km global-radius comparison point, the current 50-seed results are approximately:

| Arm | Welfare | Match rate | `r_max` violations |
|---|---:|---:|---:|
| ORA | 92.85 | 59.17% | 0.00% |
| VCG ablation | 112.25 | 75.50% | 33.23% |
| Someya | 127.07 | 80.50% | 37.41% |

These results should not be interpreted as ORA maximizing unconstrained nominal welfare. The central trade-off is that mechanisms with weaker or absent passenger-specific pickup constraints admit more assignments, but a substantial fraction of those assignments exceed passengers' reported pickup bounds.

ORA's adjusted passenger utility remains equal to its nominal passenger utility because its selected groups respect the reported pickup and detour bounds.

## Semi-real NYC TLC validation

This is the real-data twin of the main radius sweep. The mechanism arms, radius values, metrics, paired seeds and summary/CI calculation are unchanged; only the market source changes.

Run:

```bash
python run_real_radius_sweep.py
```

Construction, in two stages:

1. `src/ora/osm_network.py` builds the Manhattan taxi-zone mapping and a geometric fallback distance matrix from TLC zone centroids (EPSG:2263, converted to km, 1.3 street-detour factor). Every directed O/D cell supported by enough observed trips is then replaced by the median observed TLC trip distance; thin cells keep the geometric fallback. The result is a `ZoneNetwork` with empirical distances, supplied identically to ORA, the VCG ablation, Someya and the metrics layer.
2. `src/ora/real_demand.py` samples passenger origins and destinations from the real HVFHV trip pool (extracted once from the raw month, which is too large to load whole). Valuations and flexibility parameters (`v_solo`, `v_share`, `r_max`, `delta_max`) keep the same `DemandParams` distributions as the synthetic experiment, so any change in results is attributable to geography and demand, not to a retuned valuation model.

Fresh outputs:

```text
results/real_radius_sweep_raw.csv
results/real_radius_sweep_summary.csv
results/real_distance_matrix_diagnostics.csv
results/real_demand_strata.csv
```

Real-data figures:

```bash
python make_real_radius_figures.py
```

## Experiment A — Factorial ablation

Run:

```bash
python run_factorial_ablation.py
```

The four arms are:

| Arm | Pickup radius | Shared-ride value |
|---|---|---|
| ORA | Individual | Individual |
| No radius elicitation | Global 2.0 km | Individual |
| No pooling elicitation | Individual | `0.75 * v_solo` |
| Full ablation | Global 2.0 km | `0.75 * v_solo` |

Fresh outputs:

```text
results/factorial_ablation_raw.csv
results/factorial_ablation_summary.csv
results/factorial_ablation_effects.csv
```

Archived consolidated results are stored in:

```text
ORA_factorial_ablation.xlsx
```

Current headline results show that passenger-specific pickup radius is the dominant source of the acceptability effect, while passenger-specific sharing value has a smaller but distinct effect on welfare and passenger individual rationality.

## Experiment B — Mean-matched heterogeneity robustness

Run:

```bash
python run_heterogeneity_sweep.py
```

This is the **official heterogeneity experiment used for the paper**.

### Pickup-radius heterogeneity

The population mean remains `r_max_mean = 1.5 km`, while the standard deviation varies:

```text
0.00, 0.25, 0.50, 0.75 km
```

For every seed and heterogeneity level, ORA is compared with a global-radius mechanism whose radius is set to that market's **realized mean `r_max`**. The comparator is therefore mean-matched within each generated market rather than fixed at 1.5 km.

At zero heterogeneity, the two mechanisms should coincide exactly. Any divergence as dispersion grows is therefore attributable to heterogeneity rather than a difference in the market mean.

### Pooling-preference heterogeneity

The population mean remains `tau_mean = 0.75`, while the standard deviation varies:

```text
0.000, 0.075, 0.150, 0.225
```

For every seed and heterogeneity level, ORA is compared with a common share discount equal to that market's **realized mean `tau`**. This controls for small changes in the realized mean produced by clipping `tau` to its allowed range.

Fresh official outputs:

```text
results/heterogeneity_mean_matched_raw.csv
results/heterogeneity_mean_matched_summary.csv
results/heterogeneity_mean_matched_effects.csv
results/heterogeneity_mean_matched_trends.csv
```

Archived consolidated results are stored in:

```text
ORA_heterogeneity.xlsx
```

In that workbook, sheets beginning with `mm_` are the **official paper results**. The older `raw`, `summary`, `effects`, and `trends` sheets are retained only for methodological traceability and should not be cited as the final heterogeneity evidence.

## Experiment C — Market pressure

Run:

```bash
python run_market_pressure_sweep.py
```

The experiment holds four drivers and capacity three fixed, and varies passenger counts:

```text
8, 12, 16 passengers
```

This gives passenger/driver ratios:

```text
2:1, 3:1, 4:1
```

Markets are nested within seed. The 12-passenger case reproduces the Experiment A baseline; the 8-passenger case removes passengers; the 16-passenger case adds passengers while keeping the original driver realization fixed.

Fresh outputs:

```text
results/market_pressure_raw.csv
results/market_pressure_summary.csv
results/market_pressure_effects.csv
results/market_pressure_trends.csv
```

Archived consolidated results are stored in:

```text
ORA_market_pressure.xlsx
```

## Experiment D — Computational scalability

Run the completed paper-size benchmark with:

```bash
python run_scalability_benchmark.py --sizes 12 24 36 48 60 72 84 96 120 --n-seeds 20
```

Fresh outputs:

```text
results/scalability_raw.csv
results/scalability_summary.csv
results/scalability_scaling_fits.csv
```

Archived consolidated results are stored in:

```text
ORA_scalability.xlsx
```

Runtime distributions are reported with medians/IQRs, while mean/95% CI values are retained for consistency. Log-log scaling fits are descriptive empirical fits, not theoretical complexity bounds.

The benchmark resumes by default: pairs of (market size, seed) already present in `results/scalability_raw.csv` are skipped, so the seed count can be raised without recomputing earlier runs. Pass `--fresh` to discard existing rows and start over.

Rebuild the archived workbook from the fresh CSVs after any rerun, so that the archived copy never lags behind the results the paper reports:

```bash
python make_scalability_workbook.py
```

Generate the runtime-decomposition figure with:

```bash
python make_scalability_figure.py
```

The figure plots medians with interquartile ranges rather than means with confidence intervals, because occasional slow solver starts skew the mean at the smallest market sizes. It prints the seed count it found and the caption wording that matches it.

At 120 passengers, the archived benchmark shows that full clearing accounts for about 97.8% of mean end-to-end runtime. The payment-overhead component itself accounts for about 97.3%, confirming that repeated Clarke agent-exclusion solves — not one main WDP solve — are the dominant computational bottleneck.

## Someya comparison and validation

`src/ora/someya.py` reimplements the closest published hypergraph-VCG comparison used in the project.

The Someya arm:

- uses seat capacity as its hard feasibility condition;
- does not impose passenger-specific pickup-radius or detour bounds;
- includes wait and detour as soft penalties;
- treats drivers as fixed-rate executors rather than strategic agents;
- uses passenger-side Clarke payments;
- is evaluated with the same measurement code and against the same true passenger types as the other arms.

Driver-side IR, driver utility, earnings fairness and platform deficit are reported as N/A for Someya because those quantities are not defined under its driver model.

The reimplementation itself is validated with:

```bash
python validate_someya_benchmark.py
```

This is a within-mechanism consistency check, not a numeric reproduction of the source paper's dataset. It compares the preference-aware Someya mechanism against a Standard-VCG variant that shares the same groups, routing, set-packing solver and Clarke-pricing code but zeroes the detour/wait disutility weights, and checks that the qualitative directions reported in the source paper (higher passenger utility for the preference-aware variant, more aggressive matching for the standard variant) are reproduced on the ORA markets.

Fresh outputs:

```text
results/someya_validation_raw.csv
results/someya_validation_summary.csv
results/someya_validation_diagnostics.csv
```

## Measurement conventions

All mechanism arms are evaluated through the same metrics layer.

Important reported quantities include:

- realised welfare;
- match rate;
- pooling rate;
- mean and P95 waiting time;
- mean detour;
- idle-driver ratio;
- waiting-time Gini and Jain indices;
- driver-earnings Gini and Jain indices;
- driver IR violation rate and driver utility;
- passenger IR violation rate;
- nominal passenger utility;
- adjusted passenger utility;
- total acceptability cost;
- platform deficit;
- pickup-radius violation rate.

The adjusted passenger-utility convention penalizes only the portion of a selected route that exceeds the passenger's reported pickup or detour bounds. The same rule is applied to every arm.

## Statistical convention

Experiments use paired seeds whenever mechanisms are compared.

For paired effects, the same generated market is used in both arms and confidence intervals are computed from seed-level differences:

```text
mean difference +/- 1.96 * sd(difference) / sqrt(n)
```

The main mechanism experiments use 50 seeds. The scalability benchmark uses 20 seeds per market size.

For the headline ORA-vs-ablation comparisons, paired-seed differences with normal and bootstrap 95% confidence intervals, paired t-test p-values and Cohen's d_z are consolidated in:

```text
results/headline_paired_statistics.csv
```

## Archived consolidated evidence

The current project folder stores the checked result families as individual workbooks:

```text
ORA_radius_sweep.xlsx        Main ORA / VCG / Someya radius comparison
ORA_factorial_ablation.xlsx  Experiment A
ORA_heterogeneity.xlsx       Experiment B; mm_* sheets are official
ORA_market_pressure.xlsx     Experiment C
ORA_scalability.xlsx         Experiment D
```

These workbooks are archived snapshots of the checked experiments. The experiment scripts still write CSV files into `results/` when intentionally rerun.

## Project layout

```text
configs/                     Experiment configuration files
data/                        Optional input/cache directory (HVFHV trip pool)
figures/                     Generated figures
results/                     Fresh CSV outputs from intentionally rerun experiments
src/ora/                     Mechanism implementation
tests/                       Unit and invariant tests

ORA_radius_sweep.xlsx        Archived checked radius results
ORA_factorial_ablation.xlsx  Archived checked Experiment A results
ORA_heterogeneity.xlsx       Archived checked Experiment B results
ORA_market_pressure.xlsx     Archived checked Experiment C results
ORA_scalability.xlsx         Archived checked Experiment D results

run_radius_sweep.py          ORA / VCG ablation / Someya radius sweep (synthetic)
run_real_radius_sweep.py     Semi-real NYC TLC HVFHV radius sweep
make_radius_figures.py       Synthetic-sweep figure generation
make_real_radius_figures.py  Real-data figure generation
make_scalability_figure.py   Runtime-decomposition figure (Experiment D)
make_scalability_workbook.py Rebuild ORA_scalability.xlsx from fresh CSVs
run_factorial_ablation.py    2 x 2 component ablation
run_heterogeneity_sweep.py   Mean-matched heterogeneity experiment
run_market_pressure_sweep.py Demand/supply-pressure experiment
run_scalability_benchmark.py Runtime/scalability benchmark
validate_someya_benchmark.py Someya reimplementation consistency check
run_arms.py                  Small multi-arm smoke test
check_assumptions.py         Sanity check for key model parameters
```

Superseded material has been removed from the folder: the pre-IEEE paper draft (fully subsumed by the IEEE version), the one-shot `add_someya_to_sweep.py` migration utility (its job — appending the Someya arm to the older two-arm sweep CSVs — is complete and its output lives in the checked CSVs and workbook), the early five-seed preliminary result tables, and scratch timing scripts. Every remaining file is live code, a checked result, or a current deliverable.

Core implementation modules include:

```text
src/ora/types.py
src/ora/network.py
src/ora/osm_network.py
src/ora/demand.py
src/ora/real_demand.py
src/ora/feasibility.py
src/ora/routing.py
src/ora/grouping.py
src/ora/wdp.py
src/ora/payments.py
src/ora/baselines.py
src/ora/someya.py
src/ora/acceptability.py
src/ora/metrics.py
src/ora/fairness.py
```

## Tests

Run the test suite from the project root:

```bash
pytest
```

The project includes invariant checks for the feasibility layer and mechanism components. In particular, the selected ORA allocation should never contain a passenger whose realized pickup distance exceeds the passenger's reported `r_max`.

## Current interpretation

The present evidence supports the following structured interpretation:

1. **A global pickup radius produces a welfare / acceptability trade-off.** Expanding the global feasible set increases nominal welfare and match rate, but also increases the share of passengers served outside their true pickup bounds.

2. **Pickup-radius elicitation is the dominant source of ORA's acceptability protection.** The factorial ablation shows that individualizing `r_max` accounts for most of the reduction in pickup-radius violations and acceptability cost.

3. **Pooling-value elicitation contributes separately.** Individual `v_share` has a smaller effect than `r_max`, but changes welfare and passenger individual-rationality outcomes even when pickup constraints are held fixed.

4. **These effects are driven by heterogeneity.** In the mean-matched Experiment B, ORA and the corresponding global comparator coincide in the homogeneous control and separate as individual types become more dispersed.

5. **Demand pressure changes the magnitude of the effects.** Experiment C shows that several radius and sharing-value effects become more consequential as passengers per driver increase rather than moving uniformly in one direction.

6. **Exact allocation is not the main runtime bottleneck at the largest tested markets.** Experiment D identifies repeated Clarke payment counterfactuals as the dominant computational cost.

7. **The main pattern survives real geography and demand.** In the semi-real NYC validation, ORA keeps pickup-radius violations at zero while the global-radius ablation's violations rise with the radius, reproducing the synthetic trade-off on empirical trips.

## Current limitations

The current study remains subject to several important limitations:

- passenger valuations and flexibility parameters remain synthetic even in the semi-real validation; only geography and demand are empirical;
- the semi-real network uses zone-level empirical trip distances, not a full road network with routing;
- a static, single-window base mechanism;
- exact enumeration and repeated payment counterfactuals that limit scalability;
- hard-constraint interpretation of passenger pickup and detour bounds;
- no dynamic driver repositioning;
- no endogenous congestion;
- historical fairness has been designed and partially implemented, but is not yet the main experimental mechanism;
- the Someya comparison is a reimplementation on the ORA experimental population, not a reproduction on the original Someya dataset or experimental instance; the validation script checks qualitative consistency only.

## Reproducibility and folder hygiene

Do not overwrite checked workbooks unless you intentionally regenerate an experiment.

If you rerun an experiment:

1. keep the generated CSV files in `results/` until the new run has been checked;
2. regenerate figures from those CSVs;
3. only then replace the corresponding archived workbook if you intend the new run to become the official result set.

Superseded documents and one-shot utilities are deleted once their results are captured in the checked CSVs, workbooks, or the current paper version, so the folder contains only live code, checked evidence, and current deliverables.

Before sharing or submitting the project folder, remove generated local artifacts that are not part of the scientific evidence, including:

```text
.pytest_cache/
**/__pycache__/
*.pyc
*.egg-info/
cache/
```

The supplied `.gitignore` excludes these artifacts so they should not be committed to a repository.