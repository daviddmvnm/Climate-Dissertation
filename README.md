# Climate Game

A four-bloc dynamic threshold-coordination model of climate adoption, calibrated to 2025 strategic benchmarks via SMM and analysed with marginal sweeps and global sensitivity analysis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core code

The game and every downstream analysis run off one module.

- `climate_game.py` — the model: bloc definitions, payoffs, QRE best-response iteration, simulation.
- `core/run_analysis.py` — baseline simulation + marginal sweeps + GSA, writes CSV summaries and figures.
- `calibration/smm_calibration.py` — SMM fit of the four α weights against six benchmark moments.
- `robustness/` — equilibrium uniqueness, φ sweep, spec/ordering perturbations.

## Data

`data/` holds the World Bank inputs that drive calibration.

- `core_.csv` — carbon intensity, emissions, and related indicators by country.
- `gdp.csv`, `gdp_pc.csv` — GDP and GDP per capita.

The four blocs (US, EU, CN, RoW) are aggregated from these series; the aggregates set the cardinal scaling that SMM then weights.

## Running it

```bash
./run_all.sh
```

That clears `results/` and regenerates everything. Outputs land in:

- `results/*.csv` — baseline, sweep crossings, GSA correlations, φ calibration grid.
- `results/figures/*.png` — adoption trajectories, sweep plots, GSA bars.
- `results/dissertation/*.tex` — tables included by the LaTeX source.

Individual scripts can be run standalone (`python core/run_analysis.py`); each writes to `results/` on completion.
