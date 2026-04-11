"""
lambda_decomposition.py
────────────────────────
Decompose the λ asymmetry finding into:
  (1) Own-bloc channel:    how σ_US changes when λ_US varies
  (2) Cross-bloc spillover: how σ_EU, σ_CN change when λ_US varies

And the same for λ_CN, showing the contrast.

Method A — Equilibrium sweep (clean, no MC noise):
  Sweep λ_US from 0.5 → 5.0 at baseline structural params.
  For each point call solve_model() and extract sigma[1][G0][i]
  (exact period-1 adoption probability from the all-inactive state).
  Repeat for λ_CN sweep.

Method B — Mediation regression (using existing GSA data):
  Use gsa_baseline_samples.csv which has 1000 draws with lam_US,
  lam_CN, fm_US, fm_CN, mean_coord_time all jointly recorded.
  Run Baron-Kenny mediation: total effect of lam on timing,
  direct effect after controlling for fm (own-bloc), indirect
  (mediated) = total − direct.

Output:
  results/lambda_decomposition.csv
  results/figures/fig_lambda_decomposition.png
"""

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy import stats

warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS, solve_model
from run_analysis import (
    build_bloc_data, build_params, SMM_BASELINE,
    DISCOUNT, ETA, KAPPA, GAMMA, THETA,
)

# ── Config ─────────────────────────────────────────────────────────────────────
N_SWEEP   = 35        # points per lambda sweep
LAM_LO    = 0.5
LAM_HI    = 5.0
SEED      = 42

G0 = (0, 0, 0, 0)    # all-inactive initial state

OUT_CSV  = os.path.join(ROOT_DIR, "results", "lambda_decomposition.csv")
FIG_DIR  = os.path.join(ROOT_DIR, "results", "figures")
FIG_PATH = os.path.join(FIG_DIR, "fig_lambda_decomposition.png")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data and build baseline params ───────────────────────────────────────
print("Loading data...")
bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
raw       = bloc_data.set_index("bloc")

weights = {}
for bloc in PLAYERS:
    weights[bloc] = 0.3 * raw.loc[bloc, "emission_share"] + 0.7 * raw.loc[bloc, "gdp_share"]
total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}

b = SMM_BASELINE
params_base = build_params(
    raw, weights,
    ac=b["ac"], ad=b["ad"], ap=b["ap"], ab=b["ab"],
    lam=b["lam"],
)
lam_base = b["lam"]

print(f"Baseline λ = {lam_base:.4f}")
print(f"Baseline costs: { {k: round(v,3) for k,v in params_base.costs.items()} }")

# ── Method A: Equilibrium sweep ────────────────────────────────────────────────

def sweep_lambda(vary_bloc, n_pts, lam_lo, lam_hi):
    """
    Sweep λ for `vary_bloc` over [lam_lo, lam_hi].
    All other blocs' λ held at lam_base.
    Returns DataFrame with columns: lam, sigma_US, sigma_EU, sigma_CN, sigma_RoW.
    """
    lam_vals = np.linspace(lam_lo, lam_hi, n_pts)
    rows = []
    for lam_v in lam_vals:
        lam_map = {b: lam_base for b in PLAYERS}
        lam_map[vary_bloc] = float(lam_v)
        p = GameParams(**{**params_base.__dict__, "lam": lam_map})
        try:
            _, sigma, _, _ = solve_model(p)
            row = {"vary_bloc": vary_bloc, "lam": float(lam_v)}
            for i, bloc in enumerate(PLAYERS):
                row[f"sigma_{bloc}"] = float(sigma[1][G0][i])
        except Exception as e:
            row = {"vary_bloc": vary_bloc, "lam": float(lam_v)}
            for bloc in PLAYERS:
                row[f"sigma_{bloc}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

print(f"\nSweeping λ_US ({N_SWEEP} points)...")
df_us = sweep_lambda("US", N_SWEEP, LAM_LO, LAM_HI)

print(f"Sweeping λ_CN ({N_SWEEP} points)...")
df_cn = sweep_lambda("CN", N_SWEEP, LAM_LO, LAM_HI)

df_sweep = pd.concat([df_us, df_cn], ignore_index=True)
df_sweep.to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}")

# ── Method B: Mediation using existing GSA data ────────────────────────────────
gsa_path = os.path.join(ROOT_DIR, "results", "gsa_baseline_samples.csv")

def mediation_indirect(df, treatment, mediator, outcome):
    """
    Baron-Kenny mediation.
    Returns: total, direct, indirect effect (OLS coefficients).
    All variables mean-centred before regression.
    """
    sub = df[[treatment, mediator, outcome]].dropna()
    X = sub[treatment].values
    M = sub[mediator].values
    Y = sub[outcome].values

    # Standardise
    def norm(x): return (x - x.mean()) / (x.std() + 1e-12)
    Xn, Mn, Yn = norm(X), norm(M), norm(Y)

    # Total effect: Y ~ X
    total, _, _, _, _ = stats.linregress(Xn, Yn)

    # Path a: M ~ X
    a, _, _, _, _ = stats.linregress(Xn, Mn)

    # Path b: Y ~ X + M  (direct + partial)
    # Use OLS via numpy for two predictors
    A = np.column_stack([np.ones(len(Xn)), Xn, Mn])
    coef, _, _, _ = np.linalg.lstsq(A, Yn, rcond=None)
    direct  = coef[1]   # coefficient on X after controlling for M
    b_path  = coef[2]   # coefficient on M after controlling for X
    indirect = a * b_path

    return total, direct, indirect

if os.path.exists(gsa_path):
    print("\n── Mediation analysis (GSA data) ──────────────────────────────")
    gsa = pd.read_csv(gsa_path)

    for treat, med, bloc_label in [
        ("lam_US", "fm_US", "λ_US → fm_US → timing"),
        ("lam_CN", "fm_CN", "λ_CN → fm_CN → timing"),
    ]:
        tot, direct, indirect = mediation_indirect(gsa, treat, med, "mean_coord_time")
        pct = 100 * indirect / tot if abs(tot) > 1e-8 else np.nan
        print(f"  {bloc_label}")
        print(f"    Total effect (std):    {tot:+.4f}")
        print(f"    Direct effect (std):   {direct:+.4f}")
        print(f"    Indirect via {med}: {indirect:+.4f}  ({pct:.1f}% of total)")

    # Cross-bloc spillover: does λ_US affect fm_CN?
    print("\n  Cross-bloc spillover (Spearman ρ):")
    for treat in ["lam_US", "lam_CN"]:
        for outcome in ["fm_US", "fm_EU", "fm_CN"]:
            rho, pval = spearmanr(gsa[treat], gsa[outcome])
            sig = "*" if pval < 0.05 else " "
            print(f"    ρ({treat}, {outcome}) = {rho:+.3f}{sig}  (p={pval:.4f})")
else:
    print(f"\nWarning: {gsa_path} not found — skipping mediation analysis.")

# ── Print sweep summary ────────────────────────────────────────────────────────
print("\n── Equilibrium sweep summary ──────────────────────────────────")
for vary_bloc, df in [("US", df_us), ("CN", df_cn)]:
    lo_row = df.iloc[0]
    hi_row = df.iloc[-1]
    print(f"\n  λ_{vary_bloc} sweep: {lo_row['lam']:.1f} → {hi_row['lam']:.1f}")
    for bloc in PLAYERS:
        delta = hi_row[f"sigma_{bloc}"] - lo_row[f"sigma_{bloc}"]
        print(f"    Δσ_{bloc:3s} = {delta:+.4f}  "
              f"({lo_row[f'sigma_{bloc}']:.4f} → {hi_row[f'sigma_{bloc}']:.4f})")

# ── Figure ─────────────────────────────────────────────────────────────────────
COLOURS = {"US": "#E63946", "EU": "#457B9D", "CN": "#E9C46A", "RoW": "#2A9D8F"}
LINESTYLES = {"US": "-", "EU": "--", "CN": "-.", "RoW": ":"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

panels = [
    (df_us, "US", axes[0], r"$\lambda_{US}$", "Varying $\\lambda_{US}$ (all others fixed)"),
    (df_cn, "CN", axes[1], r"$\lambda_{CN}$", "Varying $\\lambda_{CN}$ (all others fixed)"),
]

for df, vary_bloc, ax, xlabel, title in panels:
    for bloc in PLAYERS:
        ax.plot(df["lam"], df[f"sigma_{bloc}"],
                color=COLOURS[bloc], ls=LINESTYLES[bloc],
                lw=2.2, label=f"$\\sigma_{{{bloc}}}$")
    ax.axvline(lam_base, color="grey", lw=1.0, ls=":", alpha=0.7, label=f"Baseline λ={lam_base:.2f}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Period-1 adoption probability $\\sigma_i(t=1, G_0)$", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(LAM_LO, LAM_HI)
    ax.set_ylim(bottom=0)

fig.suptitle(
    r"$\lambda$ Decomposition: own-bloc vs cross-bloc response",
    fontsize=13, y=1.02
)
plt.tight_layout()
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved → {FIG_PATH}")
print("Done.")
