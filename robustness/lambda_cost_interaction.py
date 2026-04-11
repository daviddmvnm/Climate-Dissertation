"""
lambda_cost_interaction.py
───────────────────────────
Tests whether the λ asymmetry is conditional on the cost asymmetry.

Hypothesis: λ_US accelerates coordination because US has low transition costs;
a more rational US calculates that adoption is worth it. λ_CN has weaker effects
because CN's high costs mean that even a very rational CN calculates that waiting
is optimal. If China's costs were reduced, λ_CN should start to look more like λ_US.

Method: Two separate random-draw 2D sweeps.

  Sweep 1 — US interaction:
    N draws of (λ_US, cost_US_multiplier) from [0.5, 4.0] × [0.4, 1.6].
    All other params at baseline.
    For each draw: run_mc(params, 150, seed=i) → success_rate, mean_coord_time.

  Sweep 2 — CN interaction:
    N draws of (λ_CN, cost_CN_multiplier) from same ranges.

Analysis: Bin cost_mult into quartiles. Within each quartile compute
Spearman ρ(λ, success_rate). If the mechanism is cost-rationality
complementarity, ρ(λ, success) should be strongest in Q1 (low cost)
and weakest/absent in Q4 (high cost).

Output:
  results/lambda_cost_interaction.csv
  results/figures/fig_lambda_cost_interaction.png
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

warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS
from run_analysis import (
    build_bloc_data, build_params, SMM_BASELINE,
    DISCOUNT, ETA, KAPPA, GAMMA, THETA, run_mc,
)

# ── Config ─────────────────────────────────────────────────────────────────────
N_DRAWS   = 600    # draws per sweep
N_MC      = 150    # MC paths per draw
LAM_LO    = 0.5
LAM_HI    = 4.0
COST_LO   = 0.4
COST_HI   = 1.6
SEED_US   = 42
SEED_CN   = 99

OUT_CSV  = os.path.join(ROOT_DIR, "results", "lambda_cost_interaction.csv")
FIG_DIR  = os.path.join(ROOT_DIR, "results", "figures")
FIG_PATH = os.path.join(FIG_DIR, "fig_lambda_cost_interaction.png")
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
print(f"Baseline costs: US={params_base.costs['US']:.3f}  CN={params_base.costs['CN']:.3f}")

# ── Core sweep function ────────────────────────────────────────────────────────

def run_interaction_sweep(vary_bloc, n_draws, n_mc, seed):
    """
    Draw (λ_vary_bloc, cost_mult) pairs uniformly.
    Hold all other λ at lam_base; hold all other cost multipliers at 1.0.
    Returns DataFrame with: lam_value, cost_mult, success_rate, mean_coord_time,
    fm_US, fm_CN.
    """
    rng  = np.random.default_rng(seed)
    rows = []

    print(f"\n  Running {vary_bloc} interaction ({n_draws} draws × {n_mc} MC)...")
    for i in range(n_draws):
        lam_v    = float(rng.uniform(LAM_LO, LAM_HI))
        cost_mult = float(rng.uniform(COST_LO, COST_HI))

        lam_map = {bloc: lam_base for bloc in PLAYERS}
        lam_map[vary_bloc] = lam_v

        new_costs = dict(params_base.costs)
        new_costs[vary_bloc] = params_base.costs[vary_bloc] * cost_mult

        p = GameParams(**{
            **params_base.__dict__,
            "lam":   lam_map,
            "costs": new_costs,
        })

        try:
            _, _, _, _, success, _, mean_ct, fm = run_mc(p, n_mc, seed=int(i * 7 + seed))
        except Exception:
            success = np.nan
            mean_ct = np.nan
            fm      = {bloc: np.nan for bloc in PLAYERS}

        rows.append({
            "vary_bloc":     vary_bloc,
            "lam_value":     lam_v,
            "cost_mult":     cost_mult,
            "success_rate":  success,
            "mean_coord_time": mean_ct,
            "fm_US":         fm["US"],
            "fm_CN":         fm["CN"],
        })

        if (i + 1) % 200 == 0:
            print(f"    {vary_bloc}: {i+1}/{n_draws}")

    return pd.DataFrame(rows)


df_us = run_interaction_sweep("US", N_DRAWS, N_MC, SEED_US)
df_cn = run_interaction_sweep("CN", N_DRAWS, N_MC, SEED_CN)
df_all = pd.concat([df_us, df_cn], ignore_index=True)
df_all.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")

# ── Analysis: ρ(λ, success) by cost quartile ──────────────────────────────────

print("\n── ρ(λ, success_rate) by cost quartile ────────────────────────────")
print(f"{'Bloc':5}  {'Quartile':12}  {'Cost range':18}  {'ρ':>7}  {'p':>7}  {'N':>5}")

quartile_rows = []
for vary_bloc, df in [("US", df_us), ("CN", df_cn)]:
    valid = df.dropna(subset=["lam_value", "success_rate", "cost_mult"])
    quartiles = pd.qcut(valid["cost_mult"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    for q_label in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
        mask = (quartiles == q_label)
        sub  = valid[mask]
        if len(sub) < 20:
            continue
        rho, pval = spearmanr(sub["lam_value"], sub["success_rate"])
        cost_lo = sub["cost_mult"].min()
        cost_hi = sub["cost_mult"].max()
        sig = "*" if pval < 0.05 else " "
        print(f"  {vary_bloc:3}  {q_label:12}  [{cost_lo:.2f} – {cost_hi:.2f}]      "
              f"{rho:+.3f}{sig}  {pval:.4f}  {len(sub):5d}")
        quartile_rows.append({
            "vary_bloc":  vary_bloc,
            "quartile":   q_label,
            "cost_lo":    round(cost_lo, 3),
            "cost_hi":    round(cost_hi, 3),
            "rho":        round(rho, 4),
            "pval":       round(pval, 4),
            "n":          len(sub),
        })

print("  * p < 0.05")

# ── Also report overall ρ ──────────────────────────────────────────────────────
print("\n── Overall ρ(λ, success_rate) and ρ(λ, mean_coord_time) ──────────")
for vary_bloc, df in [("US", df_us), ("CN", df_cn)]:
    valid = df.dropna(subset=["lam_value", "success_rate", "mean_coord_time"])
    r1, p1 = spearmanr(valid["lam_value"], valid["success_rate"])
    r2, p2 = spearmanr(valid["lam_value"], valid["mean_coord_time"])
    print(f"  λ_{vary_bloc}: ρ(success)={r1:+.3f} (p={p1:.4f})  "
          f"ρ(timing)={r2:+.3f} (p={p2:.4f})")

# ── Figure ─────────────────────────────────────────────────────────────────────
QUARTILE_COLOURS = {
    "Q1 (low)":  "#2ECC71",
    "Q2":        "#F39C12",
    "Q3":        "#E67E22",
    "Q4 (high)": "#C0392B",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for row_idx, (vary_bloc, df) in enumerate([("US", df_us), ("CN", df_cn)]):
    valid = df.dropna(subset=["lam_value", "success_rate", "cost_mult"])
    quartiles = pd.qcut(valid["cost_mult"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    bloc_label = "US" if vary_bloc == "US" else "CN"

    # Panel left: scatter, coloured by cost quartile
    ax_scatter = axes[row_idx, 0]
    for q_label, colour in QUARTILE_COLOURS.items():
        mask = (quartiles == q_label)
        sub  = valid[mask]
        ax_scatter.scatter(sub["lam_value"], sub["success_rate"],
                           color=colour, alpha=0.35, s=10, label=q_label)

    ax_scatter.set_xlabel(f"$\\lambda_{{{bloc_label}}}$", fontsize=12)
    ax_scatter.set_ylabel("Coordination success rate", fontsize=11)
    ax_scatter.set_title(
        f"$\\lambda_{{{bloc_label}}}$ × cost$_{{{bloc_label}}}$ interaction\n"
        "(coloured by cost quartile)", fontsize=10
    )
    ax_scatter.legend(title=f"cost$_{{{bloc_label}}}$ quartile", fontsize=8,
                      framealpha=0.9, markerscale=1.5)
    ax_scatter.set_xlim(LAM_LO, LAM_HI)

    # Panel right: ρ by quartile bar chart
    ax_bar = axes[row_idx, 1]
    q_rows = [r for r in quartile_rows if r["vary_bloc"] == vary_bloc]
    q_labels = [r["quartile"] for r in q_rows]
    rho_vals = [r["rho"] for r in q_rows]
    colours  = [QUARTILE_COLOURS.get(q, "grey") for q in q_labels]
    bars = ax_bar.bar(q_labels, rho_vals, color=colours, alpha=0.85, edgecolor="white")
    ax_bar.axhline(0, color="black", lw=0.8)
    ax_bar.set_ylabel(f"Spearman $\\rho$($\\lambda_{{{bloc_label}}}$, success)", fontsize=10)
    ax_bar.set_title(
        f"$\\rho$($\\lambda_{{{bloc_label}}}$, success) by cost$_{{{bloc_label}}}$ quartile\n"
        f"Baseline cost$_{{{bloc_label}}}$ = {params_base.costs[vary_bloc]:.2f}",
        fontsize=10
    )
    ax_bar.set_ylim(-0.35, 0.45)
    for bar, val in zip(bars, rho_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.015 if val >= 0 else -0.03),
                    f"{val:+.3f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=9)

fig.suptitle(
    r"Cost–Rationality Complementarity: $\lambda$ effect conditional on transition costs",
    fontsize=13, y=1.01
)
plt.tight_layout()
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved → {FIG_PATH}")
print("Done.")
