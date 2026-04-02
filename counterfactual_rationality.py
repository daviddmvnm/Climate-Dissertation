"""
counterfactual_rationality.py
─────────────────────────────
Structural counterfactual test for the rationality asymmetry finding.

Baseline GSA finding:
  λ_US ↑  →  CN first-mover prob ↑  (ρ = +0.155)
  λ_CN ↑  →  CN first-mover prob ↓  (ρ = −0.225)

Reviewer challenge: the asymmetry may reflect China's structural position
(higher costs, lower patience) rather than anything genuinely about rationality.

Three counterfactuals, each varying only λ_US and λ_CN:

  CF1 — Baseline structural parameters           (replication anchor)
  CF2 — Full structural swap (US ↔ CN)           (tests structural position thesis)
  CF3 — Patience-only swap (discount only)        (isolates patience as driver)

Interpretation:
  If ρ(λ_CN, fm_CN) sign FLIPS in CF2 → asymmetry is purely structural.
  If sign PRESERVED in CF2             → asymmetry is genuinely about rationality.
  CF3 identifies whether patience or costs/damages is the key structural driver.

Outputs:
  results/counterfactual_rationality.csv   — raw draws × outcomes
  results/counterfactual_correlations.csv  — Spearman ρ summary table
  results/figures/fig_counterfactual_rationality.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS, solve_model, monte_carlo
from run_analysis import (
    build_bloc_data, build_params, SMM_BASELINE,
    DISCOUNT, ETA, KAPPA, GAMMA, THETA,
)

# ── Settings ──────────────────────────────────────────────────────────────────
N_DRAWS  = 500    # joint draws of (λ_US, λ_CN)
N_MC     = 200    # MC paths per draw
LAM_LO   = 0.5
LAM_HI   = 3.0
LAM_FIXED = SMM_BASELINE["lam"]   # λ for EU, RoW held constant
SEED     = 99

OUT_CSV  = "results/counterfactual_rationality.csv"
CORR_CSV = "results/counterfactual_correlations.csv"
FIG_PATH = "results/figures/fig_counterfactual_rationality.png"

# ── Load data and build baseline params ───────────────────────────────────────
print("Loading data...")
bloc_data = build_bloc_data(data_dir=".")
raw       = bloc_data.set_index("bloc")

weights = {}
for bloc in PLAYERS:
    weights[bloc] = 0.5 * raw.loc[bloc, "emission_share"] + 0.5 * raw.loc[bloc, "gdp_share"]
total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}

b = SMM_BASELINE
params_base = build_params(
    raw, weights,
    ac=b["ac"], ad=b["ad"], ap=b["ap"], ab=b["ab"],
    lam=b["lam"],
)
print(f"Baseline params built. T={params_base.T}, θ={params_base.theta}")
print(f"  costs:    { {k: round(v,3) for k,v in params_base.costs.items()} }")
print(f"  discount: {params_base.discount}")
print(f"  weights:  { {k: round(v,3) for k,v in params_base.weights.items()} }")

# ── Build counterfactual structural parameter sets ────────────────────────────

# CF2: full swap of US ↔ CN structural position
params_cf2 = GameParams(**{
    **params_base.__dict__,
    "costs": {
        **params_base.costs,
        "US":  params_base.costs["CN"],
        "CN":  params_base.costs["US"],
    },
    "damages": {
        **params_base.damages,
        "US":  params_base.damages["CN"],
        "CN":  params_base.damages["US"],
    },
    "pressure": {
        **params_base.pressure,
        "US":  params_base.pressure["CN"],
        "CN":  params_base.pressure["US"],
    },
    "discount": {
        **params_base.discount,
        "US":  params_base.discount["CN"],   # CN's patience → US
        "CN":  params_base.discount["US"],   # US's patience → CN
    },
    "weights": {
        **params_base.weights,
        "US":  params_base.weights["CN"],
        "CN":  params_base.weights["US"],
    },
})

# CF3: patience-only swap (discount only)
params_cf3 = GameParams(**{
    **params_base.__dict__,
    "discount": {
        **params_base.discount,
        "US":  params_base.discount["CN"],   # CN's δ → US
        "CN":  params_base.discount["US"],   # US's δ → CN
    },
})

print("\nCounterfactual structural parameter differences:")
for label, p in [("CF2 (full swap)", params_cf2), ("CF3 (patience only)", params_cf3)]:
    print(f"  {label}:")
    print(f"    costs:    US={p.costs['US']:.3f}  CN={p.costs['CN']:.3f}")
    print(f"    discount: US={p.discount['US']:.2f}  CN={p.discount['CN']:.2f}")


# ── Core sweep function ───────────────────────────────────────────────────────

def run_lambda_sweep(structural_params, n_draws, n_mc, seed, label):
    """
    Draw (λ_US, λ_CN) pairs uniformly from [LAM_LO, LAM_HI].
    Hold λ_EU, λ_RoW at LAM_FIXED.
    For each draw, solve the model and run MC.
    Returns a DataFrame with columns: lam_US, lam_CN, fm_CN, fm_US, success_rate.
    """
    rng  = np.random.default_rng(seed)
    rows = []

    for i in range(n_draws):
        if (i + 1) % 100 == 0:
            print(f"  [{label}] draw {i+1}/{n_draws}...")

        lam_us = float(rng.uniform(LAM_LO, LAM_HI))
        lam_cn = float(rng.uniform(LAM_LO, LAM_HI))

        lam_map = {
            "US":  lam_us,
            "CN":  lam_cn,
            "EU":  LAM_FIXED,
            "RoW": LAM_FIXED,
        }

        p = GameParams(**{**structural_params.__dict__, "lam": lam_map})

        try:
            V, sigma, _, _ = solve_model(p)
            W_paths, adopt_time = monte_carlo(V, sigma, p, n_runs=n_mc,
                                              seed=int(i * 7 + seed))
            success = float((W_paths[:, -1] >= p.theta).mean())
            # First-mover: adopts in period 1
            fm_CN = float((adopt_time[:, PLAYERS.index("CN")] == 1).mean())
            fm_US = float((adopt_time[:, PLAYERS.index("US")] == 1).mean())
        except Exception:
            success = np.nan
            fm_CN   = np.nan
            fm_US   = np.nan

        rows.append({
            "counterfactual": label,
            "lam_US":         lam_us,
            "lam_CN":         lam_cn,
            "fm_CN":          fm_CN,
            "fm_US":          fm_US,
            "success_rate":   success,
        })

    return pd.DataFrame(rows)


# ── Run all three counterfactuals ─────────────────────────────────────────────

COUNTERFACTUALS = [
    ("CF1: Baseline",          params_base),
    ("CF2: Full US↔CN swap",   params_cf2),
    ("CF3: Patience-only swap", params_cf3),
]

all_dfs = []
for label, struct_params in COUNTERFACTUALS:
    print(f"\nRunning {label} ({N_DRAWS} draws × {N_MC} MC)...")
    df = run_lambda_sweep(struct_params, N_DRAWS, N_MC, seed=SEED, label=label)
    all_dfs.append(df)
    valid = df.dropna()
    rho_cn, p_cn = spearmanr(valid["lam_CN"], valid["fm_CN"])
    rho_us, p_us = spearmanr(valid["lam_US"], valid["fm_CN"])
    print(f"  ρ(λ_CN → fm_CN) = {rho_cn:+.3f}  (p={p_cn:.4f})")
    print(f"  ρ(λ_US → fm_CN) = {rho_us:+.3f}  (p={p_us:.4f})")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(OUT_CSV, index=False)
print(f"\nRaw draws saved: {OUT_CSV}")


# ── Spearman correlation summary ──────────────────────────────────────────────

corr_rows = []
for label in df_all["counterfactual"].unique():
    sub = df_all[df_all["counterfactual"] == label].dropna()
    for param_col, param_name in [("lam_CN", "λ_CN"), ("lam_US", "λ_US")]:
        for outcome_col, outcome_name in [("fm_CN", "fm_CN"), ("fm_US", "fm_US"),
                                          ("success_rate", "success_rate")]:
            rho, pval = spearmanr(sub[param_col], sub[outcome_col])
            corr_rows.append({
                "counterfactual": label,
                "parameter":      param_name,
                "outcome":        outcome_name,
                "rho":            round(rho, 4),
                "pvalue":         round(pval, 6),
                "sign":           "+" if rho >= 0 else "−",
                "significant":    pval < 0.05,
            })

df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(CORR_CSV, index=False)
print(f"Correlation table saved: {CORR_CSV}")

# ── Print summary table ───────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  RATIONALITY ASYMMETRY COUNTERFACTUAL — KEY RESULTS")
print("=" * 65)
pivot = df_corr[df_corr["outcome"] == "fm_CN"].pivot(
    index="counterfactual", columns="parameter", values="rho"
)
print(f"\n  Spearman ρ with CN first-mover probability:\n")
print(f"  {'Counterfactual':<30}  {'ρ(λ_CN, fm_CN)':>15}  {'ρ(λ_US, fm_CN)':>15}")
print(f"  {'-'*30}  {'-'*15}  {'-'*15}")
for label in pivot.index:
    r_cn = pivot.loc[label, "λ_CN"] if "λ_CN" in pivot.columns else np.nan
    r_us = pivot.loc[label, "λ_US"] if "λ_US" in pivot.columns else np.nan
    p_cn_row = df_corr[(df_corr["counterfactual"]==label) &
                       (df_corr["parameter"]=="λ_CN") &
                       (df_corr["outcome"]=="fm_CN")]["pvalue"].values
    sig = "*" if len(p_cn_row) > 0 and p_cn_row[0] < 0.05 else ""
    print(f"  {label:<30}  {r_cn:>+14.3f}{sig}  {r_us:>+15.3f}")

print(f"\n  * p < 0.05")
print("\n  Interpretation:")
cf1_sign = pivot.loc["CF1: Baseline", "λ_CN"] if "CF1: Baseline" in pivot.index else 0
cf2_sign = pivot.loc["CF2: Full US↔CN swap", "λ_CN"] if "CF2: Full US↔CN swap" in pivot.index else 0
if (cf1_sign < 0) == (cf2_sign < 0):
    print("  ✓ Sign of ρ(λ_CN, fm_CN) PRESERVED in CF2.")
    print("    Asymmetry is robust to structural position swap.")
    print("    → Report as genuine rationality asymmetry.")
else:
    print("  ✗ Sign of ρ(λ_CN, fm_CN) FLIPPED in CF2.")
    print("    Asymmetry driven by structural position, not rationality per se.")
    print("    → Re-report as a structural patience/cost finding.")

cf3_sign = pivot.loc["CF3: Patience-only swap", "λ_CN"] if "CF3: Patience-only swap" in pivot.index else 0
if (cf1_sign < 0) == (cf3_sign < 0):
    print("  CF3: Patience-only swap preserves sign → patience is NOT the key driver.")
    print("       The asymmetry survives even when discount factors are equalized.")
else:
    print("  CF3: Patience-only swap FLIPS sign → patience is the primary structural driver.")
    print("       Swapping δ alone is sufficient to reverse the asymmetry.")
print("=" * 65)


# ── Figure: CN first-mover prob vs λ_CN, all three counterfactuals ───────────

print("\nGenerating figure...")

CF_COLOURS = {
    "CF1: Baseline":           "#2980B9",
    "CF2: Full US↔CN swap":    "#C0392B",
    "CF3: Patience-only swap": "#27AE60",
}
CF_LABELS = {
    "CF1: Baseline":           "CF1: Baseline",
    "CF2: Full US↔CN swap":    "CF2: Full structural swap (US↔CN)",
    "CF3: Patience-only swap": "CF3: Patience-only swap",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: ρ(λ_CN, fm_CN) scatter / binned trend
ax = axes[0]
bins = np.linspace(LAM_LO, LAM_HI, 12)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

for label in df_all["counterfactual"].unique():
    sub = df_all[df_all["counterfactual"] == label].dropna()
    # Bin λ_CN and compute mean fm_CN per bin (marginalising over λ_US)
    binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sub["lam_CN"] >= lo) & (sub["lam_CN"] < hi)
        if mask.sum() >= 5:
            binned.append(sub.loc[mask, "fm_CN"].mean())
        else:
            binned.append(np.nan)
    binned = np.array(binned)
    valid_mask = ~np.isnan(binned)
    ax.plot(bin_centers[valid_mask], binned[valid_mask],
            color=CF_COLOURS[label], label=CF_LABELS[label],
            lw=2.2, marker="o", markersize=5)

ax.set_xlabel("$\\lambda_{CN}$ (China QRE rationality)", fontsize=11)
ax.set_ylabel("P(CN first-mover, adopts in period 1)", fontsize=11)
ax.set_title("CN First-Mover Prob. vs $\\lambda_{CN}$\n(binned mean, marginalising over $\\lambda_{US}$)", fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.axhline(0, color="grey", lw=0.6, ls="--")

# Panel B: Spearman ρ summary bar chart
ax2 = axes[1]
cf_labels_short = {
    "CF1: Baseline":           "CF1\nBaseline",
    "CF2: Full US↔CN swap":    "CF2\nFull swap",
    "CF3: Patience-only swap": "CF3\nPatience\nonly",
}
x = np.arange(len(COUNTERFACTUALS))
w = 0.35

rho_cn_vals, rho_us_vals = [], []
for label, _ in COUNTERFACTUALS:
    sub = df_corr[(df_corr["counterfactual"] == label) & (df_corr["outcome"] == "fm_CN")]
    rho_cn_vals.append(sub[sub["parameter"] == "λ_CN"]["rho"].values[0]
                       if not sub[sub["parameter"] == "λ_CN"].empty else np.nan)
    rho_us_vals.append(sub[sub["parameter"] == "λ_US"]["rho"].values[0]
                       if not sub[sub["parameter"] == "λ_US"].empty else np.nan)

bars1 = ax2.bar(x - w/2, rho_cn_vals, width=w, label="$\\rho(\\lambda_{CN},\\, fm_{CN})$",
                color="#D4AC0D", alpha=0.85, edgecolor="white")
bars2 = ax2.bar(x + w/2, rho_us_vals, width=w, label="$\\rho(\\lambda_{US},\\, fm_{CN})$",
                color="#8E44AD", alpha=0.85, edgecolor="white")

ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([cf_labels_short.get(l, l) for l, _ in COUNTERFACTUALS], fontsize=9)
ax2.set_ylabel("Spearman $\\rho$", fontsize=11)
ax2.set_title("Spearman $\\rho$ with CN First-Mover Prob.\nacross Counterfactuals", fontsize=10)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.set_ylim(-0.55, 0.55)

# Annotate bars with rho values
for bar, val in list(zip(bars1, rho_cn_vals)) + list(zip(bars2, rho_us_vals)):
    if not np.isnan(val):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + (0.02 if val >= 0 else -0.04),
                 f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=8)

plt.suptitle("Rationality Asymmetry: Structural Counterfactual Test", fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {FIG_PATH}")
print("\nAll done.")
