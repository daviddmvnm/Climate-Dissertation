"""
spec_robustness.py
──────────────────
Three functional-form robustness tests in one file.
All SMM parameters are held fixed at their calibrated values; only the
monkey-patched function changes. Each section is self-contained.

  SECTION A — Coordination threshold / stabilisation benefit  [threshold_sigmoid]
    Sigmoid  S(W) = 1/(1+exp(−η(W−θ)))  [baseline]
    Linear   S(W) = clip(W, 0, 1)
    Convex   S(W) = clip(W², 0, 1)
    → sweep analysis + GSA
    → results/figures/fig_functional_form_robustness.png
    → results/dissertation/table_functional_form_robustness.tex

  SECTION B — Transition-cost learning curve  [adoption_cost]
    Linear   c_i^0(1 − γW)              [baseline]
    Convex   c_i^0(1 − γW²)
    Concave  c_i^0(1 − γ√W)
    → GSA only
    → results/figures/fig_cost_learning_spec_test.png

  SECTION C — Political pressure  [political_pressure]
    Linear   p_i · W                    [baseline]
    Convex   p_i · W²
    Sigmoid  p_i · σ(η_p(W − θ_p))
    → GSA only
    → results/figures/fig_pressure_spec_test.png
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import climate_game
from climate_game import PLAYERS, GameParams, solve_model, monte_carlo
from climate_game import sigmoid as _sigmoid

from run_analysis import (
    build_bloc_data, build_params, SMM_BASELINE,
    make_sweep_defs, crossing_value,
    GSA_RANGES, run_mc,
)

os.makedirs(os.path.join(ROOT_DIR, "results", "figures"),      exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "results", "dissertation"), exist_ok=True)

SEP  = "=" * 70
SEP2 = "-" * 70

# ── Shared settings ────────────────────────────────────────────────────────────
N_SWEEP_PTS  = 12
SWEEP_MC     = 300
BASELINE_MC  = 1000
GSA_SAMPLES  = 600
GSA_MC       = 150

KEY_CHANNELS = ["δ_CN", "δ_US", "δ_US=δ_CN", "Global cost", "θ (threshold)", "γ (learning)"]

# ── Load data once ─────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
raw = bloc_data.set_index("bloc")
weights = {}
for bloc in PLAYERS:
    weights[bloc] = 0.5*raw.loc[bloc,"emission_share"] + 0.5*raw.loc[bloc,"gdp_share"]
total_w = sum(weights.values())
weights = {k: v/total_w for k, v in weights.items()}

b      = SMM_BASELINE
_base_params = build_params(raw, weights, ac=b["ac"], ad=b["ad"],
                             ap=b["ap"], ab=b["ab"], lam=b["lam"])
# Equalised params: α_p = α_c so pressure and cost are on the same cardinal scale
_eq_params   = build_params(raw, weights, ac=b["ac"], ad=b["ad"],
                             ap=b["ac"], ab=b["ab"], lam=b["lam"])


# ── Shared GSA runner ──────────────────────────────────────────────────────────

def run_gsa(params, seed=42):
    """
    Run lightweight GSA: N=GSA_SAMPLES independent uniform draws.
    Returns DataFrame with all drawn parameters + success_rate column.
    """
    rng  = np.random.default_rng(seed)
    rows = []
    for s_idx in range(GSA_SAMPLES):
        draw = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in GSA_RANGES.items()}
        disc       = {p: draw[f"delta_{p}"]   for p in PLAYERS}
        lam_map    = {p: draw[f"lam_{p}"]     for p in PLAYERS}
        costs_g    = {p: params.costs[p]    * draw[f"cost_{p}"]     for p in PLAYERS}
        pressure_g = {p: params.pressure[p] * draw[f"pressure_{p}"] for p in PLAYERS}
        p_gsa = GameParams(**{
            **params.__dict__,
            "discount":  disc,
            "costs":     costs_g,
            "pressure":  pressure_g,
            "lam":       lam_map,
            "theta":     draw["theta"],
            "gamma":     draw["gamma"],
            "eta":       draw["eta"],
            "kappa":     draw["kappa"],
        })
        try:
            _, _, _, _, succ, *_ = run_mc(p_gsa, GSA_MC, seed=s_idx)
        except Exception:
            succ = np.nan
        rows.append({**draw, "success_rate": succ})
    return pd.DataFrame(rows).dropna(subset=["success_rate"])


def spearman_rhos(df):
    """Return dict param → (rho, pval) against success_rate."""
    rhos = {}
    for param in GSA_RANGES:
        rho, pval = spearmanr(df[param], df["success_rate"])
        rhos[param] = (round(float(rho), 4), round(float(pval), 4))
    return rhos


def gsa_bar_figure(all_rhos, spec_labels, highlight_params,
                   suptitle, out_path):
    """
    3-panel GSA Spearman ρ bar chart.
    spec_labels: dict spec_name → panel title (include formula here for clarity)
    highlight_params: list of params to bold-border
    """
    all_params = list(GSA_RANGES.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    for ax, spec in zip(axes, spec_labels):
        rho_vals  = [all_rhos[spec][p][0] for p in all_params]
        pval_vals = [all_rhos[spec][p][1] for p in all_params]
        bar_cols  = ["#E63946" if r > 0 else "#457B9D" for r in rho_vals]
        bars = ax.barh(all_params, rho_vals, color=bar_cols,
                       alpha=0.8, edgecolor="white", linewidth=0.5)
        # Fade non-significant bars
        for bar, pv in zip(bars, pval_vals):
            if pv >= 0.05:
                bar.set_alpha(0.3)
        # Bold border on highlighted params
        for i, param in enumerate(all_params):
            if param in highlight_params:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2.0)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(spec_labels[spec], fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Spearman ρ (success rate)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.65, 0.65)

    axes[0].set_ylabel("Parameter", fontsize=9)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — COORDINATION THRESHOLD / STABILISATION BENEFIT FORM
# ══════════════════════════════════════════════════════════════════════════════
#
#  What changes: climate_game.threshold_sigmoid
#  Affects: stabilisation_benefit(W) = b·S(W)  AND  climate_damage(...) = d·(1-S(W))
#  Both benefit and damage terms use threshold_sigmoid, so this is a joint test.
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  SECTION A — BENEFIT / THRESHOLD FORM  [threshold_sigmoid]")
print(f"{SEP}")

BENEFIT_SPECS = {
    "Sigmoid": lambda W, p: float(_sigmoid(p.eta * (W - p.theta))),
    "Linear":  lambda W, p: float(np.clip(W, 0.0, 1.0)),
    "Convex":  lambda W, p: float(np.clip(W ** 2, 0.0, 1.0)),
}

# Panel titles: spec name + formula shown explicitly
BENEFIT_LABELS = {
    "Sigmoid": "Sigmoid  [baseline]\n$S(W)=\\frac{1}{1+e^{-\\eta(W-\\theta)}}$",
    "Linear":  "Linear\n$S(W) = \\mathrm{clip}(W,\\,0,1)$",
    "Convex":  "Convex\n$S(W) = \\mathrm{clip}(W^{2},\\,0,1)$",
}


def run_benefit_spec(spec_name, benefit_fn):
    print(f"\n  [{spec_name}]", flush=True)
    orig = climate_game.threshold_sigmoid
    climate_game.threshold_sigmoid = benefit_fn
    try:
        params = build_params(raw, weights, ac=b["ac"], ad=b["ad"],
                              ap=b["ap"], ab=b["ab"], lam=b["lam"])

        # Baseline MC
        _, _, _, _, success, mean_W, mean_ct, fm = run_mc(params, BASELINE_MC, seed=42)
        print(f"    P(success)={success:.3f}  mean_W={mean_W:.4f}")

        # Key sweeps
        sweep_defs = make_sweep_defs(N_SWEEP_PTS)
        key_defs   = [d for d in sweep_defs if d[0] in KEY_CHANNELS]
        sweep_rows, crossing_90 = [], {}

        for channel, x_label, xs, modifier, direction, x_baseline in key_defs:
            rates = []
            for x in xs:
                p_mod = modifier(params, x)
                try:
                    _, _, _, _, succ_m, _, _, _ = run_mc(p_mod, SWEEP_MC, seed=0)
                except Exception:
                    succ_m = np.nan
                rates.append(succ_m)
                pct = (float(x) - x_baseline) / x_baseline * 100
                sweep_rows.append({"spec": spec_name, "channel": channel,
                                   "pct_change": round(pct, 2), "success_rate": succ_m})
            c90 = crossing_value(xs, np.array(rates), 0.90, direction)
            c90_pct = round((c90 - x_baseline) / x_baseline * 100, 2) if c90 is not None else None
            crossing_90[channel] = c90_pct
            print(f"    {channel:<22} 90%-crossing: {str(c90_pct):>8}%")

        # GSA
        print(f"    Running GSA ({GSA_SAMPLES} samples)...", flush=True)
        df_gsa = run_gsa(params)
        rhos = {p: rho for p, (rho, _) in spearman_rhos(df_gsa).items()}
        top3 = sorted(rhos, key=lambda k: abs(rhos[k]), reverse=True)[:3]
        print(f"    Top-3 GSA: {', '.join(f'{p}({rhos[p]:+.2f})' for p in top3)}")

    finally:
        climate_game.threshold_sigmoid = orig

    return {
        "spec": spec_name, "success_rate": round(success, 4), "mean_W": round(mean_W, 4),
        **{f"cross90_{ch}": crossing_90.get(ch) for ch in KEY_CHANNELS},
        "top_gsa_1": top3[0], "top_gsa_2": top3[1], "top_gsa_3": top3[2],
        "gsa_rho_1": round(rhos[top3[0]], 3),
        "gsa_rho_2": round(rhos[top3[1]], 3),
        "gsa_rho_3": round(rhos[top3[2]], 3),
        "_sweep_rows": sweep_rows, "_rhos": rhos,
    }


benefit_results = []
for spec_name, benefit_fn in BENEFIT_SPECS.items():
    benefit_results.append(run_benefit_spec(spec_name, benefit_fn))


# Policy ranking preservation
def policy_ranking_preserved(base_res, alt_res):
    sentinel = 999.0
    bv = [base_res[f"cross90_{ch}"] or sentinel for ch in KEY_CHANNELS]
    av = [alt_res[f"cross90_{ch}"]  or sentinel for ch in KEY_CHANNELS]
    rho, _ = spearmanr(bv, av)
    return round(rho, 3)

baseline_benefit = benefit_results[0]   # Sigmoid
verdicts = {"Sigmoid": "baseline"}
print(f"\n  Policy ranking preservation (vs Sigmoid):")
for res in benefit_results[1:]:
    rho = policy_ranking_preserved(baseline_benefit, res)
    verdict = "PRESERVED" if rho > 0.80 else "SHIFTED"
    verdicts[res["spec"]] = f"{verdict} (ρ={rho})"
    print(f"    {res['spec']:<10}  ρ={rho:.3f}  →  {verdict}")


# ── Benefit form sweep figure ──────────────────────────────────────────────────
ch_colours = dict(zip(KEY_CHANNELS,
                      ["#2c5f8a","#e76f51","#2a9d8f","#e9c46a","#8338ec","#264653"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
for ax, res in zip(axes, benefit_results):
    df_sw = pd.DataFrame(res["_sweep_rows"])
    for ch in KEY_CHANNELS:
        sub = df_sw[df_sw["channel"] == ch].sort_values("pct_change")
        if not sub.empty:
            ax.plot(sub["pct_change"], sub["success_rate"],
                    label=ch, color=ch_colours[ch], lw=1.8)
    ax.axhline(0.90, ls="--", color="#e63946", lw=1.0, alpha=0.8)
    ax.axhline(0.95, ls=":",  color="#2a9d8f", lw=1.0, alpha=0.8)
    ax.axvline(0,    ls="-",  color="gray",    lw=0.6, alpha=0.4)
    ax.set_title(BENEFIT_LABELS[res["spec"]], fontsize=9.5, fontweight="bold")
    ax.set_xlabel("% change from SMM baseline", fontsize=8.5)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=8)
    ax.text(0.03, 0.05, f"P(success)={res['success_rate']:.3f}",
            transform=ax.transAxes, fontsize=8, color="gray")

axes[0].set_ylabel("P(coordination success)", fontsize=9)
handles = [mpatches.Patch(color=ch_colours[ch], label=ch) for ch in KEY_CHANNELS]
handles += [plt.Line2D([0],[0], ls="--", color="#e63946", lw=1.0, label="90% frontier"),
            plt.Line2D([0],[0], ls=":",  color="#2a9d8f", lw=1.0, label="95% frontier")]
fig.legend(handles=handles, loc="lower center", ncol=4,
           fontsize=8, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Section A: Coordination Threshold / Benefit Form — Key Channel Sweeps",
             fontsize=11, fontweight="bold")
plt.tight_layout()
path_a_fig = os.path.join(ROOT_DIR, "results", "figures", "fig_functional_form_robustness.png")
fig.savefig(path_a_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Figure A: {path_a_fig}")


# ── Benefit form LaTeX table ───────────────────────────────────────────────────
ch_short = {
    "δ_CN": r"$\delta_{CN}$", "δ_US": r"$\delta_{US}$",
    "δ_US=δ_CN": r"$\delta_{US}{=}\delta_{CN}$",
    "Global cost": r"Glob.\ cost",
    "θ (threshold)": r"$\theta$", "γ (learning)": r"$\gamma$",
}
spec_formula = {
    "Sigmoid": r"$\frac{1}{1+e^{-\eta(W-\theta)}}$",
    "Linear":  r"$\mathrm{clip}(W,0,1)$",
    "Convex":  r"$\mathrm{clip}(W^2,0,1)$",
}

col_header = (
    r"\begin{tabular}{llcc" + "c"*len(KEY_CHANNELS) + r"lc}" + "\n"
    r"\toprule" + "\n"
    r"Spec & $S(W)$ formula & $P(\text{success})$ & $\bar{W}$ & "
    + " & ".join(ch_short[ch] for ch in KEY_CHANNELS)
    + r" & Top GSA param & Ranking \\" + "\n"
    r"\midrule" + "\n"
)
rows_tex = ""
for res in benefit_results:
    spec = res["spec"]
    cross_cells = [("—" if res[f"cross90_{ch}"] is None
                    else f"{res[f'cross90_{ch}']:+.1f}\\%")
                   for ch in KEY_CHANNELS]
    rows_tex += (
        f"  {spec} & {spec_formula[spec]} & {res['success_rate']:.3f} & {res['mean_W']:.3f} & "
        + " & ".join(cross_cells)
        + f" & {res['top_gsa_1']} & {verdicts[spec]} \\\\\n"
    )

tex_a = (
    r"\begin{table}[ht]" + "\n"
    r"\centering" + "\n"
    r"\caption{Section A — Benefit form robustness: policy-channel 90\%-crossing thresholds "
    r"under three stabilisation-benefit / coordination-threshold specifications. "
    r"All SMM parameters held fixed. ``Ranking'' reports Spearman $\rho$ vs.\ Sigmoid baseline.}"
    "\n"
    r"\label{tab:robustness_functional_form}" + "\n"
    + col_header + rows_tex
    + r"\bottomrule" + "\n"
    + r"\end{tabular}" + "\n"
    + r"\end{table}"
)
tex_path_a = os.path.join(ROOT_DIR, "results", "dissertation", "table_functional_form_robustness.tex")
with open(tex_path_a, "w") as f:
    f.write(tex_a)
print(f"  Table A: {tex_path_a}")

# CSV
export_cols = (["spec","success_rate","mean_W"]
               + [f"cross90_{ch}" for ch in KEY_CHANNELS]
               + ["top_gsa_1","gsa_rho_1","top_gsa_2","gsa_rho_2","top_gsa_3","gsa_rho_3"])
df_benefit = pd.DataFrame([{k: r[k] for k in export_cols} for r in benefit_results])
csv_path_a = os.path.join(ROOT_DIR, "results", "robustness_functional_form.csv")
df_benefit.to_csv(csv_path_a, index=False)
print(f"  CSV A:   {csv_path_a}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — TRANSITION-COST LEARNING CURVE
# ══════════════════════════════════════════════════════════════════════════════
#
#  What changes: climate_game.adoption_cost
#  Affects: one-off transition cost paid in the adoption period only.
#  Tests whether cost-reduction dynamics change GSA sensitivity patterns.
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  SECTION B — TRANSITION-COST LEARNING CURVE  [adoption_cost]")
print(f"{SEP}")

COST_SPECS = {
    "Linear":  lambda player, W, p: max(0.0, p.costs[player] * (1 - p.gamma * W)),
    "Convex":  lambda player, W, p: max(0.0, p.costs[player] * (1 - p.gamma * (W**2))),
    "Concave": lambda player, W, p: max(0.0, p.costs[player] * (1 - p.gamma * np.sqrt(W))),
}

COST_LABELS = {
    "Linear":  "Linear  [baseline]\n$c_i^0\\,(1 - \\gamma W)$",
    "Convex":  "Convex\n$c_i^0\\,(1 - \\gamma W^{2})$",
    "Concave": "Concave\n$c_i^0\\,(1 - \\gamma \\sqrt{W})$",
}

COST_FOCUS = ["gamma", "cost_US", "cost_EU", "cost_CN", "cost_RoW"]

cost_rhos = {}
cost_success = {}
for spec_name, cost_fn in COST_SPECS.items():
    print(f"\n  [{spec_name}] GSA ({GSA_SAMPLES} samples)...", flush=True)
    orig = climate_game.adoption_cost
    climate_game.adoption_cost = cost_fn
    try:
        params = build_params(raw, weights, ac=b["ac"], ad=b["ad"],
                              ap=b["ap"], ab=b["ab"], lam=b["lam"])
        _, _, _, _, base_succ, *_ = run_mc(params, 1000, seed=42)
        cost_success[spec_name] = round(base_succ, 4)
        df_gsa = run_gsa(params)
        cost_rhos[spec_name] = spearman_rhos(df_gsa)
        print(f"    P(success)={base_succ:.3f}", flush=True)
    finally:
        climate_game.adoption_cost = orig

# Focus param comparison (console)
print(f"\n  COST FOCUS PARAM CORRELATIONS  (* = p<0.05)")
print(f"  {'Parameter':<16}", end="")
for spec in COST_SPECS:
    print(f"  {spec:>14}", end="")
print()
print(f"  {'-'*60}")
for param in COST_FOCUS:
    print(f"  {param:<16}", end="")
    for spec in COST_SPECS:
        rho, pval = cost_rhos[spec][param]
        sig = "*" if pval < 0.05 else " "
        print(f"  {rho:>+12.4f}{sig}", end="")
    print()

# Figure B — GSA bars with formula in panel title
gsa_bar_figure(
    all_rhos     = cost_rhos,
    spec_labels  = COST_LABELS,
    highlight_params = COST_FOCUS,
    suptitle = (f"Section B: Cost Learning Curve — GSA Correlations  "
                f"(n={GSA_SAMPLES} samples, bold = γ + cost params, faded = p≥0.05)"),
    out_path = os.path.join(ROOT_DIR, "results", "figures", "fig_cost_learning_spec_test.png"),
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — POLITICAL PRESSURE SPECIFICATION
# ══════════════════════════════════════════════════════════════════════════════
#
#  What changes: climate_game.political_pressure
#  Affects: delay payoff only — r_i^D = −p_i·f(W_t) + r_i(t,W_t)
#  Tests (a) whether the bandwagon dynamic changes character under different
#  functional forms AND (b) whether findings are a cardinal scaling artefact
#  by also running with equalised α_p = α_c.
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  SECTION C — POLITICAL PRESSURE SPECIFICATION  [political_pressure]")
print(f"  Baseline α_p={b['ap']:.4f}   Equalised α_p=α_c={b['ac']:.4f}")
print(f"{SEP}")

THETA_P = 0.40   # Visibility threshold for sigmoid pressure
ETA_P   = 10.0

PRESSURE_SPECS = {
    "Linear":  lambda player, W, p: p.pressure[player] * W,
    "Convex":  lambda player, W, p: p.pressure[player] * (W**2),
    "Sigmoid": lambda player, W, p: p.pressure[player] * float(_sigmoid(ETA_P*(W - THETA_P))),
}

PRESSURE_LABELS = {
    "Linear":  "Linear  [baseline]\n$p_i \\cdot W$",
    "Convex":  "Convex\n$p_i \\cdot W^{2}$",
    "Sigmoid": f"Sigmoid  (θ_p={THETA_P})\n$p_i \\cdot \\sigma(\\eta_p(W - \\theta_p))$",
}

PRESSURE_FOCUS = ["pressure_US", "pressure_EU", "pressure_CN", "pressure_RoW"]

# pressure_rhos[spec]["baseline"] and ["equalised"] → dict param → (rho, pval)
pressure_rhos = {}

for spec_name, pressure_fn in PRESSURE_SPECS.items():
    pressure_rhos[spec_name] = {}
    for cond_name, cond_params in [("baseline", _base_params), ("equalised", _eq_params)]:
        print(f"\n  [{spec_name} / {cond_name}] GSA ({GSA_SAMPLES} samples)...", flush=True)
        orig = climate_game.political_pressure
        climate_game.political_pressure = pressure_fn
        try:
            df_gsa = run_gsa(cond_params)
            pressure_rhos[spec_name][cond_name] = spearman_rhos(df_gsa)
        finally:
            climate_game.political_pressure = orig

# Console comparison: for each spec, show baseline vs equalised for focus params
print(f"\n  PRESSURE FOCUS PARAM CORRELATIONS  (* = p<0.05)")
print(f"  Key: (B)=baseline α_p={b['ap']:.2f}   (E)=equalised α_p=α_c={b['ac']:.2f}")
for spec_name in PRESSURE_SPECS:
    print(f"\n  [{spec_name}]")
    print(f"  {'Parameter':<18} {'Base ρ':>8}   {'Equal ρ':>8}   {'Δρ':>7}")
    print(f"  {'-'*50}")
    for param in PRESSURE_FOCUS:
        b_rho, b_pval = pressure_rhos[spec_name]["baseline"][param]
        e_rho, e_pval = pressure_rhos[spec_name]["equalised"][param]
        bs = "*" if b_pval < 0.05 else " "
        es = "*" if e_pval < 0.05 else " "
        print(f"  {param:<18} {b_rho:>+8.4f}{bs}  {e_rho:>+8.4f}{es}  {e_rho-b_rho:>+7.4f}")

# Figure C — 2-row × 3-col: top row = baseline, bottom row = equalised
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True, sharex=False)
all_params = list(GSA_RANGES.keys())

for col, (spec_name, _) in enumerate(PRESSURE_SPECS.items()):
    for row, cond_name in enumerate(["baseline", "equalised"]):
        ax = axes[row, col]
        rho_vals  = [pressure_rhos[spec_name][cond_name][p][0] for p in all_params]
        pval_vals = [pressure_rhos[spec_name][cond_name][p][1] for p in all_params]
        bar_cols  = ["#E63946" if r > 0 else "#457B9D" for r in rho_vals]
        bars = ax.barh(all_params, rho_vals, color=bar_cols,
                       alpha=0.8, edgecolor="white", linewidth=0.5)
        for bar, pv in zip(bars, pval_vals):
            if pv >= 0.05:
                bar.set_alpha(0.3)
        for i, param in enumerate(all_params):
            if param in PRESSURE_FOCUS:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2.0)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-0.65, 0.65)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Spearman ρ", fontsize=8)

        cond_label = (f"α_p={b['ap']:.2f} [baseline]"
                      if cond_name == "baseline"
                      else f"α_p=α_c={b['ac']:.2f} [equalised]")
        ax.set_title(f"{PRESSURE_LABELS[spec_name]}\n{cond_label}",
                     fontsize=8.5, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Parameter", fontsize=9)

fig.suptitle(
    f"Section C: Political Pressure — GSA Correlations\n"
    f"Top row: baseline (α_p={b['ap']:.2f})  |  "
    f"Bottom row: equalised (α_p=α_c={b['ac']:.2f})\n"
    f"Bold border = pressure params, faded = p≥0.05  "
    f"[Sigmoid spec: θ_p={THETA_P}, η_p={ETA_P}]",
    fontsize=10, fontweight="bold",
)
plt.tight_layout()
fig_c_path = os.path.join(ROOT_DIR, "results", "figures", "fig_pressure_spec_test.png")
fig.savefig(fig_c_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Figure C: {fig_c_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  SPEC ROBUSTNESS — COMPLETE")
print(f"{SEP}")
print("  Section A outputs:")
print(f"    {path_a_fig}")
print(f"    {tex_path_a}")
print(f"    {csv_path_a}")
print("  Section B outputs:")
print("    results/figures/fig_cost_learning_spec_test.png")
print("  Section C outputs:")
print(f"    {fig_c_path}")
print(f"    (3 specs × baseline α_p={b['ap']:.4f} + equalised α_p=α_c={b['ac']:.4f})")
print(f"{SEP}\n")
