"""
test_pressure_robustness.py
───────────────────────────
Robustness check: is the GSA/sweep pressure-irrelevance finding a
structural result or a cardinal scaling artefact?

Runs both the BASELINE and EQUALISED (α_p = α_c) GSA and sweeps, then
prints one mega comparison table.

Run as:
    python test_pressure_robustness.py          # full (1000 GSA, 12-pt sweeps)
    python test_pressure_robustness.py --fast   # quick (200 GSA, 8-pt sweeps)
"""

import argparse
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
sys.path.insert(0, os.path.join(ROOT_DIR, "calibration"))
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from climate_game import PLAYERS, GameParams
from run_analysis import (
    build_bloc_data, run_mc, crossing_value,
    GSA_RANGES, GSA_OUTCOMES, make_sweep_defs, OUT_DIR, FIG_DIR,
)
from smm_calibration import build_smm_params

SEP  = "=" * 70
SEP2 = "-" * 70

PRESSURE_PARAMS   = ["pressure_US", "pressure_EU", "pressure_CN", "pressure_RoW"]
PRESSURE_CHANNELS = ["US pressure", "CN pressure", "Global pressure"]
GSA_REPORT        = ["success_rate", "mean_coord_time", "fm_EU", "fm_US", "fm_CN"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_weights(raw):
    weights = {b: 0.5 * raw.loc[b, "emission_share"] + 0.5 * raw.loc[b, "gdp_share"]
               for b in PLAYERS}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def run_gsa(params, n_samples, gsa_mc, seed=42):
    rng  = np.random.default_rng(seed)
    rows = []
    for s_idx in range(n_samples):
        if (s_idx + 1) % 100 == 0:
            print(f"      {s_idx+1}/{n_samples}...", flush=True)
        draw = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in GSA_RANGES.items()}
        p_gsa = GameParams(**{
            **params.__dict__,
            "discount": {b: draw[f"delta_{b}"] for b in PLAYERS},
            "costs":    {b: params.costs[b]    * draw[f"cost_{b}"]     for b in PLAYERS},
            "pressure": {b: params.pressure[b] * draw[f"pressure_{b}"] for b in PLAYERS},
            "lam":      {b: draw[f"lam_{b}"]   for b in PLAYERS},
            "theta":    draw["theta"],
            "gamma":    draw["gamma"],
            "eta":      draw["eta"],
            "kappa":    draw["kappa"],
        })
        try:
            _, _, _, _, succ, mW, mct, fm = run_mc(p_gsa, gsa_mc, seed=s_idx)
        except Exception:
            succ, mW, mct, fm = np.nan, np.nan, np.nan, {p: np.nan for p in PLAYERS}
        row = {**draw, "success_rate": succ, "mean_W": mW, "mean_coord_time": mct}
        for p in PLAYERS:
            row[f"fm_{p}"] = fm[p]
        rows.append(row)

    df = pd.DataFrame(rows)
    params_cols = list(GSA_RANGES.keys())
    corr_rows = []
    for outcome in GSA_OUTCOMES:
        valid = df[[*params_cols, outcome]].dropna()
        for param in params_cols:
            rho, pval = spearmanr(valid[param], valid[outcome])
            corr_rows.append({"parameter": param, "outcome": outcome,
                               "rho": round(rho, 4), "pvalue": round(pval, 4)})
    return df, pd.DataFrame(corr_rows)


def run_sweeps(params, n_pts, sweep_mc):
    defs    = make_sweep_defs(n_pts)
    results = {}
    for channel, x_label, xs, modifier, direction, x_baseline in defs:
        rates = []
        for x in xs:
            try:
                _, _, _, _, succ_m, _, _, _ = run_mc(modifier(params, x), sweep_mc, seed=0)
            except Exception:
                succ_m = np.nan
            rates.append(succ_m)
        rates  = np.array(rates)
        c90    = crossing_value(xs, rates, 0.90, direction)
        c95    = crossing_value(xs, rates, 0.95, direction)
        c90pct = round((c90 - x_baseline) / x_baseline * 100, 1) if c90 is not None else None
        c95pct = round((c95 - x_baseline) / x_baseline * 100, 1) if c95 is not None else None
        results[channel] = {"c90_pct": c90pct, "c95_pct": c95pct,
                            "xs": xs, "rates": rates, "baseline": x_baseline}
    return results


# ── Print helpers ─────────────────────────────────────────────────────────────

def sig_star(pval):
    return "**" if pval < 0.01 else ("*" if pval < 0.05 else "ns")


def print_gsa_table(base_corr, eq_corr, outcome):
    b = base_corr[base_corr["outcome"] == outcome].set_index("parameter")
    f = eq_corr[eq_corr["outcome"] == outcome].set_index("parameter")

    # all params, sorted by |forced ρ| desc
    all_params = list(GSA_RANGES.keys())
    rows = []
    for p in all_params:
        b_rho  = b.loc[p, "rho"]   if p in b.index else np.nan
        b_pval = b.loc[p, "pvalue"] if p in b.index else np.nan
        f_rho  = f.loc[p, "rho"]   if p in f.index else np.nan
        f_pval = f.loc[p, "pvalue"] if p in f.index else np.nan
        rows.append((p, b_rho, b_pval, f_rho, f_pval))
    rows.sort(key=lambda r: abs(r[3]) if not np.isnan(r[3]) else 0, reverse=True)

    print(f"  {'Parameter':<18} {'Base ρ':>8} {'':3} {'Equal ρ':>8} {'':3} {'Δρ':>7}")
    print(f"  {'-'*52}")
    for p, br, bp, fr, fp in rows:
        sb = sig_star(bp) if not np.isnan(bp) else ""
        sf = sig_star(fp) if not np.isnan(fp) else ""
        delta = fr - br if not (np.isnan(fr) or np.isnan(br)) else np.nan
        delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "  n/a"
        print(f"  {p:<18} {br:>8.4f} {sb:<3} {fr:>8.4f} {sf:<3} {delta_s:>7}")


def print_sweep_table(base_sw, eq_sw):
    all_channels = list(base_sw.keys())
    print(f"  {'Channel':<22} {'Base 90%':>10} {'Equal 90%':>10} {'Base 95%':>10} {'Equal 95%':>10}")
    print(f"  {'-'*65}")
    for ch in all_channels:
        b = base_sw.get(ch, {})
        e = eq_sw.get(ch, {})
        b90 = f"{b['c90_pct']:+.1f}%" if b.get("c90_pct") is not None else "  >"
        e90 = f"{e['c90_pct']:+.1f}%" if e.get("c90_pct") is not None else "  >"
        b95 = f"{b['c95_pct']:+.1f}%" if b.get("c95_pct") is not None else "  >"
        e95 = f"{e['c95_pct']:+.1f}%" if e.get("c95_pct") is not None else "  >"
        # flag pressure channels
        flag = " ◄" if any(pc.lower() in ch.lower() for pc in ["pressure", "pres"]) else ""
        print(f"  {ch:<22} {b90:>10} {e90:>10} {b95:>10} {e95:>10}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    N_GSA    = 200  if args.fast else 1000
    GSA_MC   = 50   if args.fast else 200
    N_PTS    = 8    if args.fast else 12
    SWEEP_MC = 100  if args.fast else 300

    print("Loading data...", flush=True)
    bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
    raw       = bloc_data.set_index("bloc")
    weights   = build_weights(raw)

    theta = np.load(os.path.join(ROOT_DIR, "results", "smm_best_theta.npy"))
    ac, ad, ap, ab = theta
    print(f"SMM estimates: α_c={ac:.4f}  α_d={ad:.4f}  α_p={ap:.4f}  α_b={ab:.4f}")

    params_base = build_smm_params(ac, ad, ap,    ab, raw, weights)
    params_eq   = build_smm_params(ac, ad, ac,    ab, raw, weights)  # α_p = α_c

    print(f"Baseline EU pressure : {params_base.pressure['EU']:.4f}")
    print(f"Equalised EU pressure: {params_eq.pressure['EU']:.4f}  "
          f"({params_eq.pressure['EU']/params_base.pressure['EU']:.1f}x scale-up)")

    # ── GSA ───────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BASELINE GSA  (n={N_GSA}, mc={GSA_MC})")
    print(SEP, flush=True)
    df_base_gsa, base_corr = run_gsa(params_base, N_GSA, GSA_MC, seed=42)
    base_corr.to_csv(os.path.join(OUT_DIR, "pressure_rob_baseline_corr.csv"), index=False)

    print(f"\n{SEP}")
    print(f"  EQUALISED GSA  (α_p=α_c={ac:.4f}, n={N_GSA}, mc={GSA_MC})")
    print(SEP, flush=True)
    df_eq_gsa, eq_corr = run_gsa(params_eq, N_GSA, GSA_MC, seed=42)
    eq_corr.to_csv(os.path.join(OUT_DIR, "pressure_rob_equalised_corr.csv"), index=False)

    # ── Sweeps ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BASELINE SWEEPS  ({N_PTS}-pt, mc={SWEEP_MC})")
    print(SEP, flush=True)
    base_sw = run_sweeps(params_base, N_PTS, SWEEP_MC)

    print(f"\n{SEP}")
    print(f"  EQUALISED SWEEPS  (α_p=α_c={ac:.4f}, {N_PTS}-pt, mc={SWEEP_MC})")
    print(SEP, flush=True)
    eq_sw = run_sweeps(params_eq, N_PTS, SWEEP_MC)

    # ── MEGA TABLE ────────────────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  MEGA COMPARISON TABLE — BASELINE vs EQUALISED (α_p=α_c={ac:.2f})")
    print(f"  (* p<0.05  ** p<0.01  ns  ◄ = pressure channel)")
    print(f"{'#'*70}")

    for outcome_label, outcome_col in [
        ("GSA: Coordination Success (success_rate)",   "success_rate"),
        ("GSA: Coordination Timing (mean_coord_time)", "mean_coord_time"),
        ("GSA: EU First-Mover (fm_EU)",                "fm_EU"),
        ("GSA: US First-Mover (fm_US)",                "fm_US"),
        ("GSA: CN First-Mover (fm_CN)",                "fm_CN"),
    ]:
        print(f"\n  ── {outcome_label}")
        print_gsa_table(base_corr, eq_corr, outcome_col)

    print(f"\n  ── SWEEPS: 90%/95% crossing (% change from baseline param value)")
    print_sweep_table(base_sw, eq_sw)

    # ── Hierarchy summary ─────────────────────────────────────────────────────
    print(f"\n  ── CHANNEL HIERARCHY  (mean |ρ| across outcomes)")
    print(f"  {'Channel':<16} {'Baseline':>10} {'Equalised':>10}")
    print(f"  {'-'*40}")
    cost_p  = ["cost_US", "cost_EU", "cost_CN", "cost_RoW"]
    disc_p  = ["delta_US", "delta_EU", "delta_CN", "delta_RoW"]
    pres_p  = PRESSURE_PARAMS
    lam_p   = ["lam_US", "lam_EU", "lam_CN", "lam_RoW"]
    outcomes_h = ["success_rate", "mean_coord_time", "fm_EU", "fm_US", "fm_CN"]
    hierarchy = []
    for label, plist in [("Costs", cost_p), ("Patience", disc_p),
                         ("Pressure", pres_p), ("Rationality", lam_p)]:
        b_mean = base_corr[base_corr["parameter"].isin(plist) &
                           base_corr["outcome"].isin(outcomes_h)]["rho"].abs().mean()
        e_mean = eq_corr[eq_corr["parameter"].isin(plist) &
                         eq_corr["outcome"].isin(outcomes_h)]["rho"].abs().mean()
        hierarchy.append({"channel": label, "baseline": b_mean, "equalised": e_mean})
        print(f"  {label:<16} {b_mean:>10.4f} {e_mean:>10.4f}")

    print(f"\n{'#'*70}\n")

    # ── Save CSV table ────────────────────────────────────────────────────────
    # Full GSA comparison (all params x outcomes)
    gsa_rows = []
    for outcome_col in outcomes_h:
        b = base_corr[base_corr["outcome"] == outcome_col].set_index("parameter")
        f = eq_corr[eq_corr["outcome"] == outcome_col].set_index("parameter")
        for p in list(GSA_RANGES.keys()):
            gsa_rows.append({
                "outcome":       outcome_col,
                "parameter":     p,
                "baseline_rho":  b.loc[p, "rho"]    if p in b.index else np.nan,
                "baseline_pval": b.loc[p, "pvalue"]  if p in b.index else np.nan,
                "equalised_rho": f.loc[p, "rho"]    if p in f.index else np.nan,
                "equalised_pval":f.loc[p, "pvalue"]  if p in f.index else np.nan,
                "delta_rho":     (f.loc[p, "rho"] - b.loc[p, "rho"])
                                 if (p in b.index and p in f.index) else np.nan,
            })
    df_gsa_cmp = pd.DataFrame(gsa_rows)
    gsa_path = os.path.join(OUT_DIR, "pressure_robustness_gsa_comparison.csv")
    df_gsa_cmp.to_csv(gsa_path, index=False)

    # Sweep comparison (pressure channels only)
    sw_rows = []
    for ch in list(base_sw.keys()):
        b = base_sw.get(ch, {})
        e = eq_sw.get(ch, {})
        sw_rows.append({
            "channel":       ch,
            "baseline_c90":  b.get("c90_pct"),
            "baseline_c95":  b.get("c95_pct"),
            "equalised_c90": e.get("c90_pct"),
            "equalised_c95": e.get("c95_pct"),
        })
    df_sw_cmp = pd.DataFrame(sw_rows)
    sw_path = os.path.join(OUT_DIR, "pressure_robustness_sweep_comparison.csv")
    df_sw_cmp.to_csv(sw_path, index=False)

    # Hierarchy summary
    df_hier = pd.DataFrame(hierarchy)
    hier_path = os.path.join(OUT_DIR, "pressure_robustness_hierarchy.csv")
    df_hier.to_csv(hier_path, index=False)

    print(f"Saved: {gsa_path}")
    print(f"Saved: {sw_path}")
    print(f"Saved: {hier_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: channel hierarchy bar chart
    ax = axes[0]
    channels = [r["channel"] for r in hierarchy]
    x = np.arange(len(channels))
    w = 0.35
    bars_b = ax.bar(x - w/2, [r["baseline"]  for r in hierarchy], w,
                    label="Baseline (α_p=0.10)", color="#457B9D", alpha=0.85)
    bars_e = ax.bar(x + w/2, [r["equalised"] for r in hierarchy], w,
                    label=f"Equalised (α_p=α_c={ac:.2f})", color="#E63946", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, fontsize=11)
    ax.set_ylabel("Mean |Spearman ρ| across outcomes", fontsize=10)
    ax.set_title("Channel Hierarchy: Baseline vs Equalised", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(r["baseline"] for r in hierarchy) * 1.4)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_e:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Panel B: pressure correlations heatmap (baseline vs equalised, success_rate)
    ax = axes[1]
    pres_labels = ["pressure_US", "pressure_EU", "pressure_CN", "pressure_RoW"]
    outcomes_plot = ["success_rate", "mean_coord_time", "fm_EU", "fm_US", "fm_CN"]
    outcome_labels = ["Success", "Timing", "fm_EU", "fm_US", "fm_CN"]

    data_b = np.array([[base_corr[(base_corr["parameter"]==p) &
                                   (base_corr["outcome"]==o)]["rho"].values[0]
                         for o in outcomes_plot] for p in pres_labels])
    data_e = np.array([[eq_corr[(eq_corr["parameter"]==p) &
                                 (eq_corr["outcome"]==o)]["rho"].values[0]
                         for o in outcomes_plot] for p in pres_labels])

    # Stack baseline (top) and equalised (bottom)
    combined = np.vstack([data_b, data_e])
    row_labels = ([f"{p.replace('pressure_','')}\nbaseline" for p in pres_labels] +
                  [f"{p.replace('pressure_','')}\nequalised" for p in pres_labels])

    im = ax.imshow(combined, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(outcomes_plot)))
    ax.set_xticklabels(outcome_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.axhline(3.5, color="black", linewidth=1.5, linestyle="--")
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            ax.text(j, i, f"{combined[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(combined[i,j]) > 0.15 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Pressure Correlations: Baseline vs Equalised", fontsize=12, fontweight="bold")

    fig.suptitle(f"Pressure Robustness Check  (α_p: {ap:.2f} → {ac:.2f})", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "pressure_robustness.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
