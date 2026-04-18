"""
discount_ordering_robustness.py
───────────────────────────────
Appendix D: Robustness of the patience-dominates finding to the assumed
bloc-level discount ordering and spacing.

At each of eight alternative discount configurations the SMM-calibrated
cardinal vector (α_c, α_d, α_p, α_b, λ) is held fixed and only the
per-bloc δ changes. Key success-rate sweep crossings (percentage shifts
from *each configuration's own* baseline) are recomputed for:

    δ_US = δ_CN  (joint patience channel)
    Global cost
    δ_US   δ_CN   δ_EU
    θ (coordination threshold)

Uses a coarser 14-point grid with 250 Monte Carlo draws per point (per
the footnote in §Appendix D of the dissertation).

Outputs
-------
  results/discount_ordering_robustness.csv
  results/dissertation/table_discount_ordering_robustness.tex
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS, solve_model, monte_carlo
from run_analysis import (
    build_bloc_data, build_params, SMM_BASELINE, THETA, crossing_value,
)

# ── Settings ──────────────────────────────────────────────────────────────
N_SWEEP_PTS = 14
SWEEP_MC    = 250
SEED        = 42

OUT_CSV = os.path.join(ROOT_DIR, "results", "discount_ordering_robustness.csv")
OUT_TEX = os.path.join(ROOT_DIR, "results", "dissertation",
                       "table_discount_ordering_robustness.tex")
os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)


# ── Eight discount configurations ─────────────────────────────────────────
# Each entry: (key, pretty_label, {bloc: δ})
CONFIGS = [
    ("baseline",          "Baseline (EU>CN>US>RoW)",
        {"EU": 0.85, "CN": 0.80, "US": 0.75, "RoW": 0.70}),
    ("us_cn_flip",        "US/CN flipped",
        {"EU": 0.85, "US": 0.80, "CN": 0.75, "RoW": 0.70}),
    ("all_equal",         "All equalised (0.775)",
        {"EU": 0.775, "CN": 0.775, "US": 0.775, "RoW": 0.775}),
    ("us_most_patient",   "US most patient",
        {"US": 0.85, "EU": 0.80, "CN": 0.75, "RoW": 0.70}),
    ("cn_most_patient",   "CN most patient",
        {"CN": 0.85, "EU": 0.80, "US": 0.75, "RoW": 0.70}),
    ("compressed_range",  "Compressed 0.78/0.82",
        {"EU": 0.82, "CN": 0.8067, "US": 0.7933, "RoW": 0.78}),
    ("uniform_0.02",      "Uniform 0.02 gaps",
        {"EU": 0.82, "CN": 0.80, "US": 0.78, "RoW": 0.76}),
    ("uniform_0.08",      "Uniform 0.08 gaps",
        {"EU": 0.86, "CN": 0.78, "US": 0.70, "RoW": 0.62}),
]


# ── Helpers ───────────────────────────────────────────────────────────────
def run_mc(params, n_mc, seed=SEED):
    V, sig, _, _ = solve_model(params)
    W_paths, _   = monte_carlo(V, sig, params, n_runs=n_mc, seed=seed)
    success = float(np.mean(W_paths[:, -1] >= params.theta))
    return success


def make_sweep_defs(disc):
    """Config-specific sweeps. x_baseline is the value under *this* config."""
    d_us, d_cn, d_eu, d_row = disc["US"], disc["CN"], disc["EU"], disc["RoW"]
    joint_base = 0.5 * (d_us + d_cn)

    defs = []

    # Joint US/CN patience — sweep both to the same value v
    defs.append((
        "δ_US=δ_CN", np.linspace(0.50, 0.95, N_SWEEP_PTS),
        lambda p, v: GameParams(**{**p.__dict__,
            "discount": {**p.discount, "US": v, "CN": v}}),
        "inc", joint_base))

    # Individual discount factors
    defs.append((
        "δ_US", np.linspace(0.50, 0.95, N_SWEEP_PTS),
        lambda p, v: GameParams(**{**p.__dict__,
            "discount": {**p.discount, "US": v}}),
        "inc", d_us))

    defs.append((
        "δ_CN", np.linspace(0.50, 0.95, N_SWEEP_PTS),
        lambda p, v: GameParams(**{**p.__dict__,
            "discount": {**p.discount, "CN": v}}),
        "inc", d_cn))

    defs.append((
        "δ_EU", np.linspace(0.70, 0.95, N_SWEEP_PTS),
        lambda p, v: GameParams(**{**p.__dict__,
            "discount": {**p.discount, "EU": v}}),
        "inc", d_eu))

    # Global cost multiplier (baseline 1.0, lower helps)
    defs.append((
        "Global cost", np.linspace(0.30, 1.30, N_SWEEP_PTS),
        lambda p, v: GameParams(**{**p.__dict__,
            "costs": {k: c * v for k, c in p.costs.items()}}),
        "dec", 1.0))

    # θ threshold (lower helps)
    defs.append((
        "θ", np.linspace(0.50, 0.95, N_SWEEP_PTS),
        lambda p, v: GameParams(**{**p.__dict__, "theta": v}),
        "dec", THETA))

    return defs


def sweep_crossings_for_config(params_base, disc):
    """Run all channels for one discount configuration, return crossings."""
    out = {}
    for channel, xs, modifier, direction, x_base in make_sweep_defs(disc):
        rates = []
        for x in xs:
            p_mod = modifier(params_base, float(x))
            try:
                rates.append(run_mc(p_mod, SWEEP_MC, seed=0))
            except Exception:
                rates.append(np.nan)
        rates = np.array(rates)
        c90_abs = crossing_value(xs, rates, 0.90, direction)
        c95_abs = crossing_value(xs, rates, 0.95, direction)
        c90_pct = (None if c90_abs is None
                   else round((c90_abs - x_base) / x_base * 100, 2))
        c95_pct = (None if c95_abs is None
                   else round((c95_abs - x_base) / x_base * 100, 2))
        out[channel] = {"cross_90": c90_pct, "cross_95": c95_pct}
    return out


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...", flush=True)
    bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
    raw = bloc_data.set_index("bloc")
    weights = {}
    for bloc in PLAYERS:
        weights[bloc] = (0.3 * raw.loc[bloc, "emission_share"]
                       + 0.7 * raw.loc[bloc, "gdp_share"])
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    b = SMM_BASELINE
    print(f"Baseline α_c={b['ac']:.3f} α_d={b['ad']:.3f} "
          f"α_spill={b['a_spill']:.3f} α_b={b['ab']:.3f} λ={b['lam']:.2f}")
    print(f"Grid: {N_SWEEP_PTS} points × {SWEEP_MC} MC\n")

    channel_order = ["δ_US=δ_CN", "Global cost", "δ_US", "δ_CN", "δ_EU", "θ"]
    rows = []

    for key, label, disc in CONFIGS:
        print(f"── {label}  δ={disc}")
        params = build_params(
            raw, weights,
            ac=b["ac"], ad=b["ad"], a_spill=b["a_spill"], ab=b["ab"],
            lam=b["lam"], discount=dict(disc),
        )
        base_success = run_mc(params, SWEEP_MC * 4, seed=SEED)
        print(f"   baseline success = {base_success:.3f}")

        crossings = sweep_crossings_for_config(params, disc)

        row = {"config": key, "label": label, "baseline_success": round(base_success, 3)}
        for bloc in PLAYERS:
            row[f"delta_{bloc}"] = disc[bloc]
        for ch in channel_order:
            row[f"{ch}_90"] = crossings[ch]["cross_90"]
            row[f"{ch}_95"] = crossings[ch]["cross_95"]
        rows.append(row)

        for ch in channel_order:
            c90 = crossings[ch]["cross_90"]
            c95 = crossings[ch]["cross_95"]
            s90 = f"{c90:+.2f}%" if c90 is not None else "absent"
            s95 = f"{c95:+.2f}%" if c95 is not None else "absent"
            print(f"     {ch:<14}  90%={s90:>9}   95%={s95:>9}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    # ── LaTeX table (90% crossings only, 8 rows × 6 lever columns) ────────
    def fmt(v):
        return "absent" if v is None or pd.isna(v) else f"{v:+.1f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Robustness of the patience-dominates finding across eight "
        r"alternative discount-factor configurations. The SMM-calibrated cardinal "
        r"vector ($\alpha_c, \alpha_d, \alpha_p, \alpha_b, \lambda$) is held fixed; "
        r"only the per-bloc discount factors change. Each entry is the percentage "
        r"shift from that configuration's own baseline required to reach a 90\% "
        r"coordination success rate on the given sweep channel. ``absent'' "
        r"indicates the 90\% threshold is not reached within the sweep range. "
        f"Grid: {N_SWEEP_PTS} points, {SWEEP_MC} Monte Carlo draws per point."
        r"}",
        r"\label{tab:discount_ordering_robustness}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Configuration & $P_0$ & $\delta_{US}{=}\delta_{CN}$ & Global cost & "
        r"$\delta_{US}$ & $\delta_{CN}$ & $\delta_{EU}$ & $\theta$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['baseline_success']:.2f} & "
            f"{fmt(r['δ_US=δ_CN_90'])} & {fmt(r['Global cost_90'])} & "
            f"{fmt(r['δ_US_90'])} & {fmt(r['δ_CN_90'])} & "
            f"{fmt(r['δ_EU_90'])} & {fmt(r['θ_90'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {OUT_TEX}")
