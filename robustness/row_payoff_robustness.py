"""
row_payoff_robustness.py
────────────────────────
Appendix D: Robustness of the main findings to alternative RoW payoff
compositions.

The baseline calibration aggregates all developing economies into a single
Rest-of-World (RoW) bloc whose payoff parameters are derived from bloc-averaged
World Bank indicators. This script holds RoW's coalition weight fixed and
re-runs the analysis under three stylised RoW archetypes (India-like,
ASEAN-like, LATAM-like) plus the baseline. The question is whether the policy
hierarchy reorders under plausible alternative RoW compositions.

Holds fixed: RoW coalition weight w_RoW; all non-RoW per-bloc parameters; the
SMM-calibrated cardinal vector; all global parameters.

Varies (RoW entries only): costs, c_tilde (scaled together), damages, pressure,
discount.

Outputs
-------
  results/row_payoff_robustness.csv
  results/dissertation/table_row_payoff_robustness.tex
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
SWEEP_MC    = 200
BASE_MC     = 200
SEED        = 42

OUT_CSV = os.path.join(ROOT_DIR, "results", "row_payoff_robustness.csv")
OUT_TEX = os.path.join(ROOT_DIR, "results", "dissertation",
                       "table_row_payoff_robustness.tex")
os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)


# ── RoW archetypes ────────────────────────────────────────────────────────
# (key, label, cost multiplier, damage multiplier, pressure multiplier, δ_RoW)
ARCHETYPES = [
    ("baseline",    "Baseline (RoW-as-calibrated)", 1.00, 1.00, 1.00, 0.70),
    ("india_like",  "India-like (high cost, short horizon)",    1.25, 1.00, 1.00, 0.65),
    ("asean_like",  "ASEAN-like (trade- and damage-exposed)",   1.00, 1.40, 1.40, 0.70),
    ("latam_like",  "LATAM-like (longer horizon, moderate)",    1.10, 1.15, 1.00, 0.78),
]


# ── Helpers ───────────────────────────────────────────────────────────────
def apply_archetype(p_base, m_cost, m_dam, m_press, d_row):
    """Return a copy of p_base with only RoW payoff entries overridden."""
    return GameParams(**{**p_base.__dict__,
        "costs":    {**p_base.costs,    "RoW": p_base.costs["RoW"]    * m_cost},
        "c_tilde":  {**p_base.c_tilde,  "RoW": p_base.c_tilde["RoW"]  * m_cost},
        "damages":  {**p_base.damages,  "RoW": p_base.damages["RoW"]  * m_dam},
        "pressure": {**p_base.pressure, "RoW": p_base.pressure["RoW"] * m_press},
        "discount": {**p_base.discount, "RoW": d_row},
    })


def run_mc(params, n_mc, seed=SEED):
    V, sig, _, _ = solve_model(params)
    W_paths, adopt = monte_carlo(V, sig, params, n_runs=n_mc, seed=seed)
    success = float(np.mean(W_paths[:, -1] >= params.theta))
    # mean timing conditional on success
    succ_mask = W_paths[:, -1] >= params.theta
    if succ_mask.any():
        # first period t where W_paths[run, t] >= theta
        reached = W_paths >= params.theta
        first = np.argmax(reached, axis=1).astype(float)
        first[~reached.any(axis=1)] = np.nan
        mean_timing = float(np.nanmean(first[succ_mask]))
    else:
        mean_timing = float("nan")
    # period-1 adoption probability per bloc
    p1 = {bloc: float(np.mean(adopt[:, i] == 1))
          for i, bloc in enumerate(PLAYERS)}
    return success, mean_timing, p1


def make_sweep_defs(params):
    """Channel sweeps operating on a GameParams. x_baseline is taken from
    params where possible; Global cost uses multiplier baseline 1.0."""
    d_us = params.discount["US"]
    d_cn = params.discount["CN"]
    d_eu = params.discount["EU"]
    joint_base = 0.5 * (d_us + d_cn)

    defs = [
        ("δ_US=δ_CN", np.linspace(0.50, 0.95, N_SWEEP_PTS),
            lambda p, v: GameParams(**{**p.__dict__,
                "discount": {**p.discount, "US": v, "CN": v}}),
            "inc", joint_base),
        ("δ_US", np.linspace(0.50, 0.95, N_SWEEP_PTS),
            lambda p, v: GameParams(**{**p.__dict__,
                "discount": {**p.discount, "US": v}}),
            "inc", d_us),
        ("δ_CN", np.linspace(0.50, 0.95, N_SWEEP_PTS),
            lambda p, v: GameParams(**{**p.__dict__,
                "discount": {**p.discount, "CN": v}}),
            "inc", d_cn),
        ("δ_EU", np.linspace(0.70, 0.95, N_SWEEP_PTS),
            lambda p, v: GameParams(**{**p.__dict__,
                "discount": {**p.discount, "EU": v}}),
            "inc", d_eu),
        ("Global cost", np.linspace(0.30, 1.30, N_SWEEP_PTS),
            lambda p, v: GameParams(**{**p.__dict__,
                "costs": {k: c * v for k, c in p.costs.items()}}),
            "dec", 1.0),
        ("θ", np.linspace(0.50, 0.95, N_SWEEP_PTS),
            lambda p, v: GameParams(**{**p.__dict__, "theta": v}),
            "dec", THETA),
    ]
    return defs


def sweep_crossings(params):
    """90% crossings (% shift from each channel's own baseline) for one archetype."""
    out = {}
    for channel, xs, modifier, direction, x_base in make_sweep_defs(params):
        rates = []
        for x in xs:
            p_mod = modifier(params, float(x))
            try:
                s, _, _ = run_mc(p_mod, SWEEP_MC, seed=0)
                rates.append(s)
            except Exception:
                rates.append(np.nan)
        rates = np.array(rates)
        c90 = crossing_value(xs, rates, 0.90, direction)
        c90_pct = (None if c90 is None
                   else round((c90 - x_base) / x_base * 100, 2))
        out[channel] = c90_pct
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
    params_base = build_params(
        raw, weights,
        ac=b["ac"], ad=b["ad"], a_spill=b["a_spill"], ab=b["ab"], lam=b["lam"],
    )
    # Enforce the baseline RoW discount from the dissertation's Table 1.
    params_base = GameParams(**{**params_base.__dict__,
        "discount": {**params_base.discount, "RoW": 0.70}})
    w_row_fixed = weights["RoW"]
    print(f"Fixed RoW coalition weight w_RoW = {w_row_fixed:.4f}")
    print(f"Grid: {N_SWEEP_PTS} points × {SWEEP_MC} MC per sweep point\n")

    channel_order = ["δ_US=δ_CN", "Global cost", "δ_US", "δ_CN", "δ_EU", "θ"]
    rows = []

    for key, label, m_cost, m_dam, m_press, d_row in ARCHETYPES:
        print(f"── {label}")
        p_arch = apply_archetype(params_base, m_cost, m_dam, m_press, d_row)
        # sanity-check weight preservation
        assert abs(p_arch.weights["RoW"] - w_row_fixed) < 1e-12

        success, timing, p1 = run_mc(p_arch, BASE_MC, seed=SEED)
        print(f"   success={success:.3f}  mean timing={timing:.2f}  "
              f"p1: EU={p1['EU']:.3f} CN={p1['CN']:.3f} "
              f"US={p1['US']:.3f} RoW={p1['RoW']:.3f}")

        cr = sweep_crossings(p_arch)
        row = {
            "archetype":  key,
            "label":      label,
            "m_cost":     m_cost,
            "m_damage":   m_dam,
            "m_pressure": m_press,
            "delta_RoW":  d_row,
            "success":    round(success, 3),
            "mean_timing": round(timing, 2),
            "p1_EU":      round(p1["EU"], 3),
            "p1_CN":      round(p1["CN"], 3),
            "p1_US":      round(p1["US"], 3),
            "p1_RoW":     round(p1["RoW"], 3),
        }
        for ch in channel_order:
            row[f"{ch}_90"] = cr[ch]
            s = f"{cr[ch]:+.2f}%" if cr[ch] is not None else "absent"
            print(f"     {ch:<14} 90% = {s}")
        rows.append(row)
        print()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    # ── LaTeX table ───────────────────────────────────────────────────────
    def fmt(v):
        return "absent" if v is None or pd.isna(v) else f"{v:+.1f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Robustness of the main findings to alternative RoW payoff "
        r"compositions. The RoW coalition weight $w_{\text{RoW}}$ is held fixed "
        r"at its baseline value; all non-RoW parameters and the SMM-calibrated "
        r"cardinal vector are unchanged. Only RoW's cost, damage, pressure, and "
        r"discount entries are rescaled to reflect each archetype. Upper panel: "
        r"baseline diagnostics per archetype (coordination success rate, mean "
        r"timing conditional on success, and period-one adoption probabilities). "
        r"Lower panel: 90\% sweep crossings (percentage shift from each "
        r"archetype's own baseline required to reach 90\% success) for the six "
        r"headline channels. ``absent'' indicates the 90\% threshold is not "
        f"reached within the sweep range. Grid: {N_SWEEP_PTS} points, "
        f"{SWEEP_MC} Monte Carlo draws per sweep point; {BASE_MC} draws for "
        r"the baseline diagnostics."
        r"}",
        r"\label{tab:row_payoff_robustness}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Archetype & $P_0$ & $\bar{t}$ & $p_1^{EU}$ & $p_1^{CN}$ & "
        r"$p_1^{US}$ & $p_1^{RoW}$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['success']:.2f} & {r['mean_timing']:.2f} & "
            f"{r['p1_EU']:.2f} & {r['p1_CN']:.2f} & "
            f"{r['p1_US']:.2f} & {r['p1_RoW']:.2f} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{90\% sweep crossings (\% shift from own baseline)}} \\",
        r"\midrule",
        r"Archetype & $\delta_{US}{=}\delta_{CN}$ & Global cost & "
        r"$\delta_{US}$ & $\delta_{CN}$ & $\delta_{EU}$ & $\theta$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & "
            f"{fmt(r['δ_US=δ_CN_90'])} & {fmt(r['Global cost_90'])} & "
            f"{fmt(r['δ_US_90'])} & {fmt(r['δ_CN_90'])} & "
            f"{fmt(r['δ_EU_90'])} & {fmt(r['θ_90'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {OUT_TEX}")
