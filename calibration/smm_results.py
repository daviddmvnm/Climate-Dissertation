"""
smm_results.py
──────────────
Produces all SMM output tables and re-runs key analysis under SMM parameters.
Run as:  python smm_results.py > smm_output.txt
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS, solve_model, monte_carlo as _mc
from run_analysis import (
    build_bloc_data, build_params, run_mc,
    BASE_ALPHAS, crossing_value,
)
from smm_calibration import (
    compute_moments, run_smm,
    MOMENTS_DATA, MOMENT_NAMES, MOMENT_WEIGHTS, PARAM_NAMES,
    STARTING_POINTS, SCENARIO_LABELS,
    build_smm_params, N_MC_OPT, N_MC_FINAL, MC_SEED,
)

SEP  = "=" * 65
SEP2 = "-" * 65


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── Setup ──────────────────────────────────────────────────────────────────
print("Loading data and running SMM...", flush=True)
bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
raw       = bloc_data.set_index("bloc")

weights = {}
for bloc in PLAYERS:
    weights[bloc] = (0.5 * raw.loc[bloc, "emission_share"]
                   + 0.5 * raw.loc[bloc, "gdp_share"])
total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}

THETA_ORIG = [BASE_ALPHAS["ac"], BASE_ALPHAS["ad"],
              BASE_ALPHAS["ap"], BASE_ALPHAS["ab"]]

# ── Load saved SMM result ──────────────────────────────────────────────────
print("Loading saved SMM calibration...", flush=True)
theta_smm = np.load(os.path.join(ROOT_DIR, "results", "smm_best_theta.npy"))

# ── Final evaluation at n=1000 ─────────────────────────────────────────────
print(f"Final evaluation at n_mc={N_MC_FINAL}...", flush=True)
scen_a_pre = compute_moments(THETA_ORIG, raw, weights, n_mc=N_MC_FINAL)
m_post     = compute_moments(theta_smm, raw, weights, n_mc=N_MC_FINAL)

# ══════════════════════════════════════════════════════════════════════
section("TABLE 1 — MOMENT FIT")
print(f"\n  {'Moment':<30} {'Target':>8} {'Pre-SMM':>12} {'Post-SMM':>10}")
print(f"  {SEP2}")
for name, target, pre, post in zip(MOMENT_NAMES, MOMENTS_DATA, scen_a_pre, m_post):
    print(f"  {name:<30} {target:>8.3f} {pre:>12.3f} {post:>10.3f}")
d_pre = MOMENTS_DATA - scen_a_pre
print(f"\n  Objective at pre-SMM (original): {float(MOMENT_WEIGHTS @ (d_pre**2)):.4f}")
print(f"  Objective at post-SMM         : {best.fun:.6f}")

# ══════════════════════════════════════════════════════════════════════
section("TABLE 2 — SMM ALPHA ESTIMATES vs ORIGINAL SCENARIOS")

scen_vals = {label: x0 for label, x0 in zip(SCENARIO_LABELS, STARTING_POINTS)}
print(f"\n  {'Parameter':<10} {'Scen A':>10} {'Scen B':>10} {'Scen C':>10} {'SMM Est.':>12}")
print(f"  {SEP2}")
for i, name in enumerate(PARAM_NAMES):
    print(f"  {name:<10} {scen_vals['A'][i]:>10.4f} {scen_vals['B'][i]:>10.4f} "
          f"{scen_vals['C'][i]:>10.4f} {theta_smm[i]:>12.4f}")

print(f"\n  Convergence check (all 3 scenario starts):")
for label, res in zip(SCENARIO_LABELS, all_results):
    vals = " ".join(f"{v:.4f}" for v in res.x)
    print(f"    Scen {label}: obj={res.fun:.6f}  θ=[{vals}]  "
          f"{'converged' if res.success else 'FAILED'}")

solutions = np.array([r.x for r in all_results])
spread = solutions.std(axis=0)
if spread.max() > 0.1:
    print(f"\n  ⚠  Starting points did not converge to same solution.")
    print(f"     Spread (std): {dict(zip(PARAM_NAMES, spread.round(4)))}")
    print(f"     Report manifold, not single point.")
else:
    print(f"\n  ✓ All 3 starts converged (spread < 0.1). Unique minimum.")

# ══════════════════════════════════════════════════════════════════════
section("TABLE 3 — MODEL DIAGNOSTICS UNDER SMM")

params_smm  = build_smm_params(*theta_smm, raw, weights)
params_orig = build_smm_params(*THETA_ORIG, raw, weights)

G0 = (0, 0, 0, 0)
V_orig, sig_orig, _, _ = solve_model(params_orig)
V_smm,  sig_smm,  _, _ = solve_model(params_smm)

W_orig_paths, adopt_orig = _mc(V_orig, sig_orig, params_orig, n_runs=N_MC_FINAL, seed=MC_SEED)
W_smm_paths,  adopt_smm  = _mc(V_smm,  sig_smm,  params_smm,  n_runs=N_MC_FINAL, seed=MC_SEED)

eu_idx = list(PLAYERS).index("EU")

eu_leads_orig = float((adopt_orig[:, eu_idx] == 1).mean())
eu_leads_smm  = float((adopt_smm[:,  eu_idx] == 1).mean())
early_orig    = float((W_orig_paths[:, 1:3] >= params_orig.theta).any(axis=1).mean())
early_smm     = float((W_smm_paths[:,  1:3] >= params_smm.theta).any(axis=1).mean())
succ_orig     = float((W_orig_paths[:, -1] >= params_orig.theta).mean())
succ_smm      = float((W_smm_paths[:,  -1] >= params_smm.theta).mean())
mW_orig       = float(W_orig_paths[:, -1].mean())
mW_smm        = float(W_smm_paths[:,  -1].mean())

_, _, _, _, _, _, mct_orig, _ = run_mc(params_orig, N_MC_FINAL, seed=MC_SEED)
_, _, _, _, _, _, mct_smm,  _ = run_mc(params_smm,  N_MC_FINAL, seed=MC_SEED)

print(f"\n  {'Metric':<34} {'Original (A)':>13} {'SMM':>10}")
print(f"  {SEP2}")
print(f"  {'P(EU leads, t=1)':<34} {eu_leads_orig:>13.3f} {eu_leads_smm:>10.3f}")
print(f"  {'P(global coord., t≤2)':<34} {early_orig:>13.3f} {early_smm:>10.3f}")
print(f"  {'Coordination success rate (final)':<34} {succ_orig:>13.3f} {succ_smm:>10.3f}")
print(f"  {'Mean final W':<34} {mW_orig:>13.4f} {mW_smm:>10.4f}")
ct_o = f"{mct_orig:.2f}" if not np.isnan(mct_orig) else "n/a"
ct_s = f"{mct_smm:.2f}"  if not np.isnan(mct_smm)  else "n/a"
print(f"  {'Mean coord. period':<34} {ct_o:>13} {ct_s:>10}")

print(f"\n  Period-1 equilibrium sigmas:")
print(f"  {'Bloc':<8} {'σ (Original A)':>16} {'σ (SMM)':>10}")
print(f"  {'-'*36}")
for i, p in enumerate(PLAYERS):
    print(f"  {p:<8} {sig_orig[1][G0][i]:>16.4f} {sig_smm[1][G0][i]:>10.4f}")

# ══════════════════════════════════════════════════════════════════════
section("RE-RUN: δ_CN MARGINAL SWEEP UNDER SMM PARAMS (TIER-1 CHECK)")

xs = np.linspace(0.50, 0.95, 12)
rates_smm, rates_orig = [], []

import copy

for v in xs:
    # Use copy.deepcopy to ensure the dictionaries (lam, weights, etc.) are truly independent
    p_s = copy.deepcopy(params_smm)
    p_s.discount["CN"] = v
    
    p_o = copy.deepcopy(params_orig)
    p_o.discount["CN"] = v
    
    try:
        _, _, _, _, s_s, _, _, _ = run_mc(p_s, 150, seed=0)
        _, _, _, _, s_o, _, _, _ = run_mc(p_o, 150, seed=0)
    except Exception:
        s_s = s_o = np.nan
    rates_smm.append(s_s)
    rates_orig.append(s_o)

rates_smm  = np.array(rates_smm)
rates_orig = np.array(rates_orig)
cross_smm  = crossing_value(xs, rates_smm,  0.70, "inc")
cross_orig = crossing_value(xs, rates_orig, 0.70, "inc")

print(f"\n  {'δ_CN':<8} {'Orig success':>14} {'SMM success':>14}")
print(f"  {'-'*38}")
for x, ro, rs in zip(xs, rates_orig, rates_smm):
    print(f"  {x:<8.4f} {ro:>14.3f} {rs:>14.3f}")

print(f"\n  70% frontier crossing:")
print(f"    Original : δ_CN = {cross_orig}")
print(f"    SMM      : δ_CN = {cross_smm}")
if cross_smm is not None and cross_orig is not None:
    shift = abs(cross_smm - cross_orig)
    print(f"    Shift    : {shift:.4f}  "
          f"({'STABLE' if shift < 0.05 else 'SHIFTED — check Tier classification'})")
elif cross_smm is None:
    print("    WARNING: 70% crossing absent under SMM params")

print(f"\nDone.")
