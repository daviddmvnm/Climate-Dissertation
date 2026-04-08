"""
smm_verification.py
───────────────────
Three post-SMM sanity checks. Run as:
    python smm_verification.py
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

from climate_game import PLAYERS, solve_model, monte_carlo
from run_analysis import build_bloc_data
from smm_calibration import (
    build_smm_params, smm_objective,
    PARAM_NAMES, NM_OPTIONS,
)
from smm_calibration import (
    build_smm_params, smm_objective, compute_moments,
    PARAM_NAMES, NM_OPTIONS, 
    MOMENTS_DATA, MOMENT_NAMES, MOMENT_WEIGHTS, BOUNDS # Added these imports
)

# ── Best SMM estimate — loaded from calibration output ────────────
THETA_BEST = list(np.load(os.path.join(ROOT_DIR, "results", "smm_best_theta.npy")))
N_MC       = 1000
SEEDS      = [42, 123, 999, 2024, 7]

SEP  = "=" * 65
SEP2 = "-" * 52

# ── Setup ──────────────────────────────────────────────────────────
print("Loading data...", flush=True)
bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
raw = bloc_data.set_index("bloc")

weights = {}
for bloc in PLAYERS:
    weights[bloc] = 0.5 * raw.loc[bloc, "emission_share"] \
                  + 0.5 * raw.loc[bloc, "gdp_share"]
total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}

params = build_smm_params(*THETA_BEST, raw, weights)


# ══════════════════════════════════════════════════════════════════
# CHECK 1 — Optimiser stability
# Restart from the best point. A genuine local minimum stays put.
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  CHECK 1 — OPTIMISER STABILITY\n{SEP}")
print("  Restarting from best SMM point...", flush=True)

obj_start = smm_objective(THETA_BEST, raw, weights)
res       = minimize(smm_objective, THETA_BEST, args=(raw, weights),
                     method="Nelder-Mead", options=NM_OPTIONS)
shift     = np.abs(np.array(res.x) - np.array(THETA_BEST))

print(f"\n  Start  : {[round(v, 4) for v in THETA_BEST]}  obj={obj_start:.6f}")
print(f"  Result : {[round(v, 4) for v in res.x]}  obj={res.fun:.6f}")
print(f"\n  {'Param':<8} {'Start':>10} {'End':>10} {'Shift':>10}")
print(f"  {'-'*40}")
for name, s, e in zip(PARAM_NAMES, THETA_BEST, res.x):
    print(f"  {name:<8} {s:>10.4f} {e:>10.4f} {abs(e-s):>10.6f}")

print(f"\n  Max parameter shift : {shift.max():.6f}")
print(f"  Obj improvement     : {obj_start - res.fun:.6f}")
print(f"  Optimiser status    : {'converged' if res.success else 'FAILED'}")
stable = shift.max() < 0.01
print(f"\n  Verdict : {'STABLE — genuine local minimum' if stable else 'DRIFTED — flat landscape, report convergence region not point'}")


# ══════════════════════════════════════════════════════════════════
# CHECK 2 — Seed stability
# Run Monte Carlo at n=1000 across 5 seeds. Stable baseline should
# show std(success_rate) < 0.02.
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  CHECK 2 — SEED STABILITY  (n={N_MC}, {len(SEEDS)} seeds)\n{SEP}")
print("  Running Monte Carlo...", flush=True)

V, sigma, _, _ = solve_model(params)

succs, means = [], []
print(f"\n  {'Seed':>6} {'Success':>10} {'Mean W':>10} {'Median W':>10} {'Below θ':>10}")
print(f"  {SEP2}")
for seed in SEEDS:
    W_paths, _ = monte_carlo(V, sigma, params, n_runs=N_MC, seed=seed)
    fw   = W_paths[:, -1]
    succ = float((fw >= params.theta).mean())
    mw   = float(fw.mean())
    med  = float(np.median(fw))
    lo   = float((fw < params.theta).mean())
    succs.append(succ)
    means.append(mw)
    print(f"  {seed:>6} {succ:>10.3f} {mw:>10.4f} {med:>10.4f} {lo:>10.3f}")

print(f"\n  Success rate — mean={np.mean(succs):.3f}  "
      f"std={np.std(succs):.4f}  "
      f"range=[{min(succs):.3f}, {max(succs):.3f}]")
seed_stable = np.std(succs) < 0.02
print(f"\n  Verdict : {'STABLE — safe to run parameter sweeps' if seed_stable else 'SEED SENSITIVE — increase n_mc before sweeps'}")


# ══════════════════════════════════════════════════════════════════
# CHECK 3 — Ordinal ranking preservation
# SMM should not flip the data-implied ordinal structure.
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  CHECK 3 — ORDINAL RANKING PRESERVATION\n{SEP}")

checks = [
    (
        "Costs (transition difficulty)",
        "RoW > CN > US > EU",
        ["RoW", "CN", "US", "EU"],
        {p: params.costs[p] for p in PLAYERS},
    ),
    (
        "Damages (climate vulnerability)",
        "RoW > CN > EU > US",
        ["RoW", "CN", "EU", "US"],
        {p: params.damages[p] for p in PLAYERS},
    ),
    (
        "Discount factors (patience)",
        "EU > CN > US > RoW",
        ["EU", "CN", "US", "RoW"],
        {p: params.discount[p] for p in PLAYERS},
    ),
    (
        "Pressure (domestic salience)",
        "EU > RoW > CN > US",   # EU strong green politics; RoW high exposure
        ["EU", "RoW", "CN", "US"],
        {p: params.pressure[p] for p in PLAYERS},
    ),
    (
        "Influence weights",
        "CN > US > EU > RoW",
        ["CN", "US", "EU", "RoW"],
        {p: params.weights[p] for p in PLAYERS},
    ),
]

all_ok = True
for label, expected_str, expected_order, values in checks:
    actual_order = sorted(PLAYERS, key=lambda p: values[p], reverse=True)
    ok = actual_order == expected_order
    if not ok:
        all_ok = False
    vals_str = "  ".join(f"{p}={values[p]:.4f}" for p in actual_order)
    print(f"\n  {label}")
    print(f"  Expected : {expected_str}")
    print(f"  Actual   : {' > '.join(actual_order)}")
    print(f"  Values   : {vals_str}")
    print(f"  {'OK' if ok else 'VIOLATED'}")

print(f"\n  Overall : {'ALL RANKINGS PRESERVED' if all_ok else 'ONE OR MORE RANKINGS VIOLATED — review above'}")

print(f"\nDone.")

# ══════════════════════════════════════════════════════════════════
# CHECK 4 — PARAMETER ELASTICITY (STABILITY TEST)
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  CHECK 4 — PARAMETER ELASTICITY (±10% STABILITY TEST)\n{SEP}")
EPSILON_PERTURB = 0.05
print(f"  Perturbing all target moments by ±{EPSILON_PERTURB*100:.0f}%...", flush=True)

def stressed_objective(theta, raw, weights, targets):
    from smm_calibration import BOUNDS, MOMENT_WEIGHTS
    for v, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < v < hi): return 1e6
    m_model = compute_moments(theta, raw, weights, n_mc=250)
    diff = targets - m_model
    return float(MOMENT_WEIGHTS @ (diff ** 2))

stressed_pos = MOMENTS_DATA * (1 + EPSILON_PERTURB)
stressed_neg = MOMENTS_DATA * (1 - EPSILON_PERTURB)

print(f"  Running +{EPSILON_PERTURB*100:.0f}% optimisation...", flush=True)
res_pos = minimize(stressed_objective, THETA_BEST,
                   args=(raw, weights, stressed_pos),
                   method="Nelder-Mead", options=NM_OPTIONS)

print(f"  Running -{EPSILON_PERTURB*100:.0f}% optimisation...", flush=True)
res_neg = minimize(stressed_objective, THETA_BEST,
                   args=(raw, weights, stressed_neg),
                   method="Nelder-Mead", options=NM_OPTIONS)

pos_label = f"+{EPSILON_PERTURB*100:.0f}%"
neg_label = f"-{EPSILON_PERTURB*100:.0f}%"
print(f"\n  {'Param':<8} {'Baseline':>10} {pos_label:>10} {neg_label:>10} {'Mean |Elast|':>14}")
print(f"  {'-'*58}")

mean_elasticities = []
for name, base, pos_val, neg_val in zip(PARAM_NAMES, THETA_BEST, res_pos.x, res_neg.x):
    el_pos = abs((pos_val - base) / base) / EPSILON_PERTURB
    el_neg = abs((neg_val - base) / base) / EPSILON_PERTURB
    mean_el = (el_pos + el_neg) / 2
    mean_elasticities.append(mean_el)
    print(f"  {name:<8} {base:>10.4f} {pos_val:>10.4f} {neg_val:>10.4f} {mean_el:>14.4f}")

overall_mean = np.mean(mean_elasticities)
verdict = "PASS" if overall_mean < 2.0 else "FAIL"
print(f"\n  Mean |Elasticity| : {overall_mean:.4f}  (Target < 2.0) — {verdict}")

# ══════════════════════════════════════════════════════════════════
# CHECK 5 — JACOBIAN IDENTIFICATION (THE PRO MOVE)
# Calculates local sensitivity of moments to parameters.
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  CHECK 5 — JACOBIAN IDENTIFICATION MATRIX\n{SEP}")
print("  Calculating numerical derivatives (Finite Differences)...", flush=True)

h = 1e-3 # Step size for derivative
J = np.zeros((len(MOMENTS_DATA), len(THETA_BEST)))
m0 = compute_moments(THETA_BEST, raw, weights, n_mc=500)

for j in range(len(THETA_BEST)):
    theta_plus = np.copy(THETA_BEST)
    theta_plus[j] += h
    m_plus = compute_moments(theta_plus, raw, weights, n_mc=500)
    J[:, j] = (m_plus - m0) / h

print(f"\n  Sensitivity Matrix (d_Moment / d_Param):")
header = " " * 32 + "  ".join(f"{p:>8}" for p in PARAM_NAMES)
print(header)
print("-" * len(header))

for i, m_name in enumerate(MOMENT_NAMES):
    row_vals = "  ".join(f"{J[i, j]:>8.3f}" for j in range(len(THETA_BEST)))
    short_name = (m_name[:30] + '..') if len(m_name) > 30 else m_name.ljust(32)
    print(f"  {short_name} {row_vals}")

cond_num = np.linalg.cond(J)
print(f"\n  Jacobian Condition Number: {cond_num:.2f}")
print(f"  Verdict : {'WELL-IDENTIFIED' if cond_num < 500 else 'WEAKLY IDENTIFIED (Check for collinearity)'}")

print(f"\n{SEP}\n  VERIFICATION COMPLETE\n{SEP}")