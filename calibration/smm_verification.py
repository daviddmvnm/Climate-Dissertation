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
    weights[bloc] = 0.3 * raw.loc[bloc, "emission_share"] \
                  + 0.7 * raw.loc[bloc, "gdp_share"]
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
        "EU > RoW > CN > US",
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
# JACOBIAN — computed once, used by both Check 4 and Check 5.
# Central differences at THETA_BEST for better accuracy than forward.
# ══════════════════════════════════════════════════════════════════
print(f"\n  Computing moment Jacobian G = ∂m/∂θ (central differences)...", flush=True)
h = 1e-3
n_p = len(THETA_BEST)
n_m = len(MOMENTS_DATA)
G = np.zeros((n_m, n_p))
for j in range(n_p):
    tp = np.array(THETA_BEST, dtype=float); tp[j] += h
    tn = np.array(THETA_BEST, dtype=float); tn[j] -= h
    m_up = compute_moments(tp, raw, weights)
    m_dn = compute_moments(tn, raw, weights)
    G[:, j] = (m_up - m_dn) / (2 * h)

# ══════════════════════════════════════════════════════════════════
# CHECK 4 — MOMENT SENSITIVITY (ANDREWS–GENTZKOW–SHAPIRO)
# Andrews, Gentzkow, Shapiro (AER 2017) derive a closed-form
# sensitivity matrix Λ = (G'WG)^{-1} G'W that approximates
#     Δθ ≈ Λ Δm
# without re-optimising under perturbed targets. Dividing by
# baseline scales gives dimensionless elasticities
#     E_{jk} = Λ_{jk} · m_k / θ_j
# which are the fractional change in parameter j per fractional
# change in moment k. This sidesteps the scale-blindness and
# multi-basin artefacts of uniform %-scaling re-estimation tests.
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  CHECK 4 — MOMENT SENSITIVITY (Andrews–Gentzkow–Shapiro)\n{SEP}")

W = np.diag(MOMENT_WEIGHTS)
try:
    Lam = np.linalg.solve(G.T @ W @ G, G.T @ W)   # Λ = (G'WG)^{-1} G'W
except np.linalg.LinAlgError:
    Lam = None

if Lam is None:
    print("  G'WG is singular — parameters not locally identified.")
else:
    m_vec = np.asarray(MOMENTS_DATA, dtype=float)
    t_vec = np.asarray(THETA_BEST,  dtype=float)
    E = (Lam * m_vec) / t_vec[:, None]            # E[j,k] = Λ_{jk} m_k / θ_j

    mom_short = [(n[:12] + "…") if len(n) > 12 else n for n in MOMENT_NAMES]
    col_w = 12
    hdr = f"  {'Param':<10} " + " ".join(f"{m:>{col_w}}" for m in mom_short)
    print("\n  Elasticity matrix  (Δθ/θ per 1% Δm, via AGS Λ)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for j, name in enumerate(PARAM_NAMES):
        cells = " ".join(f"{E[j, k]:>{col_w}.4f}" for k in range(n_m))
        print(f"  {name:<10} {cells}")

    abs_E = np.abs(E)
    max_abs = float(abs_E.max())
    j_max, k_max = np.unravel_index(np.argmax(abs_E), E.shape)
    row_l1 = abs_E.sum(axis=1)
    print("  " + "-" * (len(hdr) - 2))
    print(f"\n  Row Σ|E| (total moment leverage on each parameter):")
    for j, name in enumerate(PARAM_NAMES):
        print(f"    {name:<10} {row_l1[j]:.4f}")

    verdict = "PASS" if max_abs < 2.0 else "CAUTION"
    print(f"\n  Max |elasticity|      : {max_abs:.4f}")
    print(f"  Most sensitive pair   : {PARAM_NAMES[j_max]} ← {MOMENT_NAMES[k_max]}")
    print(f"  Target                : max |E| < 2.0 — {verdict}")
    print("  Interpretation: a 1% shift in m_k induces ~|E_{jk}|% shift in θ_j,")
    print("                  to first order and holding all other moments fixed.")

# ══════════════════════════════════════════════════════════════════
# CHECK 5 — JACOBIAN IDENTIFICATION MATRIX
# Reuses G from above. Raw moment sensitivities + condition number
# of J provide a second, complementary view of local identification.
# ══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  CHECK 5 — JACOBIAN IDENTIFICATION MATRIX\n{SEP}")
print("\n  Sensitivity Matrix (∂ moment / ∂ param):")
header = " " * 32 + "  ".join(f"{p:>8}" for p in PARAM_NAMES)
print(header)
print("-" * len(header))
for i, m_name in enumerate(MOMENT_NAMES):
    row_vals = "  ".join(f"{G[i, j]:>8.3f}" for j in range(n_p))
    short_name = (m_name[:30] + '..') if len(m_name) > 30 else m_name.ljust(32)
    print(f"  {short_name} {row_vals}")

cond_num = np.linalg.cond(G)
print(f"\n  Jacobian Condition Number: {cond_num:.2f}")
print(f"  Verdict : {'WELL-IDENTIFIED' if cond_num < 500 else 'WEAKLY IDENTIFIED (Check for collinearity)'}")

print(f"\n{SEP}\n  VERIFICATION COMPLETE\n{SEP}")