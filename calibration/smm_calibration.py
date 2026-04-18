"""
smm_calibration.py
──────────────────
Simulated Method of Moments calibration for the Climate Cooperation Markov Game.

Canonical moment set (v2 writeup, §3.2.2):
  M1: EU/US period-1 adoption gap  σ_EU,1 − σ_US,1         (Target: 0.25)
  M2: US period-1 adoption prob    σ_US,1                  (Target: 0.10)
  M3: China period-1 adoption prob σ_CN,1                  (Target: 0.10)
  M4: China period-2 / period-1 ratio                      (Target: 2.00)
      E[σ_CN,2 | CN not adopted at t=1, avg over others'
      period-1 actions] / σ_CN,1
  M5: US expected period-2 adoption prob                   (Target: 0.15)
      E[σ_US,2 | US not adopted at t=1, avg over others'
      period-1 actions]
  M6: Mean coordination timing | success (periods)         (Target: 5.00)

Identification:
  - W_0 = 0 at t=1, so spillover is inactive: M1-M3 identify α_c, α_d, α_b.
  - By t=2 the coalition has begun forming, so M4 and M5 carry
    identifying information for α_spill (composite carrot/stick scaling).
  - M6 anchors aggregate timing.

Composite spillover split is held fixed at φ = 0.5 during estimation
(Appendix B sweeps φ ∈ {0, 0.25, 0.5, 0.75, 1.0} and shows parameter
stability in the pressure-inclusive region [0.5, 1.0]).
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
from itertools import product as iproduct
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS, solve_model, compute_W
from run_analysis import build_bloc_data, build_params, DISCOUNT, ETA, KAPPA

# ── Fixed parameters ──────────────────────────────────────────────────────
LAMBDA_FIXED   = 1.54
ETA_FIXED      = ETA
KAPPA_FIXED    = KAPPA
DISCOUNT_FIXED = {"US": 0.75, "EU": 0.85, "CN": 0.80, "RoW": 0.70}
PHI_FIXED      = 0.5

# ── Target moments ────────────────────────────────────────────────────────
MOMENTS_DATA = np.array([
    0.25,   # M1: EU − US period-1 adoption gap
    0.10,   # M2: US period-1 adoption probability
    0.10,   # M3: CN period-1 adoption probability
    2.00,   # M4: CN period-2 / period-1 ratio
    0.15,   # M5: US expected period-2 adoption probability
    5.00,   # M6: Mean coordination timing | success (periods)
])

MOMENT_NAMES = [
    "EU − US adoption gap",
    "US period-1 adoption prob",
    "CN period-1 adoption prob",
    "CN period-2/period-1 ratio",
    "US expected period-2 prob",
    "Mean coord timing | success",
]

# Scaling: keep M1-M3, M5 on unit weight; downweight ratio M4 and timing M6
# so a ~10% deviation on each target contributes comparably to the objective.
MOMENT_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.04, 1.0, 0.04])

# ── Parameter bounds for [α_c, α_d, α_spill, α_b] ────────────────────────
BOUNDS = [
    (0.1, 10.0),   # α_c     transition cost scaling
    (0.1, 10.0),   # α_d     climate damage scaling
    (0.1, 10.0),   # α_spill composite spillover scaling (carrot/stick)
    (0.1, 10.0),   # α_b     coordination benefit scaling
]
PARAM_NAMES = ["α_c", "α_d", "α_spill", "α_b"]

# MC draws (retained for any MC-based diagnostic; all SMM moments are analytical)
N_MC_OPT   = 500
N_MC_FINAL = 1000
MC_SEED    = 42


def build_smm_params(ac, ad, a_spill, ab, raw, weights, phi=PHI_FIXED):
    """Build GameParams from cardinal vector; η, κ, λ, δ, φ held fixed."""
    params = build_params(raw, weights, ac, ad, a_spill, ab,
                          discount=DISCOUNT_FIXED,
                          eta=ETA_FIXED, kappa=KAPPA_FIXED,
                          phi=phi)
    lam_dict = {p: LAMBDA_FIXED for p in PLAYERS}
    return GameParams(**{**params.__dict__, "lam": lam_dict})


def _period2_expected_adopt(sigma, active_idx, G0):
    """E[σ_{active_idx, 2} | active_idx didn't adopt at t=1], averaging over
    the period-1 action profile of the other three blocs (independent Bernoulli
    under the QRE equilibrium at state G0)."""
    n = len(PLAYERS)
    others = [i for i in range(n) if i != active_idx]
    expected = 0.0
    for actions in iproduct([0, 1], repeat=len(others)):
        prob = 1.0
        G2 = list(G0)
        for k, j in enumerate(others):
            s_j = float(sigma[1][G0][j])
            if actions[k] == 1:
                prob *= s_j
                G2[j] = 1
            else:
                prob *= (1 - s_j)
        G2[active_idx] = 0  # by construction
        expected += prob * float(sigma[2][tuple(G2)][active_idx])
    return expected


def compute_moments(theta, raw, weights, n_mc=N_MC_OPT, seed=MC_SEED, phi=PHI_FIXED):
    """Compute M1–M6 analytically from the solved equilibrium."""
    ac, ad, a_spill, ab = theta
    params = build_smm_params(ac, ad, a_spill, ab, raw, weights, phi=phi)

    try:
        V, sigma, _, _ = solve_model(params)
    except Exception:
        return np.array([1e6] * 6)

    G0  = (0, 0, 0, 0)
    idx = {p: i for i, p in enumerate(PLAYERS)}

    s_EU = float(sigma[1][G0][idx["EU"]])
    s_US = float(sigma[1][G0][idx["US"]])
    s_CN = float(sigma[1][G0][idx["CN"]])

    m1 = s_EU - s_US
    m2 = s_US
    m3 = s_CN

    # M4: CN period-2 / period-1 adoption ratio
    s_CN_t2 = _period2_expected_adopt(sigma, idx["CN"], G0)
    m4 = s_CN_t2 / s_CN if s_CN > 1e-9 else 1e6

    # M5: US expected period-2 adoption probability
    m5 = _period2_expected_adopt(sigma, idx["US"], G0)

    # M6: first-passage timing — forward DP with absorption at theta
    n = len(PLAYERS)
    dist = {G0: 1.0}
    success_prob = 0.0
    timing_sum   = 0.0
    T_horizon = params.T
    for t in range(1, T_horizon + 1):
        new_dist = {}
        for G, gprob in dist.items():
            W = compute_W(G, params)
            if W >= params.theta:
                success_prob += gprob
                timing_sum   += t * gprob
                continue
            active = [i for i in range(n) if G[i] == 0]
            if not active:
                new_dist[G] = new_dist.get(G, 0.0) + gprob
                continue
            for actions in iproduct([0, 1], repeat=len(active)):
                p_action = 1.0
                G_next = list(G)
                for k, i in enumerate(active):
                    s = float(sigma[t][G][i])
                    if actions[k] == 1:
                        p_action *= s
                        G_next[i] = 1
                    else:
                        p_action *= (1 - s)
                G_next = tuple(G_next)
                new_dist[G_next] = new_dist.get(G_next, 0.0) + gprob * p_action
        dist = new_dist
    for G, prob in dist.items():
        if compute_W(G, params) >= params.theta:
            success_prob += prob
            timing_sum   += T_horizon * prob
    m6 = timing_sum / success_prob if success_prob > 1e-9 else 1e6

    return np.array([m1, m2, m3, float(m4), float(m5), float(m6)])


def smm_objective(theta, raw, weights, moments_data=MOMENTS_DATA, phi=PHI_FIXED):
    for v, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < v < hi):
            return 1e6
    moments_model = compute_moments(theta, raw, weights, phi=phi)
    diff = moments_data - moments_model
    return float(MOMENT_WEIGHTS @ (diff ** 2))


# ── Multi-start optimisation ───────────────────────────────────────────────
# Structured spread across the bound box [0.1, 10.0]^4 — one start
# per region so the multi-start genuinely probes different basins:
#   low corner   — tests whether params collapse to the lower bound
#   high corner  — tests whether params blow up
#   mid-range    — default exploration
#   warm start   — refines at the current best theta (reproducibility check)
STARTING_POINTS = [
    [0.5,  0.10, 0.5,  0.5],
    [5.0,  1.0,  5.0,  5.0],
    [2.5,  0.5,  2.5,  1.5],
    [3.6,  0.6,  7.7,  1.0],
]
SCENARIO_LABELS = ["low corner", "high corner", "mid", "warm"]

NM_OPTIONS = {"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4, "disp": False}


def run_smm(raw, weights, verbose=True, phi=PHI_FIXED):
    results = []
    for label, x0 in zip(SCENARIO_LABELS, STARTING_POINTS):
        res = minimize(smm_objective, x0, args=(raw, weights, MOMENTS_DATA, phi),
                       method="Nelder-Mead", options=NM_OPTIONS)
        results.append(res)
        if verbose:
            vals = ", ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, res.x))
            print(f"  Start {label} {[round(x, 2) for x in x0]}")
            print(f"    → obj={res.fun:.6f}  [{vals}]  {'✓' if res.success else '✗'}")

    best = min(results, key=lambda r: r.fun)
    return best, results


if __name__ == "__main__":
    print("Loading data...")
    bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
    raw = bloc_data.set_index("bloc")

    weights_dict = {}
    for bloc in PLAYERS:
        weights_dict[bloc] = (0.3 * raw.loc[bloc, "emission_share"]
                              + 0.7 * raw.loc[bloc, "gdp_share"])
    total_w = sum(weights_dict.values())
    weights_norm = {k: v / total_w for k, v in weights_dict.items()}

    theta_orig = [3.0, 0.25, 1.5, 2.0]
    pre = compute_moments(theta_orig, raw, weights_norm)
    print(f"\nPre-SMM moments:")
    for name, m, t in zip(MOMENT_NAMES, pre, MOMENTS_DATA):
        print(f"  {name:32}: model={m:.4f}  target={t:.4f}")

    print("\nRunning SMM (Multi-start Nelder-Mead, φ=0.5)...")
    best, all_results = run_smm(raw, weights_norm, verbose=True)

    print(f"\nBest Calibration Result (obj={best.fun:.6f}):")
    for n, orig, est in zip(PARAM_NAMES, theta_orig, best.x):
        print(f"  {n}: {orig:.4f} → {est:.4f}")

    post = compute_moments(best.x, raw, weights_norm)
    print(f"\nPost-SMM moments:")
    for name, m, t in zip(MOMENT_NAMES, post, MOMENTS_DATA):
        print(f"  {name:32}: model={m:.4f}  target={t:.4f}  Δ={m-t:+.4f}")

    solutions = np.array([r.x for r in all_results])
    spread = solutions.std(axis=0)
    if spread.max() > 0.1:
        print(f"\n⚠ Parameter spread (std): {dict(zip(PARAM_NAMES, spread.round(4)))}")
    else:
        print(f"\n✓ Convergence stable across starting points.")

    np.save(os.path.join(ROOT_DIR, "results", "smm_best_theta.npy"), best.x)
    print(f"\nSaved best theta → results/smm_best_theta.npy")
