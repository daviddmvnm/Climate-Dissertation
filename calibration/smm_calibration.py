"""
smm_calibration.py
──────────────────
Simulated Method of Moments calibration for the Climate Cooperation Markov Game.

MOMENTS (all analytical — σ at t=1, state G0):
  M1: EU - US adoption gap                    (Target: 0.25)
  M2: US period-1 adoption probability        (Target: 0.10)
  M3: EU lead ratio (EU/US)                   (Target: 2.5)
  M4: CN period-1 adoption probability        (Target: 0.05)
  M5: P(zero adoption by period 2)            (Target: 0.15)
  M6: P(at least 2 blocs adopted by period 3) (Target: 0.60)
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
from itertools import product as iproduct
from functools import lru_cache
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS, solve_model, monte_carlo
from run_analysis import build_bloc_data, build_params, DISCOUNT, ETA, KAPPA

# ── Fixed parameters ──────────────────────────────────────────────────────
LAMBDA_FIXED   = 1.54
ETA_FIXED      = ETA     # 15.0
KAPPA_FIXED    = KAPPA   # 0.05
DISCOUNT_FIXED = {"US": 0.75, "EU": 0.85, "CN": 0.80, "RoW": 0.65}

# ── Real-world target moments ──────────────────────────────────────────────
MOMENTS_DATA = np.array([
    0.25,   # M1: EU - US adoption gap
    0.10,   # M2: US period-1 adoption probability
    2.50,   # M3: EU lead ratio (EU/US)
    0.05,   # M4: CN period-1 adoption probability
    0.15,   # M5: P(zero adoption by period 2)
    0.60,   # M6: P(at least 2 blocs adopted by period 3)
])

MOMENT_NAMES = [
    "EU - US adoption gap",
    "US period-1 adoption prob",
    "EU lead ratio (EU/US)",
    "CN period-1 adoption prob",
    "P(zero adoption by t=2)",
    "P(≥2 blocs by t=3)",
]

MOMENT_WEIGHTS = np.array([1.0, 1.0, 0.0016, 1.0, 1.0, 1.0])  # M3 scaled: (0.1/2.5)^2

# ── Parameter bounds for [α_c, α_d, α_p, α_b] ────────────────────────────
BOUNDS = [
    (0.5,  6.0),   # α_c  transition cost scaling
    (0.05, 1.0),   # α_d  climate damage scaling
    (0.1,  5.0),   # α_p  political pressure scaling
    (0.5,  6.0),   # α_b  coordination benefit scaling
]
PARAM_NAMES = ["α_c", "α_d", "α_p", "α_b"]

# MC draws
N_MC_OPT   = 500
N_MC_FINAL = 1000
MC_SEED    = 42


def build_smm_params(ac, ad, ap, ab, raw, weights):
    """Build GameParams with alpha vector; η, κ, λ, δ held fixed."""
    params = build_params(raw, weights, ac, ad, ap, ab,
                          discount=DISCOUNT_FIXED,
                          eta=ETA_FIXED, kappa=KAPPA_FIXED)
    
    # Assign heterogeneous lambda dict (here all fixed to 1.54 for calibration)
    lam_dict = {p: LAMBDA_FIXED for p in PLAYERS}
    return GameParams(**{**params.__dict__, "lam": lam_dict})


def compute_moments(theta, raw, weights, n_mc=N_MC_OPT, seed=MC_SEED):
    """
    Computes M1-M5 moments.
    """
    ac, ad, ap, ab = theta
    params = build_smm_params(ac, ad, ap, ab, raw, weights)

    try:
        V, sigma, _, _ = solve_model(params)
    except Exception:
        return np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6])

    G0  = (0, 0, 0, 0)
    idx = {p: i for i, p in enumerate(PLAYERS)}

    # All moments analytical from sigma at t=1, state G0
    s_EU  = sigma[1][G0][idx["EU"]]
    s_US  = sigma[1][G0][idx["US"]]
    s_CN  = sigma[1][G0][idx["CN"]]

    # M5: P(zero adoption by period 2) = P(all delay t=1) * P(all delay t=2 | G0)
    n = len(PLAYERS)
    p_all_delay_t1 = float(np.prod([1 - sigma[1][G0][i] for i in range(n)]))
    p_all_delay_t2 = float(np.prod([1 - sigma[2][G0][i] for i in range(n)]))
    m5 = p_all_delay_t1 * p_all_delay_t2

    m1 = float(s_EU - s_US)
    m2 = float(s_US)
    m3 = float(s_EU / s_US) if s_US > 1e-9 else 1e6
    m4 = float(s_CN)

    # M6: P(at least 2 blocs adopted by end of period 3) — forward DP
    dist = {G0: 1.0}
    for t in range(1, 4):
        new_dist = {}
        for G, gprob in dist.items():
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
    m6 = sum(prob for G, prob in dist.items() if sum(G) >= 2)

    return np.array([m1, m2, m3, m4, float(m5), float(m6)])


def smm_objective(theta, raw, weights, moments_data=MOMENTS_DATA):
    for v, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < v < hi):
            return 1e6

    moments_model = compute_moments(theta, raw, weights, n_mc=N_MC_OPT, seed=MC_SEED)
    diff = moments_data - moments_model
    return float(MOMENT_WEIGHTS @ (diff ** 2))


# ── Multi-start optimisation ───────────────────────────────────────────────
STARTING_POINTS = [
    [3.5,  0.30, 0.5,  3.0],   # Scenario A
    [2.0,  0.15, 2.5,  1.0],   # Scenario B
    [2.0,  0.30, 0.5,  1.0],   # Scenario C
    [3.0,  0.25, 1.5,  2.0],   # Original Grid Search Default
]
SCENARIO_LABELS = ["A", "B", "C", "Default"]

NM_OPTIONS = {"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4, "disp": False}


def run_smm(raw, weights, verbose=True):
    results = []
    for label, x0 in zip(SCENARIO_LABELS, STARTING_POINTS):
        res = minimize(smm_objective, x0, args=(raw, weights),
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
        weights_dict[bloc] = (0.5 * raw.loc[bloc, "emission_share"]
                             + 0.5 * raw.loc[bloc, "gdp_share"])
    total_w = sum(weights_dict.values())
    weights_norm = {k: v / total_w for k, v in weights_dict.items()}

    theta_orig = [3.0, 0.25, 1.5, 2.0]
    pre = compute_moments(theta_orig, raw, weights_norm, n_mc=N_MC_OPT)
    print(f"\nPre-SMM moments:  {dict(zip(MOMENT_NAMES, pre.round(4)))}")
    print(f"Target moments:   {dict(zip(MOMENT_NAMES, MOMENTS_DATA))}")

    print("\nRunning SMM (Multi-start Nelder-Mead)...")
    best, all_results = run_smm(raw, weights_norm, verbose=True)

    print(f"\nBest Calibration Result (obj={best.fun:.6f}):")
    for n, orig, est in zip(PARAM_NAMES, theta_orig, best.x):
        print(f"  {n}: {orig:.4f} → {est:.4f}")

    post = compute_moments(best.x, raw, weights_norm, n_mc=N_MC_FINAL)
    print(f"\nPost-SMM moments (n={N_MC_FINAL}):")
    for name, m in zip(MOMENT_NAMES, post):
        print(f"  {name:32}: {m:.4f}")

    solutions = np.array([r.x for r in all_results])
    spread = solutions.std(axis=0)
    if spread.max() > 0.1:
        print(f"\n⚠ Parameter spread (std): {dict(zip(PARAM_NAMES, spread.round(4)))}")
    else:
        print(f"\n✓ Convergence stable across starting points.")

    np.save(os.path.join(ROOT_DIR, "results", "smm_best_theta.npy"), best.x)
    print(f"\nSaved best theta → results/smm_best_theta.npy")