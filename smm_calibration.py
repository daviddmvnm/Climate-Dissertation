"""
smm_calibration.py
──────────────────
Simulated Method of Moments calibration for the Climate Cooperation Markov Game.

UPDATED MOMENTS:
  M1: EU - US adoption gap (45.3% - 22.7% = 0.23)
  M2: Mean adoption period of US | eventually adopting (Target: 3.5 periods)
      - Sensitive to patience/pressure, less so to t=1 cost.
  M3: EU lead ratio — σ_EU / max(σ_US, σ_CN) at t=1 (Target: 2.5x)
  M4: P(0.5 <= W < θ) — "Stuck-in-the-middle" failure mode (Target: 0.15)
      - Captures when a coalition forms but lacks the final push (benefit/pressure).
"""

import warnings
import numpy as np
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
    0.23,   # M1: EU - US adoption gap (0.23)
    3.50,   # M2: Mean adoption period of US | adopting
    2.50,   # M3: EU lead ratio (σ_EU / max rival)
    0.15,   # M4: P(0.5 <= W < θ) - The "stuck" failure mode
])

MOMENT_NAMES = [
    "EU - US adoption gap",
    "Mean US adoption period | success",
    "EU lead ratio (σ_EU/max rival)",
    "P(stuck: 0.5 <= W < θ)",
]

# Weights to normalize scales: M2 (~3.5) and M3 (~2.5) are downweighted 
# relative to probabilities (M1, M4) to ensure balanced influence.
MOMENT_WEIGHTS = np.array([1.0, 0.2, 0.2, 1.0])

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
    Computes updated M1-M4 moments.
    """
    ac, ad, ap, ab = theta
    params = build_smm_params(ac, ad, ap, ab, raw, weights)

    try:
        V, sigma, _, _ = solve_model(params)
    except Exception:
        return np.array([1e6, 1e6, 1e6, 1e6])

    G0  = (0, 0, 0, 0)
    idx = {p: i for i, p in enumerate(PLAYERS)}

    # M1 & M3: Analytical from sigma
    s_EU = sigma[1][G0][idx["EU"]]
    s_US = sigma[1][G0][idx["US"]]
    s_CN = sigma[1][G0][idx["CN"]]
    rival_max = max(s_US, s_CN, 0.001)

    m1 = s_EU - s_US
    m3 = s_EU / rival_max

    # M2 & M4: Monte Carlo
    try:
        W_paths, adopt_time = monte_carlo(V, sigma, params, n_runs=n_mc, seed=seed)

        # M2: Mean adoption period of US conditional on adopting
        us_adopt_t = adopt_time[:, idx["US"]]
        us_success = us_adopt_t[us_adopt_t < np.inf]
        if len(us_success) > 0:
            m2 = float(us_success.mean())
        else:
            m2 = float(params.T) # Penalty if US never adopts

        # M4: P(0.5 <= W_final < θ) - Stuck in the middle
        W_final = W_paths[:, -1]
        stuck = (W_final >= 0.5) & (W_final < params.theta)
        m4 = float(stuck.mean())

    except Exception:
        m2, m4 = 1e6, 1e6

    return np.array([m1, m2, m3, m4])


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
    bloc_data = build_bloc_data(data_dir=".")
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