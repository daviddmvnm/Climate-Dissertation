"""
phi_sweep_calibration.py
────────────────────────
Re-calibrates the SMM at each φ ∈ [0, 1] to show how the estimated
parameter vector shifts as the spillover channel tilts from pure
cost-learning (φ=0) to pure pressure (φ=1).

At each φ the full 4-parameter SMM [α_c, α_spill, α_d, α_b] is
re-estimated from 2-start Nelder-Mead, keeping all moment targets
and fixed parameters identical.

Outputs:
  results/phi_sweep_calibration.csv   — parameter vectors + moment fit per φ
  results/figures/phi_sweep_params.png — parameter trajectories
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
sys.path.insert(0, os.path.join(ROOT_DIR, "calibration"))
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product as iproduct
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

from climate_game import GameParams, PLAYERS, solve_model, compute_W
from run_analysis import build_bloc_data, build_params, DISCOUNT, ETA, KAPPA, OUT_DIR, FIG_DIR
from smm_calibration import (
    LAMBDA_FIXED, ETA_FIXED, KAPPA_FIXED, DISCOUNT_FIXED,
    MOMENTS_DATA, MOMENT_NAMES, MOMENT_WEIGHTS,
    BOUNDS, PARAM_NAMES, NM_OPTIONS,
    STARTING_POINTS as _ALL_STARTS, SCENARIO_LABELS as _ALL_LABELS,
    _period2_expected_adopt,
)

N_MOMENTS = len(MOMENTS_DATA)

# ── Helpers ───────────────────────────────────────────────────────────────

def build_params_at_phi(ac, a_spill, ad, ab, raw, weights, phi):
    """Build GameParams at a given φ; homogeneous λ, fixed δ/η/κ."""
    params = build_params(raw, weights, ac=ac, ad=ad, a_spill=a_spill, ab=ab,
                          discount=DISCOUNT_FIXED,
                          eta=ETA_FIXED, kappa=KAPPA_FIXED,
                          phi=phi)
    lam_dict = {p: LAMBDA_FIXED for p in PLAYERS}
    return GameParams(**{**params.__dict__, "lam": lam_dict})


def compute_moments_at_phi(theta, raw, weights, phi):
    """M1–M6 moment vector at a given φ."""
    ac, a_spill, ad, ab = theta
    params = build_params_at_phi(ac, a_spill, ad, ab, raw, weights, phi)

    try:
        V, sigma, _, _ = solve_model(params)
    except Exception:
        return np.array([1e6] * N_MOMENTS)

    G0  = (0, 0, 0, 0)
    idx = {p: i for i, p in enumerate(PLAYERS)}
    n   = len(PLAYERS)

    s_EU = float(sigma[1][G0][idx["EU"]])
    s_US = float(sigma[1][G0][idx["US"]])
    s_CN = float(sigma[1][G0][idx["CN"]])

    m1 = s_EU - s_US
    m2 = s_US
    m3 = s_CN

    e_cn_2 = _period2_expected_adopt(sigma, idx["CN"], G0)
    m4 = e_cn_2 / s_CN if s_CN > 1e-9 else 1e6

    m5 = _period2_expected_adopt(sigma, idx["US"], G0)

    # M6: first-passage timing
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


def objective_at_phi(theta, raw, weights, phi):
    for v, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < v < hi):
            return 1e6
    m = compute_moments_at_phi(theta, raw, weights, phi)
    diff = MOMENTS_DATA - m
    return float(MOMENT_WEIGHTS @ (diff ** 2))


SWEEP_STARTS = _ALL_STARTS[:2]
SWEEP_LABELS = _ALL_LABELS[:2]

def calibrate_at_phi(phi, raw, weights, verbose=True):
    """Multi-start SMM at a single φ value (2 starts for speed)."""
    results = []
    for label, x0 in zip(SWEEP_LABELS, SWEEP_STARTS):
        res = minimize(objective_at_phi, x0, args=(raw, weights, phi),
                       method="Nelder-Mead", options=NM_OPTIONS)
        results.append(res)
    best = min(results, key=lambda r: r.fun)
    if verbose:
        vals = ", ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, best.x))
        print(f"  φ={phi:.2f}  obj={best.fun:.6f}  [{vals}]")
    return best


def _phi_worker(args):
    """Pool worker: calibrate one φ and return the row dict."""
    phi, raw, weights = args
    best = calibrate_at_phi(phi, raw, weights, verbose=False)
    ac, a_spill, ad, ab = best.x
    m = compute_moments_at_phi(best.x, raw, weights, phi)
    row = {
        "phi": round(float(phi), 2),
        "alpha_c": round(ac, 4),
        "alpha_spill": round(a_spill, 4),
        "alpha_d": round(ad, 4),
        "alpha_b": round(ab, 4),
        "objective": round(best.fun, 6),
    }
    for name, val in zip(MOMENT_NAMES, m):
        row[f"model_{name}"] = round(float(val), 4)
    row["at_bound"] = any(v < lo + 0.05 or v > hi - 0.05
                          for v, (lo, hi) in zip(best.x, BOUNDS))
    return row


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
    raw = bloc_data.set_index("bloc")

    weights = {}
    for bloc in PLAYERS:
        weights[bloc] = 0.3 * raw.loc[bloc, "emission_share"] + 0.7 * raw.loc[bloc, "gdp_share"]
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    PHI_GRID = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0

    n_workers = min(len(PHI_GRID), cpu_count())
    print(f"\nSweeping φ over {len(PHI_GRID)} values, 2-start SMM at each "
          f"({n_workers} parallel workers)...\n")

    tasks = [(phi, raw, weights) for phi in PHI_GRID]
    with Pool(processes=n_workers) as pool:
        rows = []
        for row in pool.imap_unordered(_phi_worker, tasks):
            vals = ", ".join(f"{n}={row[f'alpha_{k}']:.4f}"
                             for n, k in zip(PARAM_NAMES, ["c", "d", "spill", "b"]))
            print(f"  φ={row['phi']:.2f}  obj={row['objective']:.6f}  [{vals}]")
            rows.append(row)
    rows.sort(key=lambda r: r["phi"])

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "phi_sweep_calibration.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Print table ───────────────────────────────────────────────────
    print(f"\n{'φ':>5}  {'α_c':>7}  {'α_spill':>7}  {'α_d':>7}  {'α_b':>7}  {'obj':>9}  {'bound?':>6}")
    print("-" * 58)
    for _, r in df.iterrows():
        flag = " !" if r["at_bound"] else ""
        print(f"{r['phi']:>5.2f}  {r['alpha_c']:>7.3f}  {r['alpha_spill']:>7.3f}  "
              f"{r['alpha_d']:>7.3f}  {r['alpha_b']:>7.3f}  {r['objective']:>9.6f}{flag}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: parameter trajectories
    ax = axes[0]
    for param, color, label in [
        ("alpha_c",     "#E63946", "α_c (baseline cost)"),
        ("alpha_spill", "#457B9D", "α_spillover"),
        ("alpha_d",     "#E9C46A", "α_d (damages)"),
        ("alpha_b",     "#2A9D8F", "α_b (benefits)"),
    ]:
        ax.plot(df["phi"], df[param], "o-", color=color, label=label, markersize=5)
    ax.axhspan(0.1, 0.15, color="red", alpha=0.07)
    ax.axhspan(9.95, 10.0, color="red", alpha=0.07)
    ax.set_xlabel("φ  (0 = pure cost-learning, 1 = pure pressure)", fontsize=11)
    ax.set_ylabel("Estimated parameter value", fontsize=11)
    ax.set_title("Calibrated Parameters vs Spillover Mix", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: objective value
    ax = axes[1]
    ax.plot(df["phi"], df["objective"], "s-", color="black", markersize=6)
    bound_df = df[df["at_bound"]]
    if not bound_df.empty:
        ax.scatter(bound_df["phi"], bound_df["objective"],
                   color="red", zorder=5, s=60, marker="x", label="param at bound")
        ax.legend(fontsize=9)
    ax.set_xlabel("φ  (0 = pure cost-learning, 1 = pure pressure)", fontsize=11)
    ax.set_ylabel("SMM objective", fontsize=11)
    ax.set_title("Moment Fit vs Spillover Mix", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "phi_sweep_params.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
