"""
smm_standard_errors.py
──────────────────────
Standard errors and confidence intervals for SMM parameter estimates (α_c, α_d, α_p, α_b).

Two approaches:

  (1) Asymptotic sandwich SE (delta method):
        Var(θ̂) = (G'WG)^{-1} G'WΩWG (G'WG)^{-1}
      where G = ∂m/∂θ is the moment Jacobian at θ̂ (numerical finite differences),
            W = diag(MOMENT_WEIGHTS)  (the objective's weight matrix),
            Ω = diag(1 / MOMENT_WEIGHTS)  (implied moment covariance — the weight
                matrix is chosen so W ≈ Ω^{-1}).

  (2) Parametric bootstrap of target moments:
        For b = 1..B, draw m*_b ~ N(MOMENTS_DATA, diag(σ²)) with σ_k = 1/√w_k,
        re-run Nelder-Mead from θ̂ against the perturbed targets,
        collect θ̂*_b, report 2.5/97.5 percentiles.

Outputs:
  results/smm_standard_errors.csv  — point estimate, sandwich SE, 95% sandwich CI,
                                     bootstrap mean, bootstrap 95% CI.
  results/smm_bootstrap_draws.csv  — all B bootstrap parameter draws.
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))

import warnings
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

from climate_game import PLAYERS
from run_analysis import build_bloc_data
from smm_calibration import (
    compute_moments, build_smm_params,
    MOMENTS_DATA, MOMENT_NAMES, MOMENT_WEIGHTS, PARAM_NAMES,
    BOUNDS, NM_OPTIONS, N_MC_OPT,
)

SEP = "=" * 72
SEP2 = "-" * 72

# ── Configuration ──────────────────────────────────────────────────────────
# Finite-difference step for Jacobian (relative to parameter scale).
FD_STEP_REL = 5e-3
FD_STEP_ABS_FLOOR = 1e-3

# Bootstrap — B draws of perturbed target moments; each solves SMM from θ̂.
B_DEFAULT = 200
BOOT_SEED = 20260417

# Clamp implied σ_k = 1/√w_k for very-low-weight moments so they do not dominate
# the parametric bootstrap. Expressed as a cap relative to the target magnitude.
SIGMA_REL_CAP = 0.5  # no moment is perturbed by more than ±50% of its target


def numerical_jacobian(theta, raw, weights, step_rel=FD_STEP_REL):
    """Central-difference Jacobian G[i,j] = ∂m_i/∂θ_j evaluated at θ."""
    theta = np.asarray(theta, dtype=float)
    K = len(theta)
    M = len(MOMENTS_DATA)
    J = np.zeros((M, K))
    for j in range(K):
        h = max(step_rel * abs(theta[j]), FD_STEP_ABS_FLOOR)
        tp = theta.copy(); tp[j] += h
        tm = theta.copy(); tm[j] -= h
        m_plus  = compute_moments(tp, raw, weights, n_mc=N_MC_OPT)
        m_minus = compute_moments(tm, raw, weights, n_mc=N_MC_OPT)
        J[:, j] = (m_plus - m_minus) / (2 * h)
    return J


def sandwich_variance(G, W_diag, Omega_diag):
    """Asymptotic SMM sandwich: Var = (G'WG)^{-1} G'WΩWG (G'WG)^{-1}."""
    W = np.diag(W_diag)
    Omega = np.diag(Omega_diag)
    GtWG = G.T @ W @ G
    GtWG_inv = np.linalg.pinv(GtWG)
    bread = GtWG_inv
    meat  = G.T @ W @ Omega @ W @ G
    return bread @ meat @ bread


def bootstrap_objective(theta, raw, weights, targets):
    """Same shape as smm_objective but with an injectable target vector."""
    for v, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < v < hi):
            return 1e6
    m_model = compute_moments(theta, raw, weights, n_mc=N_MC_OPT)
    diff = targets - m_model
    return float(MOMENT_WEIGHTS @ (diff ** 2))


def run_parametric_bootstrap(theta_hat, raw, weights, B, sigma_targets, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    draws = np.full((B, len(theta_hat)), np.nan)
    for b in range(B):
        perturbed = MOMENTS_DATA + rng.normal(0.0, sigma_targets)
        res = minimize(
            bootstrap_objective, theta_hat,
            args=(raw, weights, perturbed),
            method="Nelder-Mead", options=NM_OPTIONS,
        )
        if res.success or res.fun < 1e5:
            draws[b] = res.x
        if (b + 1) % max(1, B // 10) == 0:
            print(f"    bootstrap {b+1}/{B} done", flush=True)
    return draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=B_DEFAULT,
                        help="Number of bootstrap target draws (default 200).")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip the parametric bootstrap (sandwich SE only).")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
    raw = bloc_data.set_index("bloc")
    weights = {}
    for bloc in PLAYERS:
        weights[bloc] = (0.3 * raw.loc[bloc, "emission_share"]
                       + 0.7 * raw.loc[bloc, "gdp_share"])
    tot = sum(weights.values())
    weights = {k: v / tot for k, v in weights.items()}

    theta_path = os.path.join(ROOT_DIR, "results", "smm_best_theta.npy")
    theta_hat = np.asarray(np.load(theta_path), dtype=float)
    print(f"θ̂ = {dict(zip(PARAM_NAMES, theta_hat.round(4)))}")

    # ── (1) Sandwich standard errors ───────────────────────────────────────
    print(f"\n{SEP}\n  (1) ASYMPTOTIC SANDWICH SE\n{SEP}")
    print("  Computing moment Jacobian (central differences)...", flush=True)
    G = numerical_jacobian(theta_hat, raw, weights)
    print("\n  Jacobian  ∂m_i/∂θ_j  (rows = moments, cols = params):")
    hdr = " " * 34 + "".join(f"{p:>10}" for p in PARAM_NAMES)
    print(hdr)
    print(SEP2)
    for i, name in enumerate(MOMENT_NAMES):
        row = "".join(f"{G[i, j]:>10.4f}" for j in range(len(theta_hat)))
        print(f"  {name:<32}{row}")
    cond = np.linalg.cond(G)
    print(f"\n  cond(G) = {cond:.2f}  "
          f"({'well-identified' if cond < 500 else 'weakly identified'})")

    W_diag = MOMENT_WEIGHTS.astype(float)
    # Ω_k implied by the objective weights (minimum-distance SMM uses W = Ω^{-1}).
    # We cap σ_k at SIGMA_REL_CAP × |target| so that very low-weight moments
    # (e.g., M3 with w=1.6e-3) do not blow up the sandwich — same cap used
    # for the parametric bootstrap so the two SE concepts are comparable.
    sigma_raw  = np.where(W_diag > 0, 1.0 / np.sqrt(np.maximum(W_diag, 1e-12)), 0.0)
    sigma_cap  = SIGMA_REL_CAP * np.abs(MOMENTS_DATA)
    sigma_used = np.minimum(sigma_raw, sigma_cap)
    Omega_diag = sigma_used ** 2
    Var = sandwich_variance(G, W_diag, Omega_diag)
    se_sandwich = np.sqrt(np.maximum(np.diag(Var), 0.0))

    # Clip CIs to the parameter bounds — α's are non-negative by construction.
    bound_lo = np.array([lo for lo, _ in BOUNDS])
    bound_hi = np.array([hi for _, hi in BOUNDS])
    ci_lo = np.clip(theta_hat - 1.96 * se_sandwich, bound_lo, bound_hi)
    ci_hi = np.clip(theta_hat + 1.96 * se_sandwich, bound_lo, bound_hi)

    # Flag parameters sitting on a bound — asymptotic normality fails there.
    on_bound = [
        (abs(theta_hat[j] - bound_lo[j]) < 1e-3) or
        (abs(theta_hat[j] - bound_hi[j]) < 1e-3)
        for j in range(len(theta_hat))
    ]

    print(f"\n  {'Param':<6} {'θ̂':>10} {'SE':>10} {'95% CI (asym, clipped)':>28}  note")
    print(SEP2)
    for name, th, se, lo, hi, ob in zip(
        PARAM_NAMES, theta_hat, se_sandwich, ci_lo, ci_hi, on_bound
    ):
        note = "on bound — asym SE unreliable" if ob else ""
        print(f"  {name:<6} {th:>10.4f} {se:>10.4f}   [{lo:>8.4f}, {hi:>8.4f}]  {note}")

    # ── (2) Parametric bootstrap ───────────────────────────────────────────
    boot_draws = None
    if not args.skip_bootstrap:
        print(f"\n{SEP}\n  (2) PARAMETRIC BOOTSTRAP  (B = {args.B})\n{SEP}")
        # Reuse the (capped) σ from the sandwich so the two methods share a
        # common moment-uncertainty assumption.
        sigma_targets = sigma_used
        print("  Implied target SDs used for resampling:")
        for name, t, s in zip(MOMENT_NAMES, MOMENTS_DATA, sigma_targets):
            print(f"    {name:<32}  target={t:>7.3f}   σ={s:>7.3f}")
        print(flush=True)
        boot_draws = run_parametric_bootstrap(
            theta_hat, raw, weights, args.B, sigma_targets, seed=BOOT_SEED,
        )
        ok_mask = ~np.isnan(boot_draws).any(axis=1)
        n_ok = int(ok_mask.sum())
        boot_ok = boot_draws[ok_mask]
        print(f"\n  {n_ok}/{args.B} bootstrap replications completed.")
        boot_mean = boot_ok.mean(axis=0)
        boot_sd   = boot_ok.std(axis=0, ddof=1)
        q_lo = np.quantile(boot_ok, 0.025, axis=0)
        q_hi = np.quantile(boot_ok, 0.975, axis=0)
        print(f"\n  {'Param':<6} {'mean':>10} {'SD':>10} {'95% CI (boot)':>24}")
        print(SEP2)
        for name, m, s, lo, hi in zip(PARAM_NAMES, boot_mean, boot_sd, q_lo, q_hi):
            print(f"  {name:<6} {m:>10.4f} {s:>10.4f}   [{lo:>8.4f}, {hi:>8.4f}]")

    # ── (3) Save combined table ────────────────────────────────────────────
    rows = []
    for j, name in enumerate(PARAM_NAMES):
        row = {
            "param":        name,
            "estimate":     float(theta_hat[j]),
            "on_bound":     bool(on_bound[j]),
            "se_sandwich":  float(se_sandwich[j]),
            "ci95_lo_sand": float(ci_lo[j]),
            "ci95_hi_sand": float(ci_hi[j]),
        }
        if boot_draws is not None:
            ok_mask = ~np.isnan(boot_draws).any(axis=1)
            boot_ok = boot_draws[ok_mask]
            row.update({
                "boot_n":        int(ok_mask.sum()),
                "boot_mean":     float(boot_ok[:, j].mean()),
                "boot_sd":       float(boot_ok[:, j].std(ddof=1)),
                "ci95_lo_boot":  float(np.quantile(boot_ok[:, j], 0.025)),
                "ci95_hi_boot":  float(np.quantile(boot_ok[:, j], 0.975)),
            })
        rows.append(row)

    out_path = os.path.join(ROOT_DIR, "results", "smm_standard_errors.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved  → {out_path}")

    if boot_draws is not None:
        draws_path = os.path.join(ROOT_DIR, "results", "smm_bootstrap_draws.csv")
        pd.DataFrame(boot_draws, columns=PARAM_NAMES).to_csv(draws_path, index=False)
        print(f"Saved  → {draws_path}")

    print(f"\n{SEP}\n  DONE\n{SEP}")


if __name__ == "__main__":
    main()
