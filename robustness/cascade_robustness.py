"""
cascade_robustness.py
─────────────────────
Appendix C: Robustness of the λ_US cascade.

At each point of a 3×3 grid over
    α_c ∈ {0.75, 1.00, 1.25} × (SMM baseline α_c),
    δ_i → δ_i + s  for s ∈ {-0.05, 0.00, +0.05} and all four blocs,
solve the model at λ_US = 0.5 and λ_US = 5.0 (others held at λ baseline),
and record the change in period-one adoption probabilities plus mean
coordination timing.

The cascade is defined to hold if all four diagnostic signs appear:
    Δσ_US(1) < 0     (US pulls back)
    Δσ_EU(1) > 0     (EU cascades)
    Δσ_CN(1) > 0     (CN cascades)
    Δ timing  < 0    (coordination arrives earlier)

When coordination collapses entirely at a grid point (success rate near
zero at both λ_US endpoints), mean coordination timing is undefined and
the row is marked "collapse" rather than pass/fail.

Outputs
-------
  results/cascade_robustness.csv
  results/dissertation/table_lambda_us_cascade_robustness.tex
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

from climate_game import PLAYERS, solve_model, monte_carlo, compute_W
from run_analysis import build_bloc_data, build_params, SMM_BASELINE, DISCOUNT

# ── Settings ───────────────────────────────────────────────────────────────
ALPHA_C_SCALES   = [0.75, 1.00, 1.25]
DISCOUNT_SHIFTS  = [-0.05, 0.00, +0.05]
LAMBDA_US_LO     = 0.5
LAMBDA_US_HI     = 5.0
N_MC             = 1000
SEED             = 42

OUT_CSV  = os.path.join(ROOT_DIR, "results", "cascade_robustness.csv")
OUT_TEX  = os.path.join(ROOT_DIR, "results", "dissertation",
                        "table_lambda_us_cascade_robustness.tex")
os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)

IDX = {p: i for i, p in enumerate(PLAYERS)}
G0  = (0, 0, 0, 0)


def _coord_timing(W_paths, theta):
    """Mean period at which W first crosses θ; returns (success_rate, mean_t)."""
    T = W_paths.shape[1] - 1
    first_cross = np.full(W_paths.shape[0], np.inf)
    for t in range(T + 1):
        mask = (W_paths[:, t] >= theta) & np.isinf(first_cross)
        first_cross[mask] = t
    success = np.isfinite(first_cross)
    if not success.any():
        return 0.0, np.nan
    return float(success.mean()), float(first_cross[success].mean())


def solve_point(alpha_c_scale, discount_shift, lambda_us, raw, weights):
    """Build params at a perturbed grid point with specified λ_US, solve, MC."""
    b = SMM_BASELINE
    discount = {p: DISCOUNT[p] + discount_shift for p in PLAYERS}
    lam_map  = {p: b["lam"] for p in PLAYERS}
    lam_map["US"] = lambda_us

    params = build_params(
        raw, weights,
        ac=b["ac"] * alpha_c_scale,
        ad=b["ad"], a_spill=b["a_spill"], ab=b["ab"],
        discount=discount,
        lambda_map=lam_map,
    )

    V, sigma, _, _ = solve_model(params)
    W_paths, _     = monte_carlo(V, sigma, params, n_runs=N_MC, seed=SEED)
    success, mean_t = _coord_timing(W_paths, params.theta)

    return {
        "sigma_US": float(sigma[1][G0][IDX["US"]]),
        "sigma_EU": float(sigma[1][G0][IDX["EU"]]),
        "sigma_CN": float(sigma[1][G0][IDX["CN"]]),
        "success":  success,
        "timing":   mean_t,
    }


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
    raw = bloc_data.set_index("bloc")

    weights = {}
    for bloc in PLAYERS:
        weights[bloc] = 0.3 * raw.loc[bloc, "emission_share"] + 0.7 * raw.loc[bloc, "gdp_share"]
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    b = SMM_BASELINE
    print(f"Baseline α_c={b['ac']:.3f}, δ={DISCOUNT}, λ_others={b['lam']:.2f}")
    print(f"\nSweeping 3×3 grid, λ_US ∈ {{{LAMBDA_US_LO}, {LAMBDA_US_HI}}}...\n")

    rows = []
    for scale in ALPHA_C_SCALES:
        for shift in DISCOUNT_SHIFTS:
            lo = solve_point(scale, shift, LAMBDA_US_LO, raw, weights)
            hi = solve_point(scale, shift, LAMBDA_US_HI, raw, weights)

            d_us = hi["sigma_US"] - lo["sigma_US"]
            d_eu = hi["sigma_EU"] - lo["sigma_EU"]
            d_cn = hi["sigma_CN"] - lo["sigma_CN"]

            # Coordination collapses if neither endpoint achieves meaningful success.
            collapse = (lo["success"] < 0.05) and (hi["success"] < 0.05)
            if collapse:
                d_timing = np.nan
                cascade  = None
                verdict  = "collapse"
            else:
                # Only compare timing where both endpoints have some success.
                if np.isnan(lo["timing"]) or np.isnan(hi["timing"]):
                    d_timing = np.nan
                    verdict  = "collapse"
                    cascade  = None
                else:
                    d_timing = hi["timing"] - lo["timing"]
                    cascade  = (d_us < 0) and (d_eu > 0) and (d_cn > 0) and (d_timing < 0)
                    verdict  = "hold" if cascade else "fail"

            row = {
                "alpha_c_scale":  scale,
                "discount_shift": shift,
                "sigma_US_lo":    round(lo["sigma_US"], 4),
                "sigma_US_hi":    round(hi["sigma_US"], 4),
                "sigma_EU_lo":    round(lo["sigma_EU"], 4),
                "sigma_EU_hi":    round(hi["sigma_EU"], 4),
                "sigma_CN_lo":    round(lo["sigma_CN"], 4),
                "sigma_CN_hi":    round(hi["sigma_CN"], 4),
                "success_lo":     round(lo["success"], 4),
                "success_hi":     round(hi["success"], 4),
                "timing_lo":      None if np.isnan(lo["timing"]) else round(lo["timing"], 3),
                "timing_hi":      None if np.isnan(hi["timing"]) else round(hi["timing"], 3),
                "delta_sigma_US": round(d_us, 4),
                "delta_sigma_EU": round(d_eu, 4),
                "delta_sigma_CN": round(d_cn, 4),
                "delta_timing":   None if np.isnan(d_timing) else round(d_timing, 3),
                "cascade_holds":  cascade,
                "verdict":        verdict,
            }
            rows.append(row)
            print(f"  α_c×{scale:.2f}  δ{shift:+.2f}  "
                  f"Δσ_US={d_us:+.3f}  Δσ_EU={d_eu:+.3f}  Δσ_CN={d_cn:+.3f}  "
                  f"Δt={'n/a' if np.isnan(d_timing) else f'{d_timing:+.2f}'}  "
                  f"[{verdict}]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # ── Summary counts ─────────────────────────────────────────────────────
    n_hold     = int((df["verdict"] == "hold").sum())
    n_fail     = int((df["verdict"] == "fail").sum())
    n_collapse = int((df["verdict"] == "collapse").sum())
    print(f"\n  Cascade holds:   {n_hold} / 9")
    print(f"  Cascade fails:   {n_fail} / 9")
    print(f"  Coord. collapse: {n_collapse} / 9")

    # ── LaTeX table ────────────────────────────────────────────────────────
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Robustness of the $\lambda_{US}$ cascade across a $3\times 3$ grid of "
        r"perturbed cost ($\alpha_c$) and patience ($\delta$) environments. At each grid "
        r"point, $\lambda_{US}$ is swept from " + f"{LAMBDA_US_LO}" + r" to "
        + f"{LAMBDA_US_HI}" + r" with other blocs' $\lambda$ held at the SMM baseline; "
        r"reported values are the high-minus-low differences. The cascade holds if all "
        r"four diagnostics take the expected sign "
        r"($\Delta\sigma_{US}(1)<0$, $\Delta\sigma_{EU}(1)>0$, $\Delta\sigma_{CN}(1)>0$, "
        r"$\Delta\,\text{timing}<0$). Grid points where coordination collapses "
        r"entirely at both endpoints are marked ``collapse''; the cascade is vacuously "
        r"undefined there rather than violated.}",
        r"\label{tab:lambda_us_cascade_robustness}",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"$\alpha_c$ scale & $\delta$ shift & "
        r"$\Delta\sigma_{US}(1)$ & $\Delta\sigma_{EU}(1)$ & $\Delta\sigma_{CN}(1)$ & "
        r"$\Delta\,\text{timing}$ & Success (lo / hi) & Verdict \\",
        r"\midrule",
    ]

    verdict_tex = {"hold": r"\checkmark", "fail": r"$\times$", "collapse": r"collapse"}
    for _, r in df.iterrows():
        dt = "--" if pd.isna(r["delta_timing"]) else f"{r['delta_timing']:+.2f}"
        lines.append(
            f"{r['alpha_c_scale']:.2f} & {r['discount_shift']:+.2f} & "
            f"{r['delta_sigma_US']:+.3f} & {r['delta_sigma_EU']:+.3f} & "
            f"{r['delta_sigma_CN']:+.3f} & {dt} & "
            f"{r['success_lo']:.2f} / {r['success_hi']:.2f} & "
            f"{verdict_tex[r['verdict']]} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {OUT_TEX}")
