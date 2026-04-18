"""
equilibrium_uniqueness.py
─────────────────────────
Three tests for QRE equilibrium uniqueness across the game tree:

  TEST 1 — Multiple initialisations
    Re-runs the QRE fixed-point at every (t, G) node from K random starting
    probability vectors (plus the four corners of the simplex) and checks
    whether all converge to the same σ* as the baseline (σ=0.5 init).

  TEST 2 — Continuation value stability
    Explicitly checks corners (0,0,0,0), (1,1,1,1), and random interior
    points at every active node in the baseline solve. Reports the maximum
    deviation across all nodes and all initialisations.

  TEST 3 — Eigenvalue check on QRE Jacobian
    At the fitted σ* for each active node, computes the Jacobian of the
    logit best-response map F(σ) and checks that all eigenvalues lie
    strictly inside the unit circle. Confirms σ* is a stable attractor.

Outputs
-------
  results/equilibrium_uniqueness.csv   — per-node results
  results/dissertation/table_equilibrium_uniqueness.tex
  Console summary with PASS/FAIL verdicts
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))
import warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")

from climate_game import (
    PLAYERS, ALL_G, GameParams,
    compute_W, get_active, flow_adopt, flow_delay,
    qre_probability, inclusive_value, flow_state,
    solve_model,
)
from run_analysis import build_bloc_data, build_params, SMM_BASELINE

os.makedirs(os.path.join(ROOT_DIR, "results", "dissertation"), exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
N_RANDOM_INITS  = 20      # random interior starting points per node
TOL             = 1e-8    # convergence tolerance (same as solve_model)
MAX_ITER        = 500     # allow more iterations for corner inits
AGREE_TOL       = 1e-5    # max acceptable deviation between two converged σ*

SEP = "=" * 65


# ── Setup ─────────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
raw = bloc_data.set_index("bloc")
weights = {}
for bloc in PLAYERS:
    weights[bloc] = (0.5 * raw.loc[bloc, "emission_share"]
                   + 0.5 * raw.loc[bloc, "gdp_share"])
total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}

b = SMM_BASELINE
params = build_params(raw, weights, ac=b["ac"], ad=b["ad"],
                      a_spill=b["a_spill"], ab=b["ab"], lam=b["lam"])

print("Running baseline solve...", flush=True)
V, sigma_base, QA_table, QD_table = solve_model(params)
print(f"  Done. T={params.T}, states={len(ALL_G)}\n")


# ── QRE fixed-point runner (parameterised init) ───────────────────────────────

def run_fixed_point(t, G, active, W, V_next, params, init_sigmas):
    """
    Run QRE fixed-point from a given initialisation.
    Returns (converged_sigmas, n_iters, converged:bool).
    """
    current = dict(init_sigmas)

    for it in range(MAX_ITER):
        new_sigmas = {}
        new_QA, new_QD = {}, {}

        for i in active:
            p      = PLAYERS[i]
            others = [j for j in active if j != i]
            combos = list(product([0, 1], repeat=len(others)))

            EV_adopt = EV_delay = 0.0
            for oa in combos:
                prob = 1.0
                for idx, j in enumerate(others):
                    prob *= current[j] if oa[idx] == 1 else (1 - current[j])
                G_adopt = list(G); G_adopt[i] = 1
                G_delay = list(G)
                for idx, j in enumerate(others):
                    if oa[idx] == 1:
                        G_adopt[j] = 1
                        G_delay[j] = 1
                EV_adopt += prob * V_next[tuple(G_adopt)][i]
                EV_delay += prob * V_next[tuple(G_delay)][i]

            qa = flow_adopt(p, t, W, params) + params.discount[p] * EV_adopt
            qd = flow_delay(p, t, W, params) + params.discount[p] * EV_delay
            new_QA[i] = qa
            new_QD[i] = qd
            new_sigmas[i] = qre_probability(qa, qd, params.lam[p])

        max_diff = max(abs(new_sigmas[i] - current[i]) for i in active)
        current = new_sigmas
        if max_diff < TOL:
            return current, it + 1, True

    return current, MAX_ITER, False


def make_inits(active, rng, n_random=N_RANDOM_INITS):
    """Generate corner + random interior initialisations."""
    inits = []
    # Four structural corners
    for val in [0.0, 1.0]:
        inits.append({i: val for i in active})
    # Mixed corners — one player at each extreme
    for focal in active:
        inits.append({i: (0.01 if i == focal else 0.99) for i in active})
        inits.append({i: (0.99 if i == focal else 0.01) for i in active})
    # Midpoint (standard baseline)
    inits.append({i: 0.5 for i in active})
    # Random interior
    for _ in range(n_random):
        vals = rng.uniform(0.05, 0.95, len(active))
        inits.append({i: float(v) for i, v in zip(active, vals)})
    return inits


# ── TEST 1 & 2: Multiple initialisations ─────────────────────────────────────
print(f"{SEP}\n  TEST 1 & 2 — MULTIPLE INITIALISATIONS\n{SEP}")
print(f"  K={N_RANDOM_INITS} random + corners per node, tol={TOL}")
print(f"  Agreement tolerance: {AGREE_TOL}\n", flush=True)

rng = np.random.default_rng(42)
rows = []
total_nodes = max_deviation = 0
n_disagreements = 0
n_non_converged = 0

for t in range(params.T, 0, -1):
    for G in ALL_G:
        active = get_active(G)
        if not active:
            continue

        W      = compute_W(G, params)
        V_next = V[t + 1]

        # Baseline σ* (from solve_model, σ=0.5 init)
        sigma_star = {i: sigma_base[t][G][i] for i in active}

        inits = make_inits(active, rng)
        node_deviations = []
        node_non_conv   = 0

        for init in inits:
            sig_alt, n_it, conv = run_fixed_point(t, G, active, W, V_next, params, init)
            if not conv:
                node_non_conv += 1
                n_non_converged += 1
                continue
            dev = max(abs(sig_alt[i] - sigma_star[i]) for i in active)
            node_deviations.append(dev)

        total_nodes += 1
        if node_deviations:
            node_max_dev = max(node_deviations)
            max_deviation = max(max_deviation, node_max_dev)
            if node_max_dev > AGREE_TOL:
                n_disagreements += 1

        rows.append({
            "t": t, "G": str(G),
            "n_active": len(active),
            "n_inits": len(inits),
            "n_non_converged": node_non_conv,
            "max_deviation": round(max(node_deviations), 12) if node_deviations else None,
            "unique": (max(node_deviations) <= AGREE_TOL) if node_deviations else None,
        })

    if t % 2 == 0:
        print(f"  Processed period t={t}...", flush=True)

df_nodes = pd.DataFrame(rows)
verdict_12 = "PASS" if n_disagreements == 0 and n_non_converged == 0 else "FAIL"

print(f"\n  Total active nodes tested : {total_nodes}")
print(f"  Non-converged initialisations : {n_non_converged}")
print(f"  Nodes with disagreement (>{AGREE_TOL}) : {n_disagreements}")
print(f"  Max deviation across all nodes : {max_deviation:.2e}")
print(f"\n  Verdict : {verdict_12}")


# ── TEST 3: Eigenvalue check on QRE Jacobian ─────────────────────────────────
print(f"\n{SEP}\n  TEST 3 — EIGENVALUE STABILITY CHECK\n{SEP}")
print("  Checking Jacobian of logit best-response map at each σ*\n", flush=True)

def compute_qre_jacobian(t, G, active, W, V_next, sigma_star, params):
    """
    Jacobian of F_i(σ) = qre_probability(qa_i(σ_{-i}), qd_i(σ_{-i}), λ_i)
    w.r.t. σ_j for j ≠ i. Since qa/qd of player i only depend on σ_{-i},
    the diagonal of J is zero; off-diagonals are ∂F_i/∂σ_j.

    ∂F_i/∂σ_j = F_i(1-F_i) · λ_i · (∂qa_i/∂σ_j - ∂qd_i/∂σ_j)

    ∂qa_i/∂σ_j = Σ_{oa} [∂prob/∂σ_j] · V[t+1][G_adopt][i]
    where ∂prob/∂σ_j = prob/σ_j if oa_j=1 else -prob/(1-σ_j)
    """
    n  = len(active)
    idx = {i: k for k, i in enumerate(active)}
    J  = np.zeros((n, n))

    for i in active:
        p      = PLAYERS[i]
        lam_i  = params.lam[p]
        sig_i  = sigma_star[i]
        others = [j for j in active if j != i]
        combos = list(product([0, 1], repeat=len(others)))

        for j in active:
            if j == i:
                continue  # diagonal stays 0

            dqa_dsj = dqd_dsj = 0.0
            for oa in combos:
                prob = 1.0
                for k_idx, k in enumerate(others):
                    prob *= sigma_star[k] if oa[k_idx] == 1 else (1 - sigma_star[k])

                # ∂prob/∂σ_j
                j_pos = others.index(j)
                if oa[j_pos] == 1:
                    dprod = prob / sigma_star[j] if sigma_star[j] > 1e-10 else 0.0
                else:
                    dprod = -prob / (1 - sigma_star[j]) if (1 - sigma_star[j]) > 1e-10 else 0.0

                G_adopt = list(G); G_adopt[i] = 1
                G_delay = list(G)
                for k_idx, k in enumerate(others):
                    if oa[k_idx] == 1:
                        G_adopt[k] = 1
                        G_delay[k] = 1

                dqa_dsj += dprod * V_next[tuple(G_adopt)][i]
                dqd_dsj += dprod * V_next[tuple(G_delay)][i]

            # ∂F_i/∂σ_j = F_i(1-F_i) · λ_i · (∂qa/∂σ_j - ∂qd/∂σ_j)
            J[idx[i], idx[j]] = sig_i * (1 - sig_i) * lam_i * (dqa_dsj - dqd_dsj)

    return J


max_eigenvalue = 0.0
n_unstable     = 0
eig_rows       = []

for t in range(params.T, 0, -1):
    for G in ALL_G:
        active = get_active(G)
        if not active or len(active) < 2:
            continue

        W          = compute_W(G, params)
        V_next     = V[t + 1]
        sigma_star = {i: sigma_base[t][G][i] for i in active}

        J   = compute_qre_jacobian(t, G, active, W, V_next, sigma_star, params)
        eigs = np.abs(np.linalg.eigvals(J))
        spec_rad = float(eigs.max())
        max_eigenvalue = max(max_eigenvalue, spec_rad)

        stable = spec_rad < 1.0
        if not stable:
            n_unstable += 1

        eig_rows.append({
            "t": t, "G": str(G),
            "n_active": len(active),
            "spectral_radius": round(spec_rad, 6),
            "stable": stable,
        })

df_eigs = pd.DataFrame(eig_rows)
verdict_3 = "PASS" if n_unstable == 0 else "FAIL"

print(f"  Nodes with ≥2 active players : {len(df_eigs)}")
print(f"  Max spectral radius           : {max_eigenvalue:.6f}  (need < 1.0)")
print(f"  Unstable nodes (ρ ≥ 1)        : {n_unstable}")
print(f"\n  Verdict : {verdict_3}")

# Distribution of spectral radii
sr_vals = df_eigs["spectral_radius"].values
print(f"\n  Spectral radius distribution:")
print(f"    min={sr_vals.min():.4f}  median={np.median(sr_vals):.4f}  "
      f"max={sr_vals.max():.4f}  p99={np.percentile(sr_vals,99):.4f}")


# ── Save outputs ──────────────────────────────────────────────────────────────
df_nodes["test"] = "multistart"
df_eigs["test"]  = "eigenvalue"
df_out = pd.concat([
    df_nodes[["test","t","G","n_active","max_deviation","unique"]],
    df_eigs[["test","t","G","n_active","spectral_radius","stable"]],
], ignore_index=True)
csv_path = os.path.join(ROOT_DIR, "results", "equilibrium_uniqueness.csv")
df_out.to_csv(csv_path, index=False)
print(f"\n  CSV: {csv_path}")


# ── LaTeX summary table ───────────────────────────────────────────────────────
n_active_nodes = len(df_nodes)
n_eig_nodes    = len(df_eigs)

tex = r"""\begin{table}[ht]
\centering
\caption{QRE Equilibrium Uniqueness Tests at SMM-calibrated parameters.
Test 1--2: fixed-point iteration from """ + str(N_RANDOM_INITS) + r""" random initialisations
plus simplex corners at every active game-tree node.
Test 3: spectral radius of the logit best-response Jacobian at $\sigma^*$.}
\label{tab:equilibrium_uniqueness}
\begin{tabular}{llcl}
\toprule
Test & Criterion & Result & Verdict \\
\midrule
1--2: Multiple initialisations &
  Max deviation from $\sigma^*_{0.5}$ across all nodes $< 10^{-5}$ &
  """ + f"{max_deviation:.2e}" + r""" &
  \textbf{""" + verdict_12 + r"""} \\
  & Non-converged initialisations & """ + str(n_non_converged) + r""" & \\
  & Nodes tested & """ + str(total_nodes) + r""" & \\
\midrule
3: Eigenvalue stability &
  Spectral radius $\rho(J_F) < 1$ at all nodes &
  $\rho_{\max} = """ + f"{max_eigenvalue:.4f}" + r"""$ &
  \textbf{""" + verdict_3 + r"""} \\
  & Nodes with $\geq 2$ active players & """ + str(n_eig_nodes) + r""" & \\
\bottomrule
\end{tabular}
\end{table}"""

tex_path = os.path.join(ROOT_DIR, "results", "dissertation", "table_equilibrium_uniqueness.tex")
with open(tex_path, "w") as f:
    f.write(tex)
print(f"  LaTeX: {tex_path}")


# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  EQUILIBRIUM UNIQUENESS — SUMMARY")
print(f"{SEP}")
print(f"  Test 1 & 2 (multi-init)  : {verdict_12}  |  max dev = {max_deviation:.2e}")
print(f"  Test 3 (eigenvalue)      : {verdict_3}  |  max ρ(J) = {max_eigenvalue:.4f}")
overall = "PASS" if verdict_12 == "PASS" and verdict_3 == "PASS" else "FAIL"
print(f"\n  Overall uniqueness verdict : {overall}")
print(f"{SEP}\n")
