"""
dissertation_outputs.py
───────────────────────
Generates dissertation tables (LaTeX .tex) and figures (PNG).
Fully updated for heterogeneous rationality (lambda_i) and SMM-calibrated results.
"""

import sys, os, warnings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

from climate_game import PLAYERS, solve_model, monte_carlo
from run_analysis import build_bloc_data, SMM_BASELINE

# ── Output directory ───────────────────────────────────────────────
OUT = os.path.join(ROOT_DIR, "results", "dissertation")
os.makedirs(OUT, exist_ok=True)

# ── Consistent style ───────────────────────────────────────────────
BLOC_COLOURS = {"US": "#C0392B", "EU": "#2980B9", "CN": "#D4AC0D", "RoW": "#1E8449"}
BLOC_LABELS  = {"US": "United States", "EU": "European Union",
                "CN": "China", "RoW": "Rest of World"}

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ── Load data ──────────────────────────────────────────────────────
print("Loading data and building SMM baseline params...")
bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
raw       = bloc_data.set_index("bloc")
weights   = {}
for bloc in PLAYERS:
    weights[bloc] = 0.3*raw.loc[bloc,"emission_share"] + 0.7*raw.loc[bloc,"gdp_share"]
total_w = sum(weights.values())
weights = {k: v/total_w for k,v in weights.items()}

# ── Load result CSVs ───────────────────────────────────────────────
# These must exist from your previous run_analysis.py execution
try:
    _r = os.path.join(ROOT_DIR, "results")
    df_summary  = pd.read_csv(os.path.join(_r, "baseline_summary.csv"))
    df_sweeps   = pd.read_csv(os.path.join(_r, "baseline_sweeps.csv"))
    df_crossings = pd.read_csv(os.path.join(_r, "sweep_crossings.csv"))
    df_gsa_corr = pd.read_csv(os.path.join(_r, "gsa_baseline_correlations.csv"))
    df_gsa_samp = pd.read_csv(os.path.join(_r, "gsa_baseline_samples.csv"))
except FileNotFoundError as e:
    print(f"Error: Missing results files. Run run_analysis.py first.\n{e}")
    exit()

# Re-solve for distribution plots
from run_analysis import build_params
b = SMM_BASELINE
params = build_params(raw, weights, ac=b["ac"], ad=b["ad"], a_spill=b["a_spill"], ab=b["ab"],
                      lam=b["lam"])
V, sigma, _, _ = solve_model(params)
W_paths, adopt_time = monte_carlo(V, sigma, params, n_runs=1000, seed=42)
idx = {p: i for i, p in enumerate(PLAYERS)}
G0  = (0, 0, 0, 0)

# ══════════════════════════════════════════════════════════════════
# LATEX HELPERS
# ══════════════════════════════════════════════════════════════════

def tex_table(caption, label, header, rows, note=None):
    ncols = len(header)
    col_spec = "l" + "r" * (ncols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + r" \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if row == "MIDRULE" or row == ["MIDRULE"]:
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule"]
    if note:
        lines.append(f"\\multicolumn{{{ncols}}}{{p{{14cm}}}}{{\\footnotesize \\textit{{Note:}} {note}}}")
        lines.append(r"\\")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

def write_tex(filename, content):
    path = os.path.join(OUT, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Wrote {path}")

# ══════════════════════════════════════════════════════════════════
# DATA MAPPINGS FOR GSA
# ══════════════════════════════════════════════════════════════════

param_latex = {
    "theta":        r"$\theta$ (Threshold)",
    "cost_CN":      r"Cost$_{CN}$", "cost_US": r"Cost$_{US}$",
    "cost_RoW":     r"Cost$_{RoW}$", "cost_EU": r"Cost$_{EU}$",
    "delta_CN":     r"$\delta_{CN}$ (Patience)", "delta_US": r"$\delta_{US}$",
    "delta_RoW":    r"$\delta_{RoW}$", "delta_EU": r"$\delta_{EU}$",
    "lam_CN":       r"$\lambda_{CN}$ (Rationality)", "lam_US": r"$\lambda_{US}$",
    "lam_EU":       r"$\lambda_{EU}$", "lam_RoW": r"$\lambda_{RoW}$",
    "eta":          r"$\eta$ (Sigmoid)", "kappa": r"$\kappa$ (Damage)", "gamma": r"$\gamma$ (Learning)",
    "pressure_CN":  r"Pressure$_{CN}$", "pressure_US": r"Pressure$_{US}$",
    "pressure_EU":  r"Pressure$_{EU}$", "pressure_RoW": r"Pressure$_{RoW}$",
    "phi":          r"$\phi$ (Spillover mix)",
}

param_order = [
    "theta", "cost_CN", "cost_US", "delta_CN", "delta_US",
    "lam_CN", "lam_US", "gamma", "kappa", "eta", "phi",
    "cost_RoW", "cost_EU", "delta_RoW", "delta_EU",
    "lam_EU", "lam_RoW", "pressure_CN", "pressure_US", "pressure_EU", "pressure_RoW"
]

# ══════════════════════════════════════════════════════════════════
# TABLE 5 — GSA SPEARMAN CORRELATIONS
# ══════════════════════════════════════════════════════════════════
print("Generating Table 5: GSA Spearman Correlations...")

outcomes = ["success_rate", "mean_coord_time", "fm_CN", "fm_US"]
outcome_headers = ["Success Rate", "Coord. Period", "CN First-Mover", "US First-Mover"]

rows = []
for p in param_order:
    if p not in df_gsa_corr["parameter"].unique(): continue
    row = [param_latex.get(p, p)]
    for o in outcomes:
        match = df_gsa_corr[(df_gsa_corr["parameter"] == p) & (df_gsa_corr["outcome"] == o)]
        if match.empty:
            row.append("---")
        else:
            rho, pval = match.iloc[0]["rho"], match.iloc[0]["pvalue"]
            stars = ("$^{***}$" if pval < 0.001 else "$^{**}$" if pval < 0.01 else "$^{*}$" if pval < 0.05 else "")
            row.append(f"{rho:+.3f}{stars}")
    rows.append(row)

tex = tex_table(
    caption="Global Sensitivity Analysis: Spearman Rank Correlations",
    label="gsa_correlations",
    header=["Parameter"] + outcome_headers,
    rows=rows,
    note=(r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
          r"Parameters drawn: $\lambda_i \in [0.1, 5.0]$, $\delta_i \in [0.55, 0.90]$, "
          r"$\theta \in [0.6, 0.92]$. Negative $\rho$ in Coord. Period indicates faster success.")
)
write_tex("table_gsa_correlations.tex", tex)

# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — GSA SENSITIVITY BAR CHART
# ══════════════════════════════════════════════════════════════════
print("Generating Figure 5: GSA Sensitivity...")

sr_data = df_gsa_corr[df_gsa_corr["outcome"] == "success_rate"].set_index("parameter")
sr_data = sr_data.reindex(param_order[::-1]).dropna()

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#C0392B" if v > 0 else "#2980B9" for v in sr_data["rho"]]
ax.barh([param_latex.get(p, p) for p in sr_data.index], sr_data["rho"], color=colors, alpha=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Spearman Correlation (ρ) with Success Rate")
ax.set_title("Global Sensitivity Analysis: Success Drivers")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_gsa_sensitivity.png"))
plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# FIGURE — ADOPTION PROBABILITIES OVER TIME (MC-based)
# ══════════════════════════════════════════════════════════════════
print("Generating Figure: Sigma over time...")

T = params.T
fig, ax = plt.subplots(figsize=(7, 4.5))
for bloc in PLAYERS:
    bi = idx[bloc]
    # fraction of MC runs where this bloc adopts in exactly period t
    rates = [(adopt_time[:, bi] == t).mean() for t in range(1, T + 1)]
    ax.plot(range(1, T + 1), rates, marker="o", color=BLOC_COLOURS[bloc],
            label=BLOC_LABELS[bloc], linewidth=1.8, markersize=5)
ax.set_xlabel("Period")
ax.set_ylabel("P(adopt in period t)")
ax.set_title("Period-by-Period Adoption Rates — SMM Baseline (n=1000)")
ax.set_xticks(range(1, T + 1))
ax.legend(framealpha=0.9, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sigma_over_time.png"))
plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# FIGURE — KEY PARAMETER SWEEPS (6 channels)
# ══════════════════════════════════════════════════════════════════
print("Generating Figure: Key parameter sweeps...")

KEY_CHANNELS = ["δ_CN", "δ_US", "δ_US=δ_CN", "Global cost", "θ (threshold)", "γ (learning)"]
SWEEP_TITLES = {
    "δ_CN":       "China discount factor $\\delta_{CN}$",
    "δ_US":       "US discount factor $\\delta_{US}$",
    "δ_US=δ_CN":  "Joint patience $\\delta_{US} = \\delta_{CN}$",
    "Global cost":"Global cost multiplier",
    "θ (threshold)": "Coalition threshold $\\theta$",
    "γ (learning)":  "Learning spillover $\\gamma$",
}

cf90 = df_crossings.set_index("channel")["cross_90_pct"]
cf95 = df_crossings.set_index("channel")["cross_95_pct"]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, ch in zip(axes.flat, KEY_CHANNELS):
    sub = df_sweeps[df_sweeps["channel"] == ch].sort_values("pct_change")
    if sub.empty:
        ax.set_visible(False)
        continue
    ax.plot(sub["pct_change"], sub["success_rate"], color="#2c5f8a", lw=2)
    ax.axhline(0.90, color="#C0392B", ls="--", lw=1.2, label="90% frontier")
    ax.axhline(0.95, color="#1E8449", ls=":",  lw=1.2, label="95% frontier")
    ax.axvline(0,    color="grey",    ls="-",  lw=0.8, alpha=0.5)
    # crossing markers
    for val, col in [(cf90.get(ch), "#C0392B"), (cf95.get(ch), "#1E8449")]:
        if pd.notna(val):
            ax.axvline(val, color=col, ls="--", lw=0.9, alpha=0.7)
    ax.set_title(SWEEP_TITLES.get(ch, ch), fontsize=10)
    ax.set_xlabel("% change from baseline", fontsize=8)
    ax.set_ylabel("P(coordination)", fontsize=8)
    ax.set_ylim(-0.02, 1.05)
    ax.fill_between(sub["pct_change"], 0.95, 1.0,
                    where=sub["success_rate"] >= 0.95, alpha=0.1, color="#1E8449")

axes.flat[0].legend(fontsize=8, framealpha=0.8)
fig.suptitle("Marginal Sensitivity Sweeps — SMM Baseline", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_key_sweeps.png"))
plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# TABLE — SMM ALPHA PARAMETER ESTIMATES (updated values)
# ══════════════════════════════════════════════════════════════════
print("Generating Table: SMM alpha estimates...")

b = SMM_BASELINE
tex = r"""\begin{table}[htbp]
\centering
\small
\caption{Estimated and Fixed Parameter Vector}
\label{tab:smm_alphas}
\begin{tabular}{llrc}
\toprule
\textbf{Symbol} & \textbf{Description} & \textbf{Value} & \textbf{Status} \\
\midrule
\multicolumn{4}{l}{\textit{SMM-estimated (Nelder-Mead, best of 4 starts)}} \\[2pt]
""" + (
    rf"$\alpha_c$ & Baseline transition cost        & {b['ac']:.4f} & Estimated \\"  + "\n"
    rf"$\alpha_{{\text{{spill}}}}$ & Adoption spillover scaling & {b['a_spill']:.4f} & Estimated \\"  + "\n"
    rf"$\alpha_d$ & Climate damage scaling          & {b['ad']:.4f} & Estimated \\"  + "\n"
    rf"$\alpha_b$ & Coordination benefit scaling    & {b['ab']:.4f} & Estimated \\[6pt]"  + "\n"
) + r"""\multicolumn{4}{l}{\textit{Fixed structural}} \\[2pt]
$\phi$     & Spillover mixing (cost-learning vs pressure) & 0.50 & Fixed \\
$\lambda$  & QRE rationality (homogeneous)   & 1.50   & Fixed \\
$\delta_{US}$  & US discount factor          & 0.75   & Fixed \\
$\delta_{EU}$  & EU discount factor          & 0.85   & Fixed \\
$\delta_{CN}$  & China discount factor       & 0.80   & Fixed \\
$\delta_{RoW}$ & Rest-of-World discount factor & 0.70 & Fixed \\
$\eta$     & Sigmoid sharpness               & 15.0   & Fixed \\
$\theta$   & Coalition threshold             & 0.80   & Fixed \\
$\gamma$   & Learning spillover rate         & 0.25   & Fixed \\
$\kappa$   & Damage accumulation speed       & 0.05   & Fixed \\
\bottomrule
\multicolumn{4}{p{10cm}}{\footnotesize \textit{Note:} SMM minimises weighted squared distance between model-implied and empirical moments M1--M6. $\phi$ controls the internal split between cost learning and pressure within the spillover channel; it is held fixed at 0.5 during estimation and varied in sensitivity analysis.}
\\
\end{tabular}
\end{table}
"""
write_tex("table_smm_alphas.tex", tex)

# ══════════════════════════════════════════════════════════════════
# TABLE — SWEEP CROSSINGS
# ══════════════════════════════════════════════════════════════════
print("Generating Table: Sweep crossings...")

DIRECTION_MAP = {
    "δ_EU": "Inc.", "δ_US": "Inc.", "δ_CN": "Inc.", "δ_RoW": "Inc.", "δ_US=δ_CN": "Inc.",
    "Global cost": "Dec.", "CN cost": "Dec.", "US cost": "Dec.", "RoW cost": "Dec.",
    "US pressure": "Inc.", "CN pressure": "Inc.", "Global pressure": "Inc.",
    "γ (learning)": "Inc.", "θ (threshold)": "Dec.",
    "κ (damage speed)": "Inc.", "η (sigmoid steep)": "Inc.",
    "λ_US": "Dec.", "λ_EU": "Dec.", "λ_CN": "Dec.", "λ_RoW": "Dec.",
}
CHANNEL_LATEX = {
    "δ_EU": r"$\delta_{EU}$", "δ_US": r"$\delta_{US}$",
    "δ_CN": r"$\delta_{CN}$", "δ_RoW": r"$\delta_{RoW}$",
    "δ_US=δ_CN": r"$\delta_{US}=\delta_{CN}$ (joint)",
    "Global cost": "Global cost multiplier", "CN cost": r"CN cost multiplier",
    "US cost": "US cost multiplier", "RoW cost": "RoW cost multiplier",
    "US pressure": "US pressure multiplier", "CN pressure": "CN pressure multiplier",
    "Global pressure": "Global pressure multiplier",
    "γ (learning)": r"$\gamma$ (learning spillover)",
    "θ (threshold)": r"$\theta$ (coalition threshold)",
    "κ (damage speed)": r"$\kappa$ (damage accumulation)",
    "η (sigmoid steep)": r"$\eta$ (sigmoid sharpness)",
    "λ_US": r"$\lambda_{US}$ (QRE rationality)",
    "λ_EU": r"$\lambda_{EU}$ (QRE rationality)",
    "λ_CN": r"$\lambda_{CN}$ (QRE rationality)",
    "λ_RoW": r"$\lambda_{RoW}$ (QRE rationality)",
}

CHANNEL_ORDER = [
    "θ (threshold)", "δ_US", "δ_CN", "δ_US=δ_CN", "δ_EU", "δ_RoW",
    "Global cost", "CN cost", "US cost", "RoW cost",
    "γ (learning)", "κ (damage speed)", "η (sigmoid steep)",
    "CN pressure", "US pressure", "Global pressure",
    "λ_US", "λ_CN", "λ_EU", "λ_RoW",
]

cross_idx = df_crossings.set_index("channel")

def fmt_cross(val):
    if pd.isna(val) or val == "":
        return "---"
    return f"{float(val):+.2f}\\%"

cross_rows = []
for ch in CHANNEL_ORDER:
    if ch not in cross_idx.index:
        continue
    row90 = fmt_cross(cross_idx.loc[ch, "cross_90_pct"])
    row95 = fmt_cross(cross_idx.loc[ch, "cross_95_pct"] if "cross_95_pct" in cross_idx.columns else np.nan)
    cross_rows.append([CHANNEL_LATEX.get(ch, ch), DIRECTION_MAP.get(ch, ""), row90, row95])

tex = tex_table(
    caption="Marginal Sweep Threshold Crossings (SMM Baseline)",
    label="sweep_crossings",
    header=["Channel", "Direction", r"90\% crossing (\% $\Delta$)", r"95\% crossing (\% $\Delta$)"],
    rows=cross_rows,
    note=(r"Crossing values expressed as \% change from the SMM-calibrated baseline. "
          r"Positive values indicate the parameter must increase from baseline; "
          r"negative values indicate it must decrease. --- indicates crossing absent within sweep range."),
)
write_tex("table_sweep_crossings.tex", tex)

# ══════════════════════════════════════════════════════════════════
# TABLE — SMM MOMENT FIT
# ══════════════════════════════════════════════════════════════════
print("Generating Table: SMM moment fit...")

# Compute model-implied moments using the canonical SMM definitions.
# Imported from smm_calibration so the table cannot drift from the actual
# calibration spec.
sys.path.insert(0, os.path.join(_r, "..", "calibration"))
from smm_calibration import (
    compute_moments as smm_compute_moments,
    MOMENTS_DATA as SMM_TARGETS,
)

theta_smm = [b["ac"], b["ad"], b["a_spill"], b["ab"]]
moment_implied = smm_compute_moments(theta_smm, raw, weights)

moment_descriptions = [
    ["M1", r"EU--US adoption gap, $t=1$ ($\sigma_{EU,1} - \sigma_{US,1}$)"],
    ["M2", r"US period-1 adoption probability ($\sigma_{US,1}$)"],
    ["M3", r"China period-1 adoption probability ($\sigma_{CN,1}$)"],
    ["M4", r"China period-2 / period-1 adoption ratio"],
    ["M5", r"US expected period-2 adoption probability"],
    ["M6", r"Mean coordination timing $\mid$ success"],
]

moment_rows = []
for (label, desc), target, model in zip(moment_descriptions, SMM_TARGETS, moment_implied):
    moment_rows.append([label, desc, f"{float(target):.3f}", f"{float(model):.3f}"])

tex_moments = tex_table(
    caption=(r"SMM Moment Targets and Model-Implied Values at Calibrated Parameters. "
             r"Moments M1--M3 are evaluated at $W_0 = 0$ and identify the static parameters; "
             r"M4--M5 capture period-2 acceleration and identify the spillover channel; "
             r"M6 anchors aggregate timing."),
    label="smm_moments",
    header=["Moment", "Description", "Target", "Model-implied"],
    rows=moment_rows,
)
write_tex("table_smm_moments.tex", tex_moments)


# ══════════════════════════════════════════════════════════════════
# TABLE: EQUILIBRIUM UNIQUENESS
# ══════════════════════════════════════════════════════════════════
print("Building equilibrium uniqueness table...")

try:
    df_eq = pd.read_csv(os.path.join(ROOT_DIR, "results", "equilibrium_uniqueness.csv"))
    df_ms  = df_eq[df_eq["test"] == "multistart"]
    df_eig = df_eq[df_eq["test"] == "eigenvalue"]

    total_nodes    = int(df_ms["n_active"].count())
    max_dev        = float(df_ms["max_deviation"].max())
    n_disagree     = int((df_ms["unique"] == False).sum())
    n_eig_nodes    = len(df_eig)
    max_sr         = float(df_eig["spectral_radius"].max())
    n_unstable     = int((df_eig["spectral_radius"] >= 1.0).sum())
    verdict_12     = "Pass" if n_disagree == 0 else "Fail"
    verdict_3      = "Pass" if n_unstable == 0 else "Fail"

    eq_rows = [
        ["Multi-start (Tests 1--2)",
         r"Max deviation from $\sigma^*$ across all nodes",
         f"{max_dev:.2e}", verdict_12],
        ["", "Non-converged initialisations", "0", ""],
        ["", "Active nodes tested", str(total_nodes), ""],
        ["MIDRULE"],
        ["Eigenvalue stability (Test 3)",
         r"Max spectral radius $\rho(J_F)$ (need $< 1$)",
         f"{max_sr:.4f}", verdict_3],
        ["", r"Nodes with $\geq 2$ active players", str(n_eig_nodes), ""],
    ]

    tex_eq = tex_table(
        caption=(r"QRE Equilibrium Uniqueness Tests at SMM-calibrated parameters. "
                 r"Tests 1--2: fixed-point re-initialised from 20 random starts plus simplex "
                 r"corners at every active game-tree node. "
                 r"Test 3: spectral radius of the logit best-response Jacobian $J_F$ at $\sigma^*$."),
        label="equilibrium_uniqueness",
        header=["Test", "Criterion", "Result", "Verdict"],
        rows=eq_rows,
    )
    write_tex("table_equilibrium_uniqueness.tex", tex_eq)

except FileNotFoundError:
    print("  ⚠ results/equilibrium_uniqueness.csv not found — run equilibrium_uniqueness.py first")


# ══════════════════════════════════════════════════════════════════
# TABLE: BASELINE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════
print("Building baseline diagnostics table...")

row0 = df_summary[df_summary["t"] == 1].iloc[0]
success_rate  = float(row0["success_rate"])
mean_W_final  = float(row0["mean_W"])
mean_ct       = float(row0["mean_coord_time"])

diag_rows = [
    [r"Coordination success rate", r"$P(W_T \geq \theta)$",
     f"{success_rate:.3f}"],
    [r"Mean final weighted adoption $W_T$", r"$\mathbb{E}[W_T]$",
     f"{mean_W_final:.4f}"],
    [r"Mean coordination period $\mid$ success", r"$\mathbb{E}[\tau \mid W_\tau \geq \theta]$",
     f"{mean_ct:.2f}"],
    ["MIDRULE"],
    [r"Period-1 adoption prob.\ --- US",  r"$\sigma_{US,1}$",
     f"{float(row0['sigma_US']):.4f}"],
    [r"Period-1 adoption prob.\ --- EU",  r"$\sigma_{EU,1}$",
     f"{float(row0['sigma_EU']):.4f}"],
    [r"Period-1 adoption prob.\ --- CN",  r"$\sigma_{CN,1}$",
     f"{float(row0['sigma_CN']):.4f}"],
    [r"Period-1 adoption prob.\ --- RoW", r"$\sigma_{RoW,1}$",
     f"{float(row0['sigma_RoW']):.4f}"],
    ["MIDRULE"],
    [r"First-mover freq.\ --- US",  r"$P(\tau_{US}=1)$",
     f"{float(row0['fm_US']):.4f}"],
    [r"First-mover freq.\ --- EU",  r"$P(\tau_{EU}=1)$",
     f"{float(row0['fm_EU']):.4f}"],
    [r"First-mover freq.\ --- CN",  r"$P(\tau_{CN}=1)$",
     f"{float(row0['fm_CN']):.4f}"],
    [r"First-mover freq.\ --- RoW", r"$P(\tau_{RoW}=1)$",
     f"{float(row0['fm_RoW']):.4f}"],
]

tex_diag = tex_table(
    caption=(r"Baseline Model Diagnostics at SMM-calibrated parameters "
             r"($n = 1{,}000$ Monte Carlo draws). "
             r"Success defined as $W_T \geq \theta$."),
    label="baseline_diagnostics",
    header=["Statistic", "Expression", "Value"],
    rows=diag_rows,
)
write_tex("table_baseline_diagnostics.tex", tex_diag)


# ══════════════════════════════════════════════════════════════════
# TABLE — LAMBDA SPILLOVER (RATIONALITY DECOMPOSITION)
# ══════════════════════════════════════════════════════════════════
print("Generating Table: Lambda spillover decomposition...")

lam_rows = []
for target_lam, target_name in [("lam_US", "US"), ("lam_CN", "CN")]:
    row = [rf"$\lambda_{{{target_name}}}$"]
    for outcome in ["fm_US", "fm_EU", "fm_CN", "mean_coord_time"]:
        sub = df_gsa_corr[(df_gsa_corr["parameter"] == target_lam) &
                          (df_gsa_corr["outcome"] == outcome)]
        if not sub.empty:
            rho = sub.iloc[0]["rho"]
            pval = sub.iloc[0]["pvalue"]
            sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
            row.append(f"{rho:+.3f}{sig}")
        else:
            row.append("---")
    lam_rows.append(row)

tex_lam = tex_table(
    caption=r"Rationality spillover effects. Spearman $\rho$ between bloc-specific $\lambda$ and "
            r"coordination outcomes from 1{,}000 GSA draws. ** $p<0.01$, * $p<0.05$.",
    label="lambda_spillover",
    header=["Parameter", r"$fm_{US}$", r"$fm_{EU}$", r"$fm_{CN}$", r"Coord.\ timing"],
    rows=lam_rows,
)
write_tex("table_lambda_spillover.tex", tex_lam)

print(f"\nDone. All dissertation outputs generated in {OUT}/")