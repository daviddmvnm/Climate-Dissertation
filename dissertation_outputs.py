"""
dissertation_outputs.py
───────────────────────
Generates dissertation tables (LaTeX .tex) and figures (PNG).
Fully updated for heterogeneous rationality (lambda_i) and SMM-calibrated results.
"""

import os, warnings
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
OUT = "results/dissertation"
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
bloc_data = build_bloc_data(data_dir=".")
raw       = bloc_data.set_index("bloc")
weights   = {}
for bloc in PLAYERS:
    weights[bloc] = 0.5*raw.loc[bloc,"emission_share"] + 0.5*raw.loc[bloc,"gdp_share"]
total_w = sum(weights.values())
weights = {k: v/total_w for k,v in weights.items()}

# ── Load result CSVs ───────────────────────────────────────────────
# These must exist from your previous run_analysis.py execution
try:
    df_summary  = pd.read_csv("results/baseline_summary.csv")
    df_sweeps   = pd.read_csv("results/baseline_sweeps.csv")
    df_crossings = pd.read_csv("results/sweep_crossings.csv")
    df_gsa_corr = pd.read_csv("results/gsa_baseline_correlations.csv")
    df_gsa_samp = pd.read_csv("results/gsa_baseline_samples.csv")
except FileNotFoundError as e:
    print(f"Error: Missing results files. Run run_analysis.py first.\n{e}")
    exit()

# Re-solve for distribution plots
from run_analysis import build_params
b = SMM_BASELINE
params = build_params(raw, weights, ac=b["ac"], ad=b["ad"], ap=b["ap"], ab=b["ab"], lam=b["lam"])
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
        if row == "MIDRULE":
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
}

param_order = [
    "theta", "cost_CN", "cost_US", "delta_CN", "delta_US", 
    "lam_CN", "lam_US", "gamma", "kappa", "eta",
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
# FIGURE 6 — BIMODAL W DISTRIBUTION
# ══════════════════════════════════════════════════════════════════
print("Generating Figure 6: Bimodal W Distribution...")

fig, ax = plt.subplots(figsize=(7, 4.5))
final_W = W_paths[:, -1]
ax.hist(final_W[final_W < params.theta], bins=15, color="#2980B9", alpha=0.6, label="Coordination Failure")
ax.hist(final_W[final_W >= params.theta], bins=15, color="#C0392B", alpha=0.6, label="Coordination Success")
ax.axvline(params.theta, color="black", ls="--", label=f"Threshold θ={params.theta}")
ax.set_xlabel("Final Coalition Weight (W)")
ax.set_ylabel("Frequency")
ax.set_title("The Climate Tipping Point: Bimodal Outcomes")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_w_distribution.png"))
plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# FIGURE 8 — GSA HEATMAP
# ══════════════════════════════════════════════════════════════════
print("Generating Figure 8: GSA Heatmap...")

heat_matrix = df_gsa_corr.pivot(index="parameter", columns="outcome", values="rho")
heat_matrix = heat_matrix.reindex(index=param_order, columns=outcomes).dropna(how='all')

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(heat_matrix, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
plt.colorbar(im, label="Spearman ρ")

ax.set_xticks(range(len(outcomes)))
ax.set_xticklabels(outcome_headers, rotation=45, ha="right")
ax.set_yticks(range(len(heat_matrix.index)))
ax.set_yticklabels([param_latex.get(p, p) for p in heat_matrix.index])

# Annotate with stars
pval_matrix = df_gsa_corr.pivot(index="parameter", columns="outcome", values="pvalue").reindex(index=heat_matrix.index, columns=outcomes)
for i in range(len(heat_matrix.index)):
    for j in range(len(outcomes)):
        rho = heat_matrix.iloc[i, j]
        pv  = pval_matrix.iloc[i, j]
        star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        ax.text(j, i, f"{rho:+.2f}{star}", ha="center", va="center", fontsize=8, 
                color="white" if abs(rho) > 0.3 else "black")

ax.set_title("Global Sensitivity Analysis: Multi-Outcome Correlation")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_gsa_heatmap.png"))
plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# FIGURE — PERIOD-1 ADOPTION PROBABILITIES BY BLOC
# ══════════════════════════════════════════════════════════════════
print("Generating Figure: Period-1 adoption probabilities...")

fig, ax = plt.subplots(figsize=(6, 4))
blocs = PLAYERS
p1_probs = [sigma[1][G0][idx[b]] for b in blocs]
colors   = [BLOC_COLOURS[b] for b in blocs]
bars = ax.bar([BLOC_LABELS[b] for b in blocs], p1_probs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, p1_probs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("P(adopt in period 1)")
ax.set_title("Period-1 Equilibrium Adoption Probabilities — SMM Baseline")
ax.set_ylim(0, min(1.0, max(p1_probs) * 1.25))
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_period1_sigmas.png"))
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
# FIGURE — ADOPTION TIMING DISTRIBUTION BY BLOC
# ══════════════════════════════════════════════════════════════════
print("Generating Figure: Adoption timing distribution...")

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
for ax, bloc in zip(axes, PLAYERS):
    bi = idx[bloc]
    times = adopt_time[:, bi]
    finite = times[times < np.inf].astype(int)
    never  = (times == np.inf).sum()
    counts = np.zeros(T + 1)
    for t in finite:
        if 1 <= t <= T:
            counts[t] += 1
    ax.bar(range(1, T + 1), counts[1:], color=BLOC_COLOURS[bloc], alpha=0.8, edgecolor="white")
    ax.bar([T + 1], [never], color=BLOC_COLOURS[bloc], alpha=0.35, edgecolor="white", hatch="//")
    ax.set_title(BLOC_LABELS[bloc], fontsize=10)
    ax.set_xlabel("Period")
    ax.set_xticks(list(range(1, T + 1)) + [T + 1])
    ax.set_xticklabels([str(t) for t in range(1, T + 1)] + ["Never"], fontsize=8)
    if bloc == PLAYERS[0]:
        ax.set_ylabel("Count (n=1000 runs)")
fig.suptitle("Adoption Timing by Bloc — SMM Baseline", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_adoption_timing.png"))
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
alpha_rows = [
    [r"$\alpha_c$", "Transition cost scaling",    "3.50", "2.00", "2.00", "3.00", f"\\textbf{{{b['ac']:.4f}}}"],
    [r"$\alpha_d$", "Climate damage scaling",      "0.30", "0.15", "0.30", "0.25", f"\\textbf{{{b['ad']:.4f}}}"],
    [r"$\alpha_p$", "Political pressure scaling",  "0.50", "2.50", "0.50", "1.50", f"\\textbf{{{b['ap']:.4f}}}"],
    [r"$\alpha_b$", "Coordination benefit scaling","3.00", "1.00", "1.00", "2.00", f"\\textbf{{{b['ab']:.4f}}}"],
]
tex = tex_table(
    caption=r"SMM Alpha Parameter Estimates vs.\ Scenario Starting Points",
    label="smm_alphas",
    header=["Param.", "Description", "Scen. A", "Scen. B", "Scen. C", "Default", "SMM Est."],
    rows=alpha_rows,
    note=(r"SMM estimates from Nelder-Mead optimisation, best of 4 starting points. "
          r"Fixed parameters: $\lambda=1.54$, $\eta=15.0$, $\kappa=0.05$, "
          r"$\delta_{EU}=0.85$, $\delta_{US}=0.75$, $\delta_{CN}=0.80$, $\delta_{RoW}=0.65$."),
)
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
# COMPILE TABLE .tex → PNG  (pdflatex → pdfcrop → pdftoppm)
# ══════════════════════════════════════════════════════════════════
import subprocess, tempfile, shutil

def compile_table_png(tex_fragment_path, out_png_path):
    """Wrap a .tex table fragment, compile to PDF, crop, convert to PNG."""
    frag = open(tex_fragment_path).read()
    wrapper = (
        r"\documentclass{article}" + "\n"
        r"\usepackage{booktabs,amsmath,amssymb}" + "\n"
        r"\usepackage[margin=1cm]{geometry}" + "\n"
        r"\pagestyle{empty}" + "\n"
        r"\begin{document}" + "\n"
        + frag + "\n"
        r"\end{document}"
    )
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "table.tex")
        pdf = os.path.join(td, "table.pdf")
        crop = os.path.join(td, "table_crop.pdf")
        with open(src, "w") as f:
            f.write(wrapper)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", td, src],
                           capture_output=True)
        if not os.path.exists(pdf):
            print(f"  ✗ pdflatex failed for {tex_fragment_path}")
            print(r.stdout.decode()[-500:])
            return
        subprocess.run(["pdfcrop", pdf, crop], capture_output=True)
        src_pdf = crop if os.path.exists(crop) else pdf
        result = subprocess.run(
            ["pdftoppm", "-r", "200", "-png", src_pdf, os.path.join(td, "out")],
            capture_output=True)
        candidates = [f for f in os.listdir(td) if f.startswith("out") and f.endswith(".png")]
        if candidates:
            shutil.copy(os.path.join(td, sorted(candidates)[0]), out_png_path)
            print(f"  ✓ {out_png_path}")
        else:
            print(f"  ✗ No PNG produced for {tex_fragment_path}")

print("Compiling table PNGs via pdflatex...")
for tbl in ["table_gsa_correlations", "table_smm_alphas",
            "table_sweep_crossings", "table_baseline_diagnostics"]:
    tex_path = os.path.join(OUT, f"{tbl}.tex")
    png_path = os.path.join(OUT, f"{tbl}.png")
    if os.path.exists(tex_path):
        compile_table_png(tex_path, png_path)
    else:
        print(f"  ⚠ {tex_path} not found, skipping")

print(f"\nDone. All dissertation outputs generated in {OUT}/")