"""
run_analysis.py
───────────────
Standalone script: runs all three dissertation scenarios (A, B, C),
marginal parameter sweeps (15 channels), and global sensitivity analysis.

Outputs
-------
  results/
    scenario_{A,B,C}_summary.csv   — per-scenario MC statistics
    scenario_{A,B,C}_sweeps.csv    — full sweep curves
    gsa_{A,B,C}_samples.csv        — raw GSA draws + outcomes
    gsa_{A,B,C}_correlations.csv   — Spearman ρ tables
    figures/
      sweeps_{A,B,C}.png           — 5×3 sweep grid
      gsa_{A,B,C}.png              — GSA distribution panels
      comparison_sweeps.png        — cross-scenario sweep overlay
      gsa_correlations.png         — heatmap of Spearman ρ

Usage
-----
  python run_analysis.py [--fast]     # --fast uses fewer MC draws
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import argparse
import warnings
import time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from itertools import product as iproduct

from climate_game import (
    GameParams, PLAYERS, ALL_G,
    solve_model, monte_carlo, simulate_modal_path,
    compute_W,
)

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join(ROOT_DIR, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING + BLOC AGGREGATION  (mirrors calibrated_model_v5.ipynb Cells 2–3)
# ──────────────────────────────────────────────────────────────────────────────

EU27_CODES = [
    "FRA","DEU","ITA","ESP","NLD","POL","BEL","IRL","AUT","SWE",
    "DNK","FIN","PRT","CZE","ROU","GRC",
]

LOW_INCOME = [
    "AFG","BFA","BDI","CAF","TCD","COD","ERI","ETH","GMB","GIN",
    "GNB","PRK","LBR","MDG","MWI","MLI","MOZ","NER","RWA","SLE",
    "SOM","SSD","SDN","SYR","TGO","UGA","YEM",
]

LOWER_MIDDLE_INCOME = [
    "AGO","DZA","BGD","BEN","BTN","BOL","CPV","KHM","CMR","COM",
    "COG","CIV","DJI","EGY","SLV","SWZ","GHA","HTI","HND",
    "IDN","IRN","KEN","KIR","KGZ","LAO","LSO","MRT","FSM","MNG",
    "MAR","MMR","NPL","NIC","NGA","PAK","PNG","PHL","STP","SEN",
    "SLB","LKA","TZA","TJK","TLS","TUN","UKR","UZB","VUT","VNM",
    "ZMB","ZWE",
]

ROW_CODES = set(LOW_INCOME + LOWER_MIDDLE_INCOME)


def assign_bloc(code):
    if code == "USA": return "US"
    if code == "CHN": return "CN"
    if code in EU27_CODES: return "EU"
    if code in ROW_CODES: return "RoW"
    return None


def load_core(path):
    df = pd.read_csv(path)
    id_cols = ["Country Name", "Country Code", "Series Name", "Series Code"]
    year_cols = [c for c in df.columns if "YR" in c]
    df_long = df.melt(id_vars=id_cols, value_vars=year_cols,
                      var_name="year_raw", value_name="value")
    df_long["year"] = df_long["year_raw"].str.extract(r"(\d{4})").astype(int)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.drop(columns=["year_raw"])
    df_clean = df_long.pivot_table(
        index=["Country Name", "Country Code", "year"],
        columns="Series Name", values="value",
    ).reset_index()
    df_clean.columns.name = None
    return df_clean


def load_wb_indicator(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if "Country Name" in l)
    df = pd.read_csv(path, skiprows=header_idx, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all")
    id_cols = ["Country Name","Country Code","Indicator Name","Indicator Code"]
    year_cols = [c for c in df.columns if c.strip().isdigit()]
    df_long = df.melt(id_vars=id_cols, value_vars=year_cols,
                      var_name="year", value_name="value")
    df_long["year"] = df_long["year"].astype(int)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    return df_long


def build_bloc_data(data_dir="."):
    print("Loading CSV data...")
    core   = load_core(os.path.join(data_dir, "core_.csv"))
    gdp    = load_wb_indicator(os.path.join(data_dir, "gdp.csv"))
    gdp_pc = load_wb_indicator(os.path.join(data_dir, "gdp_pc.csv"))

    REF_YEAR = 2020
    col_map = {
        "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)": "emissions_mt",
        "Carbon intensity of GDP (kg CO2e per 2021 PPP $ of GDP)":  "carbon_intensity",
        "Trade (% of GDP)": "trade_pct",
        "Agriculture, forestry, and fishing, value added (% of GDP)": "ag_value_pct",
    }

    core_ref = core[core["year"] == REF_YEAR].copy().rename(columns=col_map)
    core_ref["bloc"] = core_ref["Country Code"].apply(assign_bloc)
    core_ref = core_ref[core_ref["bloc"].notna()]

    gdp_ref = gdp[gdp["year"] == REF_YEAR].rename(columns={"value": "gdp_usd"})
    gdp_ref["bloc"] = gdp_ref["Country Code"].apply(assign_bloc)
    gdp_ref = gdp_ref[gdp_ref["bloc"].notna()]
    core_ref = core_ref.merge(
        gdp_ref[["Country Code","gdp_usd"]].drop_duplicates(),
        on="Country Code", how="left")

    gdp_pc_ref = gdp_pc[gdp_pc["year"] == REF_YEAR].rename(columns={"value": "gdp_per_capita"})
    gdp_pc_ref["bloc"] = gdp_pc_ref["Country Code"].apply(assign_bloc)
    gdp_pc_ref = gdp_pc_ref[gdp_pc_ref["bloc"].notna()]
    core_ref = core_ref.merge(
        gdp_pc_ref[["Country Code","gdp_per_capita"]].drop_duplicates(),
        on="Country Code", how="left")

    additive = core_ref.groupby("bloc").agg(
        total_emissions=("emissions_mt","sum"),
        total_gdp=("gdp_usd","sum"),
    ).reset_index()

    def gdp_weighted_avg(group, col):
        mask = group[col].notna() & group["gdp_usd"].notna()
        s = group[mask]
        if len(s) == 0 or s["gdp_usd"].sum() == 0:
            return np.nan
        return (s[col] * s["gdp_usd"]).sum() / s["gdp_usd"].sum()

    rate_cols = ["carbon_intensity","trade_pct","ag_value_pct","gdp_per_capita"]
    weighted = core_ref.groupby("bloc").apply(
        lambda g: pd.Series({c: gdp_weighted_avg(g, c) for c in rate_cols})
    ).reset_index()

    bloc_data = additive.merge(weighted, on="bloc")
    bloc_data["emission_share"] = (bloc_data["total_emissions"]
                                   / bloc_data["total_emissions"].sum())
    bloc_data["gdp_share"] = bloc_data["total_gdp"] / bloc_data["total_gdp"].sum()
    print("  Bloc data built OK.")
    return bloc_data


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

REF_BLOC   = "EU"
EPSILON    = 0.3
THETA      = 0.80
ETA        = 15
LAMBDA     = 1.5
GAMMA      = 0.25
KAPPA      = 0.05
DISCOUNT   = {"US": 0.75, "EU": 0.85, "CN": 0.80, "RoW": 0.65}

# Original grid-search defaults (used as pre-SMM baseline reference)
BASE_ALPHAS = dict(ac=3.0, ad=0.25, ap=1.5, ab=2.0)

# SMM-calibrated baseline — loaded from saved calibration output
def _load_smm_baseline():
    import numpy as np
    from pathlib import Path
    _path = Path(__file__).parent.parent / "results" / "smm_best_theta.npy"
    if not _path.exists():
        raise FileNotFoundError(
            "results/smm_best_theta.npy not found — run smm_calibration.py first"
        )
    ac, ad, ap, ab = np.load(_path)
    return dict(ac=float(ac), ad=float(ad), ap=float(ap), ab=float(ab), lam=1.54)

SMM_BASELINE = _load_smm_baseline()

BLOC_COLOURS = {
    "US":  "#E63946",
    "EU":  "#457B9D",
    "CN":  "#E9C46A",
    "RoW": "#2A9D8F",
}


# ──────────────────────────────────────────────────────────────────────────────
# PARAM BUILDER  (mirrors Cell 4)
# ──────────────────────────────────────────────────────────────────────────────

def build_params(raw, weights, ac, ad, ap, ab, eps=EPSILON,
                 theta=THETA, eta=ETA, lam=LAMBDA,
                 gamma=GAMMA, kappa=KAPPA,
                 discount=None, lambda_map=None):
    if discount is None:
        discount = DISCOUNT
        
    # If no specific lambda_map is provided, we default all players to the base 'lam'
    if lambda_map is None:
        lambda_map = {bloc: lam for bloc in PLAYERS}

    max_gpc = raw["gdp_per_capita"].max()
    bl_intensity = ((raw["carbon_intensity"] * raw["total_gdp"]).sum()
                    / raw["total_gdp"].sum())
    
    ref_dev = raw.loc[REF_BLOC, "gdp_per_capita"] / max_gpc
    ref_eff_ci = (raw.loc[REF_BLOC, "carbon_intensity"] * ref_dev
                  + bl_intensity * (1 - ref_dev))
    ref_aff = (max_gpc / raw.loc[REF_BLOC, "gdp_per_capita"]) ** eps
    ref_composite = ref_eff_ci * ref_aff

    ag_ref = raw.loc[REF_BLOC, "ag_value_pct"]
    c, d, p = {}, {}, {}
    for bloc in PLAYERS:
        dev = raw.loc[bloc, "gdp_per_capita"] / max_gpc
        eff_ci = raw.loc[bloc, "carbon_intensity"] * dev + bl_intensity * (1 - dev)
        afford = (max_gpc / raw.loc[bloc, "gdp_per_capita"]) ** eps
        composite = eff_ci * afford
        c[bloc] = ac * (composite / ref_composite)
        d[bloc] = ad * (raw.loc[bloc, "ag_value_pct"] / ag_ref)
        p[bloc] = ap * (raw.loc[bloc, "trade_pct"] / 100.0)

    return GameParams(
        T=10, 
        theta=theta, 
        lam=lambda_map, # Now passes the dict {bloc: lambda}
        b=ab,
        gamma=gamma, 
        kappa=kappa, 
        eta=eta,
        weights=weights, 
        costs=c, 
        damages=d,
        pressure=p, 
        discount=dict(discount),
    )

# ──────────────────────────────────────────────────────────────────────────────
# SWEEP DEFINITIONS  (15 channels, mirrors Cell 5)
# ──────────────────────────────────────────────────────────────────────────────

def make_sweep_defs(n_pts):
    """Return list of (key, x_label, x_values, modifier_fn, direction, x_baseline)."""
    defs = []
    
    # Use the SMM baseline value for all lambda baseline markers
    lam_base = SMM_BASELINE["lam"]
    lam_xs = np.linspace(0.5, 10.0, n_pts) # Wider range to capture transition to full rationality

    # ── Individual Rationality Sweeps (The λ Channels) ──
    # We sweep each player's lambda specifically while others stay at SMM_BASELINE
    for bloc in PLAYERS:
        defs.append((f"λ_{bloc}", f"λ_{bloc} rationality", lam_xs,
            lambda p, v, b=bloc: GameParams(**{**p.__dict__, 
                                               "lam": {**p.lam, b: v}}),
            "inc", lam_base))

    # ── Discount factors (δ) ──
    xs_delta = np.linspace(0.50, 0.95, n_pts)
    defs.append(("δ_EU", "δ_EU", np.linspace(0.75, 0.95, n_pts),
        lambda p, v: GameParams(**{**p.__dict__,
                                   "discount": {**p.discount, "EU": v}}),
        "inc", 0.85))
    defs.append(("δ_US", "δ_US", xs_delta,
        lambda p, v: GameParams(**{**p.__dict__,
                                   "discount": {**p.discount, "US": v}}),
        "inc", 0.75))
    defs.append(("δ_CN", "δ_CN", xs_delta,
        lambda p, v: GameParams(**{**p.__dict__,
                                   "discount": {**p.discount, "CN": v}}),
        "inc", 0.80))
    defs.append(("δ_RoW", "δ_RoW", np.linspace(0.50, 0.80, n_pts),
        lambda p, v: GameParams(**{**p.__dict__,
                                   "discount": {**p.discount, "RoW": v}}),
        "inc", 0.65))
    defs.append(("δ_US=δ_CN", "δ_US = δ_CN",
        np.linspace(0.50, 0.95, n_pts),
        lambda p, v: GameParams(**{**p.__dict__,
                                   "discount": {**p.discount, "US": v, "CN": v}}),
        "inc", 0.775))

    # ── Cost multipliers ──
    for bloc, rng in [("Global cost", (0.3, 1.3)),
                      ("CN cost",     (0.3, 1.5)),
                      ("US cost",     (0.3, 1.5)),
                      ("RoW cost",    (0.2, 1.2))]:
        xs = np.linspace(*rng, n_pts)
        if bloc == "Global cost":
            defs.append((bloc, "cost multiplier", xs,
                lambda p, v: GameParams(**{**p.__dict__,
                    "costs": {k: c * v for k, c in p.costs.items()}}),
                "dec", 1.0))
        else:
            b_ = bloc.split()[0]
            defs.append((bloc, f"{b_} cost mult", xs,
                lambda p, v, b=b_: GameParams(**{**p.__dict__,
                    "costs": {**p.costs, b: p.costs[b] * v}}),
                "dec", 1.0))

    # ── Pressure multipliers ──
    for bloc, rng in [("US pressure",     (0.2, 4.0)),
                      ("CN pressure",     (0.2, 4.0)),
                      ("Global pressure", (0.2, 4.0))]:
        xs = np.linspace(*rng, n_pts)
        if bloc == "Global pressure":
            defs.append((bloc, "global pres mult", xs,
                lambda p, v: GameParams(**{**p.__dict__,
                    "pressure": {k: pv * v for k, pv in p.pressure.items()}}),
                "inc", 1.0))
        else:
            b_ = bloc.split()[0]
            defs.append((bloc, f"{b_} pres mult", xs,
                lambda p, v, b=b_: GameParams(**{**p.__dict__,
                    "pressure": {**p.pressure, b: p.pressure[b] * v}}),
                "inc", 1.0))

    # ── Global structural parameters ──
    defs.append(("γ (learning)",       "γ",    np.linspace(0.05, 0.70, n_pts),
        lambda p, v: GameParams(**{**p.__dict__, "gamma": v}), "inc", GAMMA))
    defs.append(("θ (threshold)",      "θ",    np.linspace(0.50, 0.95, n_pts),
        lambda p, v: GameParams(**{**p.__dict__, "theta": v}), "dec", THETA))
    defs.append(("κ (damage speed)",   "κ",    np.linspace(0.01, 0.20, n_pts),
        lambda p, v: GameParams(**{**p.__dict__, "kappa": v}), "inc", KAPPA))
    defs.append(("η (sigmoid steep)",  "η",    np.linspace(5,   25,   n_pts),
        lambda p, v: GameParams(**{**p.__dict__, "eta": v}), "inc", ETA))

    return defs

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def run_mc(params, n_mc, seed=42):
    V, sig, _, _ = solve_model(params)
    W_paths, adopt_time = monte_carlo(V, sig, params, n_runs=n_mc, seed=seed)
    success = float(np.mean(W_paths[:, -1] >= params.theta))
    mean_W  = float(np.mean(W_paths[:, -1]))
    # Mean period of first global coordination success (first t where W >= theta), per run
    crossed = W_paths >= params.theta                          # (n_mc, T+1) bool
    coord_t = np.where(crossed.any(axis=1),
                       np.argmax(crossed, axis=1).astype(float),
                       np.nan)                                 # col index = period number
    mean_ct = float(np.nanmean(coord_t)) if (~np.isnan(coord_t)).any() else np.nan
    fm = {p: float(np.mean(adopt_time[:, i] == 1))
          for i, p in enumerate(PLAYERS)}
    return V, sig, W_paths, adopt_time, success, mean_W, mean_ct, fm


def crossing_value(xs, ys, target, direction="inc"):
    """
    Crossing of success_rate with target.

    direction="inc"  : parameter helps coordination → scan low→high,
                       return first x where ys >= target (minimum needed).
    direction="dec"  : parameter hurts coordination → scan high→low,
                       return last x where ys >= target (maximum tolerable).
    """
    if direction == "inc":
        for x, y in zip(xs, ys):
            if y >= target:
                return round(float(x), 4)
    else:  # "dec": scan right-to-left
        for x, y in zip(reversed(xs), reversed(ys)):
            if y >= target:
                return round(float(x), 4)
    return None


def print_banner(title):
    w = 65
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


# ──────────────────────────────────────────────────────────────────────────────
# BASELINE RUN
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline(params, n_mc, n_sweep_pts, sweep_mc):
    print_banner("SMM BASELINE — Marginal Parameter Sweeps")

    # ── Main MC ──────────────────────────────────────────────────
    t0 = time.time()
    V, sig, W_paths, adopt_time, success, mean_W, mean_ct, fm = run_mc(params, n_mc)
    elapsed = time.time() - t0

    print(f"\n  MC results ({n_mc} runs, {elapsed:.1f}s):")
    print(f"    Coordination success rate : {success:.3f}")
    print(f"    Mean final W              : {mean_W:.4f}  (θ={params.theta})")
    print(f"    Mean coord timing         : {mean_ct:.2f} periods" if not np.isnan(mean_ct)
          else f"    Mean coord timing         : n/a (no successes)")
    print(f"    First-mover prob (t=1)    :", {p: f"{v:.3f}" for p, v in fm.items()})

    # ── Per-player adoption probabilities over time ──────────────
    print(f"\n  Equilibrium adoption probs (sigma[t][(0,0,0,0)]):")
    G0 = (0, 0, 0, 0)
    header = f"    {'t':>4}  " + "  ".join(f"{p:>7}" for p in PLAYERS)
    print(header)
    for t in range(1, params.T + 1):
        row = f"    {t:>4}  " + "  ".join(f"{sig[t][G0][i]:>7.4f}"
                                            for i in range(len(PLAYERS)))
        print(row)

    # ── Summary CSV ──────────────────────────────────────────────
    summary_rows = []
    for t in range(1, params.T + 1):
        row = {"t": t}
        for i, p in enumerate(PLAYERS):
            row[f"sigma_{p}"] = sig[t][G0][i]
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary["success_rate"]    = success
    df_summary["mean_W"]          = mean_W
    df_summary["mean_coord_time"] = mean_ct
    for p, v in fm.items():
        df_summary[f"fm_{p}"] = v
    path = os.path.join(OUT_DIR, "baseline_summary.csv")
    df_summary.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    # ── Marginal parameter sweeps ─────────────────────────────────
    print(f"\n  Running {n_sweep_pts}-point sweeps ({sweep_mc} MC each)...")
    sweep_defs    = make_sweep_defs(n_sweep_pts)
    sweep_results = []
    all_sweep_rows = []

    for channel, x_label, xs, modifier, direction, x_baseline in sweep_defs:
        rates = []
        for x in xs:
            p_mod = modifier(params, x)
            try:
                _, _, _, _, succ_m, _, _, _ = run_mc(p_mod, sweep_mc, seed=0)
            except Exception:
                succ_m = np.nan
            rates.append(succ_m)
            pct = (float(x) - x_baseline) / x_baseline * 100
            all_sweep_rows.append({
                "channel": channel, "x_label": x_label,
                "x": float(x), "pct_change": round(pct, 2),
                "success_rate": succ_m,
            })
        rates = np.array(rates)
        xs_pct = (xs - x_baseline) / x_baseline * 100
        c90_abs = crossing_value(xs, rates, 0.90, direction)
        c95_abs = crossing_value(xs, rates, 0.95, direction)
        c90_pct = round((c90_abs - x_baseline) / x_baseline * 100, 2) if c90_abs is not None else None
        c95_pct = round((c95_abs - x_baseline) / x_baseline * 100, 2) if c95_abs is not None else None
        sweep_results.append({
            "channel":    channel,
            "x_label":    x_label,
            "xs":         xs,
            "xs_pct":     xs_pct,
            "x_baseline": x_baseline,
            "rates":      rates,
            "cross_90":   c90_pct,
            "cross_95":   c95_pct,
        })
        print(f"    {channel:<25}  90%-cross={str(c90_pct):<8}%  95%-cross={str(c95_pct)}%")

    df_sweeps = pd.DataFrame(all_sweep_rows)
    path = os.path.join(OUT_DIR, "baseline_sweeps.csv")
    df_sweeps.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # ── Sweep figure ─────────────────────────────────────────────
    _plot_sweeps(sweep_results, params)

    return params, V, sig, W_paths, adopt_time, success, mean_W, mean_ct, fm, sweep_results


def _plot_sweeps(sweep_results, params):
    n  = len(sweep_results)
    nc = 5
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr))
    axes = np.array(axes).flatten()

    for ax, res in zip(axes, sweep_results):
        xs_pct, rates = res["xs_pct"], res["rates"]
        ax.plot(xs_pct, rates, lw=2, color="#2A9D8F")
        ax.axhline(0.5, ls="--", color="gray", lw=0.8)
        ax.axhline(params.theta, ls=":", color="#E63946", lw=0.8,
                   label=f"θ={params.theta}")
        ax.axvline(0, ls="-", color="gray", lw=0.6, alpha=0.4)   # baseline marker
        if res["cross_90"] is not None:
            ax.axvline(res["cross_90"], ls="--", color="#E63946", lw=0.8, alpha=0.6)
        ax.set_title(res["channel"], fontsize=9)
        ax.set_xlabel("% change from baseline", fontsize=7)
        ax.set_ylabel("P(coord)", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("SMM Baseline — Marginal Parameter Sweeps",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "sweeps_baseline.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL SENSITIVITY ANALYSIS  (mirrors Cell 9)
# ──────────────────────────────────────────────────────────────────────────────

GSA_RANGES = {
    # Discount factors (Patience)
    "delta_US":       (0.65, 0.88),
    "delta_EU":       (0.75, 0.90),
    "delta_CN":       (0.68, 0.88),
    "delta_RoW":      (0.55, 0.78),
    
    # Per-bloc cost multipliers (Economic Capacity)
    "cost_US":        (0.4,  1.6),
    "cost_EU":        (0.4,  1.6),
    "cost_CN":        (0.4,  1.6),
    "cost_RoW":       (0.4,  1.6),
    
    # Per-bloc pressure multipliers (Political Salience)
    "pressure_US":    (0.25, 3.0),
    "pressure_EU":    (0.25, 3.0),
    "pressure_CN":    (0.25, 3.0),
    "pressure_RoW":   (0.25, 3.0),
    
    # Individual Rationality (The "Noise vs Strategy" Sweep)
    "lam_US":         (0.1,  5.0),
    "lam_EU":         (0.1,  5.0),
    "lam_CN":         (0.1,  5.0),
    "lam_RoW":        (0.1,  5.0),
    
    # Global Structural parameters
    "theta":          (0.60, 0.92),
    "gamma":          (0.10, 0.45),
    "eta":            (8.0,  22.0),
    "kappa":          (0.02,  0.20),
}

GSA_OUTCOMES = ["success_rate", "mean_W", "mean_coord_time",
                "fm_US", "fm_EU", "fm_CN", "fm_RoW"]


def run_gsa(params, raw, weights, n_samples, gsa_mc, seed=42):
    print_banner(f"GSA — SMM Baseline  ({n_samples} samples, {gsa_mc} MC each)")
    rng = np.random.default_rng(seed)

    rows = []
    for s_idx in range(n_samples):
        if (s_idx + 1) % 100 == 0:
            print(f"    Sample {s_idx+1}/{n_samples}...")

        draw = {k: float(rng.uniform(lo, hi))
                for k, (lo, hi) in GSA_RANGES.items()}

        disc = {
            "US":  draw["delta_US"],
            "EU":  draw["delta_EU"],
            "CN":  draw["delta_CN"],
            "RoW": draw["delta_RoW"],
        }
        
        # New: Map individual rationality draws for each player
        lam_map_gsa = {
            "US":  draw["lam_US"],
            "EU":  draw["lam_EU"],
            "CN":  draw["lam_CN"],
            "RoW": draw["lam_RoW"],
        }
        
        costs_gsa    = {b: params.costs[b]    * draw[f"cost_{b}"]     for b in PLAYERS}
        pressure_gsa = {b: params.pressure[b] * draw[f"pressure_{b}"] for b in PLAYERS}
        
        # Build GameParams with the new heterogeneous maps
        p_gsa = GameParams(**{
            **params.__dict__,
            "discount": disc,
            "costs":    costs_gsa,
            "pressure": pressure_gsa,
            "lam":      lam_map_gsa, # Pass the dictionary
            "theta":    draw["theta"],
            "gamma":    draw["gamma"],
            "eta":      draw["eta"],
            "kappa":    draw["kappa"],
        })
        
        try:
            _, _, _, _, succ, mW, mct, fm = run_mc(p_gsa, gsa_mc, seed=int(s_idx))
        except Exception:
            succ, mW, mct, fm = np.nan, np.nan, np.nan, {p: np.nan for p in PLAYERS}

        row = {**draw, "success_rate": succ, "mean_W": mW, "mean_coord_time": mct}
        for p in PLAYERS:
            row[f"fm_{p}"] = fm[p]
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "gsa_baseline_samples.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # ── Spearman correlations ─────────────────────────────────────
    params_cols  = list(GSA_RANGES.keys())
    outcome_cols = GSA_OUTCOMES
    corr_rows    = []
    for outcome in outcome_cols:
        valid = df[[*params_cols, outcome]].dropna()
        for param in params_cols:
            rho, pval = spearmanr(valid[param], valid[outcome])
            corr_rows.append({
                "parameter": param, "outcome": outcome,
                "rho": round(rho, 4), "pvalue": round(pval, 4),
            })

    df_corr = pd.DataFrame(corr_rows)
    path = os.path.join(OUT_DIR, "gsa_baseline_correlations.csv")
    df_corr.to_csv(path, index=False)

    # ── Print correlation tables ──────────────────────────────────
    print(f"\n  Spearman ρ — Success Rate:")
    sr_corr = (df_corr[df_corr["outcome"] == "success_rate"]
               .set_index("parameter")[["rho","pvalue"]]
               .sort_values("rho", ascending=False))
    print(sr_corr.to_string())

    print(f"\n  Spearman ρ — Coordination Timing (negative = earlier coord):")
    ct_corr = (df_corr[df_corr["outcome"] == "mean_coord_time"]
               .set_index("parameter")[["rho","pvalue"]]
               .sort_values("rho"))
    print(ct_corr.to_string())

    print(f"\n  Spearman ρ — CN First-Mover Prob:")
    cn_corr = (df_corr[df_corr["outcome"] == "fm_CN"]
               .set_index("parameter")[["rho","pvalue"]]
               .sort_values("rho", ascending=False))
    print(cn_corr.to_string())

    return df, df_corr




# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use fewer MC draws (quick test run)")
    args = parser.parse_args()

    N_MC_BASELINE = 1000  if args.fast else 1500
    N_SWEEP_PTS   = 8    if args.fast else 12
    SWEEP_MC      = 100  if args.fast else 300
    N_GSA_SAMPLES = 100  if args.fast else 1000
    GSA_MC        = 50   if args.fast else 200

    if args.fast:
        print("  [--fast mode: reduced MC draws for quick testing]")

    # ── Load data ─────────────────────────────────────────────────
    bloc_data = build_bloc_data(data_dir=os.path.join(ROOT_DIR, "data"))
    raw = bloc_data.set_index("bloc")

    weights = {}
    for bloc in PLAYERS:
        weights[bloc] = (0.5 * raw.loc[bloc, "emission_share"]
                       + 0.5 * raw.loc[bloc, "gdp_share"])
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}
    print(f"  Weights: { {k: round(v,4) for k,v in weights.items()} }")

    # ── Build SMM baseline params ─────────────────────────────────
    b = SMM_BASELINE
    params = build_params(raw, weights,
                          ac=b["ac"], ad=b["ad"], ap=b["ap"], ab=b["ab"],
                          lam=b["lam"])

    # ── Sweeps ────────────────────────────────────────────────────
    res = run_baseline(params,
                       n_mc=N_MC_BASELINE,
                       n_sweep_pts=N_SWEEP_PTS,
                       sweep_mc=SWEEP_MC)
    params, V, sig, W_paths, adopt_time, success, mean_W, mean_ct, fm, sweeps = res

    # ── Sweep crossing table ──────────────────────────────────────
    print_banner("SWEEP CROSSING VALUES (% change from SMM baseline)")
    print(f"\n  {'Channel':<28} {'90%-cross (%)':>14} {'95%-cross (%)':>14}")
    print("  " + "-" * 58)
    crossing_rows = []
    for r in sweeps:
        c90 = f"{r['cross_90']:+.2f}%" if r["cross_90"] is not None else "absent"
        c95 = f"{r['cross_95']:+.2f}%" if r["cross_95"] is not None else "absent"
        print(f"  {r['channel']:<28} {str(c90):>14} {str(c95):>14}")
        crossing_rows.append({"channel": r["channel"],
                               "cross_90_pct": r["cross_90"],
                               "cross_95_pct": r["cross_95"]})
    pd.DataFrame(crossing_rows).to_csv(
        os.path.join(OUT_DIR, "sweep_crossings.csv"), index=False)

    # ── GSA ──────────────────────────────────────────────────────
    run_gsa(params, raw, weights, n_samples=N_GSA_SAMPLES, gsa_mc=GSA_MC)

    # ── Final summary ─────────────────────────────────────────────
    print_banner("DONE")
    print(f"\n  All outputs in: {os.path.abspath(OUT_DIR)}/")
    print(f"  Figures in:     {os.path.abspath(FIG_DIR)}/")
    print()
    print("  Files written:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".csv"):
            print(f"    {f}")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"    figures/{f}")


if __name__ == "__main__":
    main()
