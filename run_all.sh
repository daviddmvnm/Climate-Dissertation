#!/usr/bin/env bash
# run_all.sh — Full clean rebuild of all Climate Game outputs
set -e
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
LOG="results/run_all.log"

mkdir -p results/figures

# ── Wipe all regenerated outputs ──────────────────────────────────
echo "Clearing old outputs..."
rm -f results/*.csv results/*.npy
rm -f results/figures/*.png

# ── Run pipeline ──────────────────────────────────────────────────
run() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════"
    $PYTHON "$1" 2>&1 | tee -a "$LOG"
}

> "$LOG"  # reset log

# Calibrate first: produces results/smm_best_theta.npy used by the rest.
run calibration/smm_calibration.py
run calibration/smm_verification.py

# Baseline simulation, marginal sweeps, global sensitivity analysis.
run core/run_analysis.py

# Robustness checks.
run robustness/equilibrium_uniqueness.py
run robustness/phi_sweep_calibration.py
run robustness/spec_robustness.py
run robustness/discount_ordering_robustness.py
run robustness/cascade_robustness.py
run robustness/row_payoff_robustness.py

echo ""
echo "══════════════════════════════════════════"
echo "  All done. Outputs in results/"
echo "  Full log: $LOG"
echo "══════════════════════════════════════════"
