#!/usr/bin/env bash
# run_all.sh — Full clean rebuild of all Climate Game outputs
set -e
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
LOG="results/run_all.log"

mkdir -p results/figures results/dissertation

# ── Wipe all regenerated outputs ──────────────────────────────────
echo "Clearing old outputs..."
rm -f results/*.csv
rm -f results/figures/*.png
rm -f results/dissertation/*.tex \
      results/dissertation/*.png \
      results/dissertation/*.pdf \
      results/dissertation/*.log \
      results/dissertation/*.zip

# ── Run pipeline ──────────────────────────────────────────────────
run() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════"
    $PYTHON "$1" 2>&1 | tee -a "$LOG"
}

> "$LOG"  # reset log

run core/run_analysis.py
run core/game_diagram.py
run core/dissertation_outputs.py
run robustness/equilibrium_uniqueness.py
run robustness/counterfactual_rationality.py
run robustness/spec_robustness.py

echo ""
echo "══════════════════════════════════════════"
echo "  All done. Outputs in results/"
echo "  Full log: $LOG"
echo "══════════════════════════════════════════"
