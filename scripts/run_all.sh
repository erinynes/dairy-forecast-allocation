#!/usr/bin/env bash
# Runs the full pipeline in order. Execute from the repository root:  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

steps=(
  prep_data
  eda
  forecast_model
  forecast_significance
  robustness_check
  optimize_allocation
  optimize_stochastic
  sensitivity_check
  sensitivity_systematic
  district_case_study
  plot_results
)

mkdir -p results figures
for s in "${steps[@]}"; do
  echo "=== src/${s}.py ==="
  python "src/${s}.py"
done
echo "Done. Tables are in results/, figures in figures/."
