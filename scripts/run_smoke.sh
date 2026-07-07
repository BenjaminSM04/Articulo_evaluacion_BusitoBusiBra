#!/usr/bin/env bash
# Validación end-to-end del pipeline en CPU con datos sintéticos (Linux/Mac).
# Ejecuta desde la raíz del proyecto:  bash scripts/run_smoke.sh
set -euo pipefail
CFG=config/config_smoke.yaml

python scripts/00_make_synthetic_data.py --config "$CFG"
python scripts/02_preprocess.py          --config "$CFG"
python scripts/03_train_baseline.py      --config "$CFG" --dataset busi
python scripts/03_train_baseline.py      --config "$CFG" --dataset bus_bra
python scripts/04_cross_domain_matrix.py --config "$CFG"
python scripts/06_generate_report.py     --config "$CFG"

echo
echo "===== SMOKE OK: revisa results/reports/SUMMARY.md y results/figures/ ====="
