#!/usr/bin/env bash
# ==============================================================================
# rpi_quickstart.sh - Raspberry Pi 4 / 5 One-Click Diagnosis & Stress Test
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo " [Raspberry Pi 4 / 5] Gearbox Hybrid Multimodal Diagnosis & Thermal Suite"
echo "=============================================================================="
echo " Select operation:"
echo " 1) Run Deployed Models Benchmark (Pure Inference, ~5 sec)"
echo " 2) Run 1-Minute Thermal Stress Dry-Run (60 sec)"
echo " 3) Run 2-Hour Full Thermal Endurance Test (7,200 sec = 9,796 inferences)"
echo " 4) Launch Real-Time Thermal Heatmap Dashboard UI"
echo " 5) Exit"
echo "=============================================================================="

read -p "Enter choice [1-5]: " CHOICE

case "$CHOICE" in
  1)
    echo "[INFO] Running Deployed Models Benchmark..."
    python evaluate_deployed_models.py
    ;;
  2)
    echo "[INFO] Starting 1-Minute Thermal Stress Test..."
    python run_2hours_thermal_stress_test.py --duration 60 --interval 1.0
    ;;
  3)
    echo "[INFO] Starting 2-Hour Full Endurance Stress Test (7,200s)..."
    python run_2hours_thermal_stress_test.py --duration 7200 --interval 1.0
    ;;
  4)
    echo "[INFO] Launching Live Thermal Dashboard UI..."
    python thermal_live_dashboard.py --refresh 1000
    ;;
  5)
    echo "Exiting."
    exit 0
    ;;
  *)
    echo "[ERROR] Invalid choice: $CHOICE"
    exit 1
    ;;
esac
