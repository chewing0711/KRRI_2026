#!/usr/bin/env bash
# ==============================================================================
# rpi_quickstart.sh - Raspberry Pi 4 / 5 One-Click Diagnosis & Stress Test (w34 CAN)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo " [Raspberry Pi 4 / 5] Gearbox CAN-only w34 Online Diagnosis Suite"
echo "=============================================================================="
echo " Select operation:"
echo " 1) Run Deployed CAN-only Models Benchmark (evaluate_deployed_models.py)"
echo " 2) Run Online Simulation - ZOH Time-Series Merge (simulate_online_inference.py)"
echo " 3) Run Online Simulation - Direct NumPy Array (simulate_online_direct_array.py)"
echo " 4) Launch Live CAN Inference Node (live_can_inference_node.py)"
echo " 5) Exit"
echo "=============================================================================="

read -p "Enter choice [1-5]: " CHOICE

case "$CHOICE" in
  1)
    echo "[INFO] Running Deployed CAN-only Models Benchmark..."
    python evaluate_deployed_models.py
    ;;
  2)
    echo "[INFO] Starting Online Simulation (ZOH Time-Series Merge)..."
    python src/simulate_online_inference.py
    ;;
  3)
    echo "[INFO] Starting Online Simulation (Direct NumPy Array)..."
    python src/simulate_online_direct_array.py
    ;;
  4)
    echo "=============================================================================="
    echo " Enter Context Variables for Live Inference Node:"
    read -p " Enter CAN Interface (default: socketcan): " CAN_IF
    CAN_IF=${CAN_IF:-socketcan}
    read -p " Enter CAN Channel (default: can0): " CAN_CH
    CAN_CH=${CAN_CH:-can0}
    read -p " Enter Target Speed in kph (default: 20.0): " T_SPD
    T_SPD=${T_SPD:-20.0}
    read -p " Enter Target Grade in % (default: 0.0): " T_GRD
    T_GRD=${T_GRD:-0.0}
    echo "=============================================================================="
    echo "[INFO] Launching Live CAN Inference Node..."
    python live_can_inference_node.py --interface "$CAN_IF" --channel "$CAN_CH" --speed "$T_SPD" --grade "$T_GRD"
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
