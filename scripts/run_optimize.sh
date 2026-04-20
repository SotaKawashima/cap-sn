#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 1 ]; then
  echo "使い方: ./scripts/run_optimize.sh <network> [trials]"
  echo "例: ./scripts/run_optimize.sh ba_1000 100"
  exit 1
fi

NETWORK="$1"
TRIALS="${2:-100}"

case "$NETWORK" in
  ba_1000|facebook|wiki-vote)
    ;;
  *)
    echo "Error: network は ba_1000 / facebook / wiki-vote のいずれかを指定してください"
    exit 1
    ;;
esac

METHODS=("GPR" "CMAES" "RANDOM" "GA")

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

LOG_DIR="optimize_test_${NETWORK}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="${LOG_DIR}/run_all_${TIMESTAMP}.log"

echo "=== Start batch experiment ===" | tee -a "$MASTER_LOG"
echo "Network : $NETWORK" | tee -a "$MASTER_LOG"
echo "Trials  : $TRIALS" | tee -a "$MASTER_LOG"
echo "Methods : ${METHODS[*]}" | tee -a "$MASTER_LOG"
echo "Time    : $(date)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

for METHOD in "${METHODS[@]}"; do
  echo "========================================" | tee -a "$MASTER_LOG"
  echo "Start: METHOD=$METHOD, NETWORK=$NETWORK" | tee -a "$MASTER_LOG"
  echo "Time : $(date)" | tee -a "$MASTER_LOG"
  echo "========================================" | tee -a "$MASTER_LOG"

  python optimize_test.py --method "$METHOD" --network "$NETWORK" --trials "$TRIALS" \
    2>&1 | tee -a "$MASTER_LOG"

  echo | tee -a "$MASTER_LOG"
  echo "Finished: METHOD=$METHOD, NETWORK=$NETWORK" | tee -a "$MASTER_LOG"
  echo "Time    : $(date)" | tee -a "$MASTER_LOG"
  echo | tee -a "$MASTER_LOG"
done

echo "=== All experiments finished ===" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"