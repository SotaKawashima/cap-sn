#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "使い方: ./scripts/run_all_optimize.sh [trials] [score_metric] [run_label] [optuna_seed_count]"
  echo "例: ./scripts/run_all_optimize.sh 100 auc 20260617 3"
  echo "1 seedだけ試す場合: ./scripts/run_all_optimize.sh 10 auc test 1"
  echo "旧指標で実行する場合: ./scripts/run_all_optimize.sh 100 final 20260617 3"
  exit 0
fi

TRIALS="${1:-100}"
SCORE_METRIC="${2:-auc}"
RUN_LABEL="${3:-}"
OPTUNA_SEED_COUNT="${4:-3}"

NETWORKS=("ba_1000" "facebook" "wiki-vote")

echo "=== Start all-network optimization batch ==="
echo "Networks : ${NETWORKS[*]}"
echo "Trials   : $TRIALS"
echo "Score    : $SCORE_METRIC"
echo "RunLabel : ${RUN_LABEL:-none}"
echo "OptSeeds : $OPTUNA_SEED_COUNT"
echo "Time     : $(date)"
echo

for NETWORK in "${NETWORKS[@]}"; do
  echo "########################################"
  echo "Start network: $NETWORK"
  echo "########################################"

  ./scripts/run_optimize.sh "$NETWORK" "$TRIALS" "$SCORE_METRIC" "$RUN_LABEL" "$OPTUNA_SEED_COUNT"

  echo
  echo "Finished network: $NETWORK"
  echo "Time            : $(date)"
  echo
done

echo "=== All-network optimization batch finished ==="
