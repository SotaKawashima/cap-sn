#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 1 ]; then
  echo "使い方: ./scripts/run_optimize.sh <network> [trials] [score_metric] [run_label] [optuna_seed_count]"
  echo "例: ./scripts/run_optimize.sh ba_1000 100 auc"
  echo "run別ディレクトリに出す場合: ./scripts/run_optimize.sh ba_1000 100 auc 20260617"
  echo "1 seedだけ試す場合: ./scripts/run_optimize.sh ba_1000 10 auc test 1"
  echo "旧指標で実行する場合: ./scripts/run_optimize.sh ba_1000 100 final"
  exit 1
fi

NETWORK="$1"
TRIALS="${2:-100}"
SCORE_METRIC="${3:-auc}"
RUN_LABEL="${4:-}"
OPTUNA_SEED_COUNT="${5:-3}"

case "$NETWORK" in
  ba_1000|facebook|wiki-vote)
    ;;
  *)
    echo "Error: network は ba_1000 / facebook / wiki-vote のいずれかを指定してください"
    exit 1
    ;;
esac

case "$SCORE_METRIC" in
  auc|final|peak|final-window)
    ;;
  *)
    echo "Error: score_metric は auc / final / peak / final-window のいずれかを指定してください"
    exit 1
    ;;
esac

if ! [[ "$OPTUNA_SEED_COUNT" =~ ^[0-9]+$ ]]; then
  echo "Error: optuna_seed_count は 1 以上 3 以下の整数を指定してください"
  exit 1
fi

METHODS=("GPR" "CMAES" "RANDOM" "GA")
OPTUNA_SEED_BASES=(4200 5200 6200)

if [ "$OPTUNA_SEED_COUNT" -lt 1 ] || [ "$OPTUNA_SEED_COUNT" -gt "${#OPTUNA_SEED_BASES[@]}" ]; then
  echo "Error: optuna_seed_count は 1 以上 ${#OPTUNA_SEED_BASES[@]} 以下を指定してください"
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN_DIR_NAME="optimize_runs"
if [ "$SCORE_METRIC" != "final" ]; then
  RUN_DIR_NAME="optimize_runs_${SCORE_METRIC//-/_}"
fi
if [ -n "$RUN_LABEL" ]; then
  RUN_DIR_NAME="${RUN_DIR_NAME}_${RUN_LABEL}"
fi

case "$NETWORK" in
  ba_1000)
    LOG_DIR="experiments/optimization_ba1000/${RUN_DIR_NAME}"
    ;;
  facebook)
    LOG_DIR="experiments/optimization_facebook/${RUN_DIR_NAME}"
    ;;
  wiki-vote)
    LOG_DIR="experiments/optimization_wiki_vote/${RUN_DIR_NAME}"
    ;;
esac
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="${LOG_DIR}/run_all_${TIMESTAMP}.log"

echo "=== Start batch experiment ===" | tee -a "$MASTER_LOG"
echo "Network : $NETWORK" | tee -a "$MASTER_LOG"
echo "Trials  : $TRIALS" | tee -a "$MASTER_LOG"
echo "Score   : $SCORE_METRIC" | tee -a "$MASTER_LOG"
echo "RunLabel: ${RUN_LABEL:-none}" | tee -a "$MASTER_LOG"
echo "OptSeeds: $OPTUNA_SEED_COUNT" | tee -a "$MASTER_LOG"
echo "Methods : ${METHODS[*]}" | tee -a "$MASTER_LOG"
echo "Time    : $(date)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

for ((SEED_INDEX=0; SEED_INDEX<OPTUNA_SEED_COUNT; SEED_INDEX++)); do
  REPLICATE_LABEL="optseed$((SEED_INDEX + 1))"
  SEED_BASE="${OPTUNA_SEED_BASES[$SEED_INDEX]}"

  for METHOD in "${METHODS[@]}"; do
    case "$METHOD" in
      GPR)
        METHOD_SEED_SUFFIX=1
        ;;
      CMAES)
        METHOD_SEED_SUFFIX=2
        ;;
      RANDOM)
        METHOD_SEED_SUFFIX=3
        ;;
      GA)
        METHOD_SEED_SUFFIX=4
        ;;
      *)
        echo "Error: unknown method $METHOD"
        exit 1
        ;;
    esac

    SAMPLER_SEED=$((SEED_BASE + METHOD_SEED_SUFFIX))

    echo "========================================" | tee -a "$MASTER_LOG"
    echo "Start: METHOD=$METHOD, NETWORK=$NETWORK, SCORE=$SCORE_METRIC, REPLICATE=$REPLICATE_LABEL, SAMPLER_SEED=$SAMPLER_SEED" | tee -a "$MASTER_LOG"
    echo "Time : $(date)" | tee -a "$MASTER_LOG"
    echo "========================================" | tee -a "$MASTER_LOG"

    CMD=(
      python optimize_test.py
      --method "$METHOD"
      --network "$NETWORK"
      --trials "$TRIALS"
      --score-metric "$SCORE_METRIC"
      --sampler-seed "$SAMPLER_SEED"
      --replicate-label "$REPLICATE_LABEL"
    )
    if [ -n "$RUN_LABEL" ]; then
      CMD+=(--run-label "$RUN_LABEL")
    fi

    "${CMD[@]}" 2>&1 | tee -a "$MASTER_LOG"

    echo | tee -a "$MASTER_LOG"
    echo "Finished: METHOD=$METHOD, NETWORK=$NETWORK, REPLICATE=$REPLICATE_LABEL, SAMPLER_SEED=$SAMPLER_SEED" | tee -a "$MASTER_LOG"
    echo "Time    : $(date)" | tee -a "$MASTER_LOG"
    echo | tee -a "$MASTER_LOG"
  done
done

echo "=== All experiments finished ===" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
