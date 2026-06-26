#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 1 ]; then
  echo "使い方: ./scripts/run_optimize.sh <network> [trials] [score_metric] [run_label] [optuna_seed_count] [optuna_seed_start]"
  echo "例: ./scripts/run_optimize.sh ba_1000 100 auc"
  echo "run別ディレクトリに出す場合: ./scripts/run_optimize.sh ba_1000 100 auc 20260617"
  echo "1 seedだけ試す場合: ./scripts/run_optimize.sh ba_1000 10 auc test 1"
  echo "optseed4-6を実行する場合: KEEP_TRIAL_RAW=all ./scripts/run_optimize.sh ba_1000 100 auc raw_20260626 3 4"
  echo "旧指標で実行する場合: ./scripts/run_optimize.sh ba_1000 100 final"
  exit 1
fi

NETWORK="$1"
TRIALS="${2:-100}"
SCORE_METRIC="${3:-auc}"
RUN_LABEL="${4:-}"
OPTUNA_SEED_COUNT="${5:-3}"
OPTUNA_SEED_START="${6:-1}"
KEEP_TRIAL_RAW="${KEEP_TRIAL_RAW:-none}"

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
  echo "Error: optuna_seed_count は 1 以上の整数を指定してください"
  exit 1
fi

if ! [[ "$OPTUNA_SEED_START" =~ ^[0-9]+$ ]]; then
  echo "Error: optuna_seed_start は 1 以上の整数を指定してください"
  exit 1
fi

if [ "$OPTUNA_SEED_COUNT" -lt 1 ]; then
  echo "Error: optuna_seed_count は 1 以上を指定してください"
  exit 1
fi

if [ "$OPTUNA_SEED_START" -lt 1 ]; then
  echo "Error: optuna_seed_start は 1 以上を指定してください"
  exit 1
fi

case "$KEEP_TRIAL_RAW" in
  none|info-pop|all)
    ;;
  *)
    echo "Error: KEEP_TRIAL_RAW は none / info-pop / all のいずれかを指定してください"
    exit 1
    ;;
esac

METHODS=("GPR" "CMAES" "RANDOM" "GA")

if [ -n "$RUN_LABEL" ] && [ "$KEEP_TRIAL_RAW" != "none" ] && [[ "$RUN_LABEL" != *raw* ]]; then
  echo "Warning: KEEP_TRIAL_RAW=$KEEP_TRIAL_RAW ですが run_label に raw が含まれていません。"
  echo "         既存実験と混ざらない run_label を推奨します。"
fi

if [ -z "$RUN_LABEL" ] && [ "$KEEP_TRIAL_RAW" != "none" ]; then
  echo "Error: KEEP_TRIAL_RAW=$KEEP_TRIAL_RAW のときは既存結果との混在を避けるため run_label を指定してください"
  exit 1
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: python / python3 が見つかりません"
    exit 1
  fi
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
echo "SeedStart: $OPTUNA_SEED_START" | tee -a "$MASTER_LOG"
echo "KeepRaw : $KEEP_TRIAL_RAW" | tee -a "$MASTER_LOG"
echo "Methods : ${METHODS[*]}" | tee -a "$MASTER_LOG"
echo "Time    : $(date)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

for ((SEED_OFFSET=0; SEED_OFFSET<OPTUNA_SEED_COUNT; SEED_OFFSET++)); do
  REPLICATE_NUMBER=$((OPTUNA_SEED_START + SEED_OFFSET))
  REPLICATE_LABEL="optseed${REPLICATE_NUMBER}"
  SEED_BASE=$(((REPLICATE_NUMBER + 3) * 1000 + 200))

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
      "$PYTHON_BIN" optimize_test.py
      --method "$METHOD"
      --network "$NETWORK"
      --trials "$TRIALS"
      --score-metric "$SCORE_METRIC"
      --sampler-seed "$SAMPLER_SEED"
      --replicate-label "$REPLICATE_LABEL"
      --keep-trial-raw "$KEEP_TRIAL_RAW"
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
