#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUST_BIN="${RUST_BIN:-target/release/v2}"
NETWORK_CONFIG_DIR="${NETWORK_CONFIG_DIR:-v2/test_2/network}"
AGENT_CONF="${AGENT_CONF:-v2/test_2/agent/agent-type6.toml}"
OUTPUT_BASE="${OUTPUT_BASE:-experiments/2026-06-16_facebook_degree_rewire/strategy_runs}"
ITERATION_COUNT="${ITERATION_COUNT:-100}"
SEEDS_TEXT="${SEEDS:-0 1 2 3 4}"

if [ ! -x "$RUST_BIN" ]; then
  echo "Error: $RUST_BIN が見つからないか、実行できません。先に cargo build --release を実行してください。"
  exit 1
fi

shopt -s nullglob
NETWORK_CONFIGS=("$NETWORK_CONFIG_DIR"/network-fbdeg_*.toml)
shopt -u nullglob

if [ "${#NETWORK_CONFIGS[@]}" -eq 0 ]; then
  echo "Error: network-fbdeg_*.toml が見つかりません。先に以下を実行してください。"
  echo "  .venv/bin/python scripts/prepare_facebook_degree_rewire_experiment.py"
  exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS_TEXT"
if [ "${#SEED_LIST[@]}" -eq 0 ]; then
  echo "Error: SEEDS が空です。例: SEEDS=\"0 1 2 3 4\""
  exit 1
fi

STRATEGIES=(
  "balance:v2/test_2/strategy/strategy-balance.toml"
  "effective_high:v2/test_2/strategy/strategy-effective-high.toml"
  "certainty_high:v2/test_2/strategy/strategy-certainty-high.toml"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="${RUN_ID:-degree_rewire_${TIMESTAMP}}"
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_ID"
RUNTIME_DIR="$OUTPUT_ROOT/runtime"
mkdir -p "$OUTPUT_ROOT/logs" "$RUNTIME_DIR"
MASTER_LOG="$OUTPUT_ROOT/logs/run_facebook_degree_rewire_${TIMESTAMP}.log"

echo "=== Start facebook degree-rewire strategy batch ===" | tee -a "$MASTER_LOG"
echo "Run id         : $RUN_ID" | tee -a "$MASTER_LOG"
echo "Output root    : $OUTPUT_ROOT" | tee -a "$MASTER_LOG"
echo "Seeds          : ${SEED_LIST[*]}" | tee -a "$MASTER_LOG"
echo "Iteration count: $ITERATION_COUNT" | tee -a "$MASTER_LOG"
echo "Agent          : $AGENT_CONF" | tee -a "$MASTER_LOG"
echo "Networks       : ${#NETWORK_CONFIGS[@]}" | tee -a "$MASTER_LOG"
echo "Time           : $(date)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

for SEED in "${SEED_LIST[@]}"; do
  RUNTIME_CONF="$RUNTIME_DIR/runtime_seed${SEED}.toml"
  {
    echo "seed_state = $SEED"
    echo "iteration_count = $ITERATION_COUNT"
  } > "$RUNTIME_CONF"

  for NETWORK_CONF in "${NETWORK_CONFIGS[@]}"; do
    NETWORK_BASENAME="$(basename "$NETWORK_CONF" .toml)"
    NETWORK_NAME="${NETWORK_BASENAME#network-}"

    for STRATEGY_ENTRY in "${STRATEGIES[@]}"; do
      STRATEGY_NAME="${STRATEGY_ENTRY%%:*}"
      STRATEGY_CONF="${STRATEGY_ENTRY#*:}"

      IDENTIFIER="${NETWORK_NAME}_seed${SEED}_${STRATEGY_NAME}"
      RESULT_DIR="$OUTPUT_ROOT/seed_${SEED}/$NETWORK_NAME/$STRATEGY_NAME/result"
      RUN_LOG="$OUTPUT_ROOT/seed_${SEED}/$NETWORK_NAME/$STRATEGY_NAME/run_${TIMESTAMP}.log"

      mkdir -p "$RESULT_DIR" "$(dirname "$RUN_LOG")"

      echo "========================================" | tee -a "$MASTER_LOG"
      echo "Start: seed=$SEED, network=$NETWORK_NAME, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
      echo "Time : $(date)" | tee -a "$MASTER_LOG"
      echo "Runtime   : $RUNTIME_CONF" | tee -a "$MASTER_LOG"
      echo "Config    : $NETWORK_CONF" | tee -a "$MASTER_LOG"
      echo "Result dir: $RESULT_DIR" | tee -a "$MASTER_LOG"
      echo "Run log   : $RUN_LOG" | tee -a "$MASTER_LOG"
      echo "========================================" | tee -a "$MASTER_LOG"

      "$RUST_BIN" "$IDENTIFIER" "$RESULT_DIR" \
        --runtime "$RUNTIME_CONF" \
        --network "$NETWORK_CONF" \
        --agent "$AGENT_CONF" \
        --strategy "$STRATEGY_CONF" \
        -e -o -d 0 \
        2>&1 | tee "$RUN_LOG"

      echo "Finished: seed=$SEED, network=$NETWORK_NAME, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
      echo "Time    : $(date)" | tee -a "$MASTER_LOG"
      echo | tee -a "$MASTER_LOG"
    done
  done
done

echo "=== All facebook degree-rewire runs finished ===" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
