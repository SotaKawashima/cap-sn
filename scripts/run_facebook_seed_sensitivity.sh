#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUST_BIN="${RUST_BIN:-target/release/v2}"
NETWORK_CONF="${NETWORK_CONF:-v2/test_2/network/network-facebook.toml}"
AGENT_CONF="${AGENT_CONF:-v2/test_2/agent/agent-type6.toml}"
OUTPUT_BASE="${OUTPUT_BASE:-experiments/2026-06-10_facebook_seed_sensitivity/strategy_runs}"
ITERATION_COUNT="${ITERATION_COUNT:-100}"
SEEDS_TEXT="${SEEDS:-0 1 2 3 4}"

if [ ! -x "$RUST_BIN" ]; then
  echo "Error: $RUST_BIN が見つからないか、実行できません。先に cargo build --release を実行してください。"
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
RUN_ID="${RUN_ID:-seed_sensitivity_${TIMESTAMP}}"
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_ID"
RUNTIME_DIR="$OUTPUT_ROOT/runtime"
mkdir -p "$OUTPUT_ROOT/logs" "$RUNTIME_DIR"
MASTER_LOG="$OUTPUT_ROOT/logs/run_facebook_seed_sensitivity_${TIMESTAMP}.log"

echo "=== Start facebook seed sensitivity batch ===" | tee -a "$MASTER_LOG"
echo "Run id         : $RUN_ID" | tee -a "$MASTER_LOG"
echo "Output root    : $OUTPUT_ROOT" | tee -a "$MASTER_LOG"
echo "Seeds          : ${SEED_LIST[*]}" | tee -a "$MASTER_LOG"
echo "Iteration count: $ITERATION_COUNT" | tee -a "$MASTER_LOG"
echo "Network        : $NETWORK_CONF" | tee -a "$MASTER_LOG"
echo "Agent          : $AGENT_CONF" | tee -a "$MASTER_LOG"
echo "Time           : $(date)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

for SEED in "${SEED_LIST[@]}"; do
  RUNTIME_CONF="$RUNTIME_DIR/runtime_seed${SEED}.toml"
  {
    echo "seed_state = $SEED"
    echo "iteration_count = $ITERATION_COUNT"
  } > "$RUNTIME_CONF"

  for STRATEGY_ENTRY in "${STRATEGIES[@]}"; do
    STRATEGY_NAME="${STRATEGY_ENTRY%%:*}"
    STRATEGY_CONF="${STRATEGY_ENTRY#*:}"

    IDENTIFIER="facebook_seed${SEED}_${STRATEGY_NAME}"
    RESULT_DIR="$OUTPUT_ROOT/seed_${SEED}/facebook/$STRATEGY_NAME/result"
    RUN_LOG="$OUTPUT_ROOT/seed_${SEED}/facebook/$STRATEGY_NAME/run_${TIMESTAMP}.log"

    mkdir -p "$RESULT_DIR" "$(dirname "$RUN_LOG")"

    echo "========================================" | tee -a "$MASTER_LOG"
    echo "Start: seed=$SEED, network=facebook, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
    echo "Time : $(date)" | tee -a "$MASTER_LOG"
    echo "Runtime   : $RUNTIME_CONF" | tee -a "$MASTER_LOG"
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

    echo "Finished: seed=$SEED, network=facebook, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
    echo "Time    : $(date)" | tee -a "$MASTER_LOG"
    echo | tee -a "$MASTER_LOG"
  done
done

echo "=== All facebook seed sensitivity runs finished ===" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
