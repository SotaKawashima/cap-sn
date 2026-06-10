#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUST_BIN="target/release/v2"
RUNTIME_CONF="v2/test_2/runtime.toml"
NETWORK_CONF="v2/test_2/network/network-facebook.toml"
AGENT_CONF="v2/test_2/agent/agent-type6.toml"
OUTPUT_BASE="${OUTPUT_BASE:-experiments/2026-06-10_facebook_recheck/strategy_runs}"

if [ ! -x "$RUST_BIN" ]; then
  echo "Error: $RUST_BIN が見つからないか、実行できません。先に cargo build --release を実行してください。"
  exit 1
fi

STRATEGIES=(
  "balance:v2/test_2/strategy/strategy-balance.toml"
  "effective_high:v2/test_2/strategy/strategy-effective-high.toml"
  "certainty_high:v2/test_2/strategy/strategy-certainty-high.toml"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="${RUN_ID:-recheck_${TIMESTAMP}}"
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_ID"
mkdir -p "$OUTPUT_ROOT/logs"
MASTER_LOG="$OUTPUT_ROOT/logs/run_facebook_recheck_${TIMESTAMP}.log"

echo "=== Start facebook recheck strategy batch ===" | tee -a "$MASTER_LOG"
echo "Run id     : $RUN_ID" | tee -a "$MASTER_LOG"
echo "Output root: $OUTPUT_ROOT" | tee -a "$MASTER_LOG"
echo "Runtime    : $RUNTIME_CONF" | tee -a "$MASTER_LOG"
echo "Network    : $NETWORK_CONF" | tee -a "$MASTER_LOG"
echo "Agent      : $AGENT_CONF" | tee -a "$MASTER_LOG"
echo "Time       : $(date)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

for STRATEGY_ENTRY in "${STRATEGIES[@]}"; do
  STRATEGY_NAME="${STRATEGY_ENTRY%%:*}"
  STRATEGY_CONF="${STRATEGY_ENTRY#*:}"

  IDENTIFIER="facebook_${STRATEGY_NAME}"
  RESULT_DIR="$OUTPUT_ROOT/facebook/$STRATEGY_NAME/result"
  RUN_LOG="$OUTPUT_ROOT/facebook/$STRATEGY_NAME/run_${TIMESTAMP}.log"

  mkdir -p "$RESULT_DIR" "$(dirname "$RUN_LOG")"

  echo "========================================" | tee -a "$MASTER_LOG"
  echo "Start: network=facebook, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
  echo "Time : $(date)" | tee -a "$MASTER_LOG"
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

  echo "Finished: network=facebook, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
  echo "Time    : $(date)" | tee -a "$MASTER_LOG"
  echo | tee -a "$MASTER_LOG"
done

echo "=== All facebook recheck runs finished ===" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
