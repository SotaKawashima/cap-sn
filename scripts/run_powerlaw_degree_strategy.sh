#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUST_BIN="target/release/v2"
RUNTIME_CONF="v2/test_2/runtime.toml"
AGENT_CONF="v2/test_2/agent/agent-type6.toml"
OUTPUT_ROOT="powerlaw_degree_strategy_runs"
NETWORK_CONFIG_DIR="v2/test_2/network"

if [ ! -x "$RUST_BIN" ]; then
  echo "Error: $RUST_BIN が見つからないか、実行できません。先に cargo build --release を実行してください。"
  exit 1
fi

shopt -s nullglob
NETWORK_CONFIGS=("$NETWORK_CONFIG_DIR"/network-pld_*.toml)
shopt -u nullglob

if [ "${#NETWORK_CONFIGS[@]}" -eq 0 ]; then
  echo "Error: network-pld_*.toml が見つかりません。先に以下を実行してください。"
  echo "  .venv/bin/python scripts/prepare_powerlaw_degree_experiment.py"
  exit 1
fi

STRATEGIES=(
  "balance:v2/test_2/strategy/strategy-balance.toml"
  "effective_high:v2/test_2/strategy/strategy-effective-high.toml"
  "certainty_high:v2/test_2/strategy/strategy-certainty-high.toml"
)

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$OUTPUT_ROOT/logs"
MASTER_LOG="$OUTPUT_ROOT/logs/run_all_${TIMESTAMP}.log"

echo "=== Start powerlaw-degree strategy batch ===" | tee -a "$MASTER_LOG"
echo "Output root: $OUTPUT_ROOT" | tee -a "$MASTER_LOG"
echo "Runtime    : $RUNTIME_CONF" | tee -a "$MASTER_LOG"
echo "Agent      : $AGENT_CONF" | tee -a "$MASTER_LOG"
echo "Networks   : ${#NETWORK_CONFIGS[@]}" | tee -a "$MASTER_LOG"
echo "Time       : $(date)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

for NETWORK_CONF in "${NETWORK_CONFIGS[@]}"; do
  NETWORK_BASENAME="$(basename "$NETWORK_CONF" .toml)"
  NETWORK_NAME="${NETWORK_BASENAME#network-}"

  for STRATEGY_ENTRY in "${STRATEGIES[@]}"; do
    STRATEGY_NAME="${STRATEGY_ENTRY%%:*}"
    STRATEGY_CONF="${STRATEGY_ENTRY#*:}"

    IDENTIFIER="${NETWORK_NAME}_${STRATEGY_NAME}"
    RESULT_DIR="$OUTPUT_ROOT/$NETWORK_NAME/$STRATEGY_NAME/result"
    RUN_LOG="$OUTPUT_ROOT/$NETWORK_NAME/$STRATEGY_NAME/run_${TIMESTAMP}.log"

    mkdir -p "$RESULT_DIR" "$(dirname "$RUN_LOG")"

    echo "========================================" | tee -a "$MASTER_LOG"
    echo "Start: network=$NETWORK_NAME, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
    echo "Time : $(date)" | tee -a "$MASTER_LOG"
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

    echo "Finished: network=$NETWORK_NAME, strategy=$STRATEGY_NAME" | tee -a "$MASTER_LOG"
    echo "Time    : $(date)" | tee -a "$MASTER_LOG"
    echo | tee -a "$MASTER_LOG"
  done
done

echo "=== All runs finished ===" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
