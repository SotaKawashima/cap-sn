#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
./scripts/run_optimize.sh wiki-vote "${1:-100}" "${2:-auc}" "${3:-}" "${4:-3}" "${5:-1}"
