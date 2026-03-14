#!/usr/bin/env bash
# =============================================================================
# run_pilot.sh  –  Option A pilot: data → generate → verify → report
#
# Prerequisites (already set up from Experiment 1):
#   - lean_server.py  running  (or this script starts it)
#   - vllm installed  (pip install vllm)
#   - HF_HOME set if needed:  export HF_HOME=/tmp/anas_hf_cache
#
# Usage:
#   bash run_pilot.sh [--skip-lean-server] [--template A|B|C]
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
WORKSPACE="/workspace/Goedel-Prover/mathlib4"
LEAN_PORT=8000
N_SAMPLES=4          # proofs per problem in Phase 3
TENSOR_PARALLEL=2    # GPUs for Goedel-Formalizer-V2-8B
VERIFY_WORKERS=32

SKIP_LEAN_SERVER=0
TEMPLATE_ARG=""

for arg in "$@"; do
  case "$arg" in
    --skip-lean-server) SKIP_LEAN_SERVER=1 ;;
    --template) shift; TEMPLATE_ARG="--template $1" ;;
    --template=*) TEMPLATE_ARG="--template ${arg#*=}" ;;
  esac
done

mkdir -p "$LOG_DIR"
export PATH="$HOME/.elan/bin:$PATH"

echo "============================================================"
echo "  Option A Pilot  –  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "  Working dir: $SCRIPT_DIR"
echo "============================================================"

# ── Step 0: Lean server ───────────────────────────────────────────────────────
if [ "$SKIP_LEAN_SERVER" -eq 0 ]; then
  if curl -s "http://localhost:$LEAN_PORT" > /dev/null 2>&1 || \
     curl -sf -X POST "http://localhost:$LEAN_PORT" \
          -H "Content-Type: application/json" \
          -d '{"cmd":"#check Nat","allTactics":false,"ast":false,"tactics":false,"premises":false}' \
          > /dev/null 2>&1; then
    echo "[0] Lean server already running on port $LEAN_PORT"
  else
    echo "[0] Starting Lean server ..."
    nohup python "$SCRIPT_DIR/lean_server.py" \
      --workspace "$WORKSPACE" \
      --port "$LEAN_PORT" \
      > "$LOG_DIR/lean_server.log" 2>&1 &
    LEAN_PID=$!
    echo "    PID=$LEAN_PID  log=$LOG_DIR/lean_server.log"
    sleep 5
  fi
else
  echo "[0] Skipping Lean server start (--skip-lean-server)"
fi

# ── Step 1: Data preparation ─────────────────────────────────────────────────
echo ""
echo "[1] Phase 1: Data exploration + pilot subset ..."
python "$SCRIPT_DIR/pilot_data.py" \
  2>&1 | tee "$LOG_DIR/pilot_data.log"

# ── Step 2: Generation ───────────────────────────────────────────────────────
echo ""
echo "[2] Phases 2-3: Micro-batch + full generation ..."
python "$SCRIPT_DIR/pilot_generate.py" \
  --n-samples "$N_SAMPLES" \
  --tensor-parallel "$TENSOR_PARALLEL" \
  $TEMPLATE_ARG \
  2>&1 | tee "$LOG_DIR/pilot_generate.log"

# ── Step 3: Verification ─────────────────────────────────────────────────────
echo ""
echo "[3] Phase 4: Lean verification ..."
python "$SCRIPT_DIR/pilot_verify.py" \
  --workers "$VERIFY_WORKERS" \
  2>&1 | tee "$LOG_DIR/pilot_verify.log"

# ── Step 4: Report ───────────────────────────────────────────────────────────
echo ""
echo "[4] Phase 5: Analysis and recommendation ..."
python "$SCRIPT_DIR/pilot_report.py" \
  2>&1 | tee "$LOG_DIR/pilot_report.log"

echo ""
echo "============================================================"
echo "  Pilot complete.  See logs/ for full output."
echo "  Key files:"
echo "    data/pilot_subset.jsonl     – 2K stratified rows"
echo "    data/pilot_candidates.jsonl – generated proof candidates"
echo "    data/pilot_verified.jsonl   – verification results"
echo "    logs/pilot_report.log       – final decision"
echo "============================================================"
