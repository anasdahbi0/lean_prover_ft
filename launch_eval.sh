#!/usr/bin/env bash
# ============================================================
# launch_eval.sh  —  Full evaluation pipeline launcher
#
# Run AFTER lake build finishes in Goedel-Prover/mathlib4.
#
# Usage:
#   bash /workspace/lean_prover_ft/launch_eval.sh
#
# What it does:
#   1. Verifies mathlib4 build is complete
#   2. Installs Python dependencies
#   3. Downloads miniF2F test set
#   4. Starts Lean REPL server (port 8000)
#   5. Tests the server with a trivial proof
#   6. Launches Exp A eval  (yotsubian/qwen-exp-a on miniF2F, pass@32)
#   7. Launches Exp B eval  (Generator + Corrector pipeline)
# ============================================================

set -euo pipefail

WORKSPACE=/workspace
MATHLIB=/workspace/Goedel-Prover/mathlib4
REPO=/workspace/lean_prover_ft
RESULTS=$REPO/results
SERVER_PORT=8000
SERVER_LOG=$WORKSPACE/lean_server.log

echo "================================================"
echo " Lean Eval Launch Script"
echo " $(date)"
echo "================================================"

# ----------------------------------------------------------
# 1. Sanity check: mathlib4 build must be complete
# ----------------------------------------------------------
echo ""
echo "[1/7] Checking mathlib4 build..."
REPL_BIN=$MATHLIB/.lake/build/bin/repl
if [ ! -f "$REPL_BIN" ]; then
    echo "ERROR: $REPL_BIN not found."
    echo "  Run: cd $MATHLIB && lake build"
    echo "  Then re-run this script."
    exit 1
fi
echo "  OK — repl binary found: $REPL_BIN"
echo "  Build size: $(du -sh $MATHLIB/.lake/build/ | cut -f1)"

# ----------------------------------------------------------
# 2. Python environment
# ----------------------------------------------------------
echo ""
echo "[2/7] Installing Python packages..."
pip install -q transformers peft torch accelerate requests pyyaml datasets huggingface_hub
echo "  OK"

# ----------------------------------------------------------
# 3. Clone / update eval repo
# ----------------------------------------------------------
echo ""
echo "[3/7] Setting up eval code..."
if [ -d "$REPO/.git" ]; then
    echo "  Already cloned — pulling latest..."
    cd $REPO && git pull --quiet || true
else
    echo "  Cloning from GitHub..."
    git clone https://github.com/anasdahbi0/lean-model-finetuning.git $REPO
fi

# ----------------------------------------------------------
# 4. Download miniF2F test set
# ----------------------------------------------------------
echo ""
echo "[4/7] Preparing miniF2F dataset..."
cd $REPO
mkdir -p data
if [ ! -f "data/minif2f_test.jsonl" ]; then
    echo "  Downloading..."
    python prepare_data.py
else
    echo "  Already present: $(wc -l < data/minif2f_test.jsonl) problems"
fi

# ----------------------------------------------------------
# 5. Start Lean REPL server
# ----------------------------------------------------------
echo ""
echo "[5/7] Starting Lean REPL server on port $SERVER_PORT..."

# Kill any existing server on this port
fuser -k ${SERVER_PORT}/tcp 2>/dev/null || true
sleep 1

source ~/.elan/env
nohup python $REPO/lean_server.py \
    --workspace $MATHLIB \
    --port $SERVER_PORT \
    --timeout 120 \
    > $SERVER_LOG 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready (up to 30s)
echo "  Waiting for server..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" \
        -X POST http://localhost:$SERVER_PORT \
        -H "Content-Type: application/json" \
        -d '{"cmd":"#check Nat","allTactics":false,"ast":false,"tactics":false,"premises":false}' \
        2>/dev/null | grep -q "200"; then
        echo "  Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Quick smoke test
echo "  Smoke test: #check Nat..."
SMOKE=$(curl -s -X POST http://localhost:$SERVER_PORT \
    -H "Content-Type: application/json" \
    -d '{"cmd":"#check Nat","allTactics":false,"ast":false,"tactics":false,"premises":false}')
echo "  Response: $SMOKE"

# Test a real proof
echo "  Proof test: 1 + 1 = 2..."
PROOF_TEST=$(curl -s -X POST http://localhost:$SERVER_PORT \
    -H "Content-Type: application/json" \
    -d "{\"cmd\":\"import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\nexample : 1 + 1 = 2 := by norm_num\",\"allTactics\":false,\"ast\":false,\"tactics\":false,\"premises\":false}")
echo "  Response: $PROOF_TEST"

mkdir -p $RESULTS

# ----------------------------------------------------------
# 6. Exp A — yotsubian/qwen-exp-a on miniF2F
# ----------------------------------------------------------
echo ""
echo "[6/7] Launching Exp A (qwen-exp-a, pass@32 on miniF2F)..."
echo "  Adapter : yotsubian/qwen-exp-a"
echo "  Dataset : data/minif2f_test.jsonl"
echo "  Output  : $RESULTS/exp_a/"
echo ""

nohup python $REPO/evaluate.py \
    --config  config_exp_a.yaml \
    --adapter yotsubian/qwen-exp-a \
    --dataset data/minif2f_test.jsonl \
    --output  $RESULTS/exp_a.json \
    > $RESULTS/exp_a.log 2>&1 &
EXP_A_PID=$!
echo "  Exp A PID: $EXP_A_PID"
echo "  Log: tail -f $RESULTS/exp_a.log"

# ----------------------------------------------------------
# 7. Exp B — Generator (yotsubian/qwen) + Corrector (yotsubian/qwen-corrector)
# ----------------------------------------------------------
echo ""
echo "[7/7] Launching Exp B (two-model self-correction)..."
echo "  Generator : yotsubian/qwen"
echo "  Corrector : yotsubian/qwen-corrector"
echo "  Output    : $RESULTS/exp_b/"
echo ""

# Note: Exp B loads two LoRA adapters into one base model (~8GB VRAM)
# Run on same GPU as Exp A only if VRAM allows; otherwise stagger.
# With H100 80GB, both fit. Launching with 30s delay to avoid HF download collision.
sleep 30

nohup python $REPO/eval_two_model.py \
    --config             config_corrector.yaml \
    --generator_adapter  yotsubian/qwen \
    --corrector_adapter  yotsubian/qwen-corrector \
    --dataset            data/minif2f_test.jsonl \
    --max_rounds         12 \
    --output_dir         $RESULTS/exp_b/ \
    > $RESULTS/exp_b.log 2>&1 &
EXP_B_PID=$!
echo "  Exp B PID: $EXP_B_PID"
echo "  Log: tail -f $RESULTS/exp_b.log"

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "================================================"
echo " All jobs launched."
echo "================================================"
echo " Lean server : PID $SERVER_PID   (port $SERVER_PORT)"
echo " Exp A       : PID $EXP_A_PID   → $RESULTS/exp_a.log"
echo " Exp B       : PID $EXP_B_PID   → $RESULTS/exp_b.log"
echo ""
echo " Monitor both:"
echo "   tail -f $RESULTS/exp_a.log $RESULTS/exp_b.log"
echo ""
echo " Expected completion: 5-8 hours from now"
echo " Results will be in: $RESULTS/"
echo "================================================"
