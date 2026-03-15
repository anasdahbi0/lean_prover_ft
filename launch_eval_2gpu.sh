#!/usr/bin/env bash
# ============================================================
# launch_eval_2gpu.sh  —  Eval on 2 GPUs
#
# Layout:
#   GPU 0   : Exp A  (evaluate.py, full 244 problems)
#   GPU 1   : Exp B  (eval_two_model.py, full 244 problems)
#   CPU     : 2 Lean servers (port 8000 for Exp A, 8002 for Exp B)
#
# Estimated runtime: ~12-16 hours
#
# Usage:
#   bash /workspace/lean_prover_ft/launch_eval_2gpu.sh
# ============================================================

set -euo pipefail

MATHLIB=/workspace/Goedel-Prover/mathlib4
REPO=/workspace/lean_prover_ft
RESULTS=$REPO/results

echo "================================================"
echo " 2-GPU Eval Launch"
echo " $(date)"
echo "================================================"

# ----------------------------------------------------------
# 1. Environment check
# ----------------------------------------------------------
echo ""
echo "[1] Checking environment..."
source ~/.elan/env
echo "  lake: $(lake --version)"
echo "  GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# ----------------------------------------------------------
# 2. Install packages
# ----------------------------------------------------------
echo ""
echo "[2] Installing packages..."
pip install -q aiohttp transformers peft torch accelerate requests pyyaml datasets huggingface_hub

# ----------------------------------------------------------
# 3. Prepare data
# ----------------------------------------------------------
echo ""
echo "[3] Preparing dataset..."
cd $REPO
mkdir -p data results/exp_b

if [ ! -f "data/minif2f_test.jsonl" ]; then
    python prepare_data.py
fi
echo "  miniF2F: $(wc -l < data/minif2f_test.jsonl) problems"

# ----------------------------------------------------------
# 4. Start two Lean servers (one per experiment)
# ----------------------------------------------------------
echo ""
echo "[4] Starting Lean servers..."

# Kill anything on ports 8000 and 8002
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8002/tcp 2>/dev/null || true
sleep 1

# Server for Exp A
nohup python $REPO/lean_server.py \
    --workspace $MATHLIB --port 8000 --timeout 120 --max-concurrent 4 \
    > $RESULTS/lean_server_a.log 2>&1 &
echo "  Exp A server PID: $!  (port 8000)"

# Server for Exp B
nohup python $REPO/lean_server.py \
    --workspace $MATHLIB --port 8002 --timeout 120 --max-concurrent 4 \
    > $RESULTS/lean_server_b.log 2>&1 &
echo "  Exp B server PID: $!  (port 8002)"

echo "  Waiting 15s for servers to start..."
sleep 15

# Smoke test both
echo "  Smoke test port 8000: $(curl -s http://localhost:8000/health --max-time 5 || echo FAIL)"
echo "  Smoke test port 8002: $(curl -s http://localhost:8002/health --max-time 5 || echo FAIL)"

# ----------------------------------------------------------
# 5. Launch Exp A on GPU 0 (full dataset)
# ----------------------------------------------------------
echo ""
echo "[5] Launching Exp A on GPU 0..."

CUDA_VISIBLE_DEVICES=0 nohup python $REPO/evaluate.py \
    --config  config_exp_a.yaml \
    --adapter yotsubian/qwen-exp-a \
    --dataset data/minif2f_test.jsonl \
    --output  $RESULTS/exp_a.json \
    > $RESULTS/exp_a.log 2>&1 &
echo "  Exp A PID: $!"

# ----------------------------------------------------------
# 6. Launch Exp B on GPU 1 (full dataset)
# ----------------------------------------------------------
echo ""
echo "[6] Launching Exp B on GPU 1..."

CUDA_VISIBLE_DEVICES=1 nohup python $REPO/eval_two_model.py \
    --config             config_corrector.yaml \
    --generator_adapter  yotsubian/qwen \
    --corrector_adapter  yotsubian/qwen-corrector \
    --dataset            data/minif2f_test.jsonl \
    --max_rounds         3 \
    --output_dir         $RESULTS/exp_b/ \
    > $RESULTS/exp_b.log 2>&1 &
echo "  Exp B PID: $!"

# ----------------------------------------------------------
# 7. Monitor
# ----------------------------------------------------------
echo ""
echo "================================================"
echo " Both experiments launched. Monitor with:"
echo "   tail -f $RESULTS/exp_a.log"
echo "   tail -f $RESULTS/exp_b.log"
echo "   watch -n 60 'curl -s localhost:8000/health; curl -s localhost:8002/health'"
echo "================================================"

wait

echo ""
echo "[7] All done — $(date)"
echo " Exp A: $RESULTS/exp_a.json"
echo " Exp B: $RESULTS/exp_b/"

python - << 'EOF'
import json
try:
    a = json.load(open("/workspace/lean_prover_ft/results/exp_a.json"))
    print(f"\nExp A ({a['adapter']}):")
    for k, v in a['pass_at_k'].items():
        print(f"  {k}: {v*100:.1f}%")
except Exception as e:
    print(f"Exp A: {e}")
EOF
