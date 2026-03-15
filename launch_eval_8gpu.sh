#!/usr/bin/env bash
# ============================================================
# launch_eval_8gpu.sh  —  Parallel eval on 8× H200 SXM
#
# Layout:
#   GPU 0-3 : Exp A  (evaluate.py with qwen-exp-a)
#   GPU 4-7 : Exp B  (eval_two_model.py with qwen + qwen-corrector)
#   CPU     : Lean REPL server (shared, ThreadedHTTPServer)
#
# Problem splitting: 244 problems / 4 GPUs = 61 per shard
# Both experiments run in parallel → total ~1.5 hours
#
# Usage:
#   bash /workspace/lean_prover_ft/launch_eval_8gpu.sh
# ============================================================

set -euo pipefail

WORKSPACE=/workspace
MATHLIB=/workspace/Goedel-Prover/mathlib4
REPO=/workspace/lean_prover_ft
RESULTS=$REPO/results
SERVER_PORT=8000
N_GPUS_PER_EXP=4   # 4 GPUs for Exp A, 4 for Exp B

echo "================================================"
echo " 8× H200 Parallel Eval Launch"
echo " $(date)"
echo "================================================"

# ----------------------------------------------------------
# 1. Sanity check
# ----------------------------------------------------------
echo ""
echo "[1] Checking environment..."
source ~/.elan/env
if [ ! -f "$MATHLIB/.lake/build/bin/repl" ]; then
    echo "ERROR: Lean repl binary not found. Run lake build first."
    exit 1
fi
echo "  Lean repl: OK"
python -c "import torch; print(f'  GPUs available: {torch.cuda.device_count()}')"

# ----------------------------------------------------------
# 2. Python packages
# ----------------------------------------------------------
echo ""
echo "[2] Installing Python packages..."
pip install -q transformers peft torch accelerate requests pyyaml datasets huggingface_hub

# ----------------------------------------------------------
# 3. Prepare data
# ----------------------------------------------------------
echo ""
echo "[3] Preparing datasets..."
cd $REPO
mkdir -p data results

if [ ! -f "data/minif2f_test.jsonl" ]; then
    python prepare_data.py
else
    echo "  miniF2F: $(wc -l < data/minif2f_test.jsonl) problems (cached)"
fi

# Split dataset into N_GPUS_PER_EXP shards
echo "  Splitting into $N_GPUS_PER_EXP shards..."
python - << 'EOF'
import json, math, os

with open("data/minif2f_test.jsonl") as f:
    problems = [json.loads(l) for l in f if l.strip()]

n = len(problems)
n_shards = int(os.environ.get("N_GPUS_PER_EXP", 4))
shard_size = math.ceil(n / n_shards)

os.makedirs("data/shards", exist_ok=True)
for i in range(n_shards):
    chunk = problems[i*shard_size:(i+1)*shard_size]
    path = f"data/shards/minif2f_shard_{i}.jsonl"
    with open(path, "w") as f:
        for p in chunk:
            f.write(json.dumps(p) + "\n")
    print(f"  Shard {i}: {len(chunk)} problems → {path}")
EOF

# ----------------------------------------------------------
# 4. Start Lean REPL server (shared by all 8 GPU processes)
# ----------------------------------------------------------
echo ""
echo "[4] Starting Lean REPL server (port $SERVER_PORT)..."
fuser -k ${SERVER_PORT}/tcp 2>/dev/null || true
sleep 1

nohup python $REPO/lean_server.py \
    --workspace $MATHLIB \
    --port $SERVER_PORT \
    --timeout 120 \
    > $RESULTS/lean_server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server
echo "  Waiting for server warm-up (30s)..."
sleep 30
SMOKE=$(curl -s -X POST http://localhost:$SERVER_PORT \
    -H "Content-Type: application/json" \
    -d '{"cmd":"#check Nat","allTactics":false,"ast":false,"tactics":false,"premises":false}' \
    --max-time 10 || echo "timeout")
echo "  Smoke test: $SMOKE"

mkdir -p $RESULTS/exp_a_shards $RESULTS/exp_b_shards

# ----------------------------------------------------------
# 5. Launch Exp A on GPUs 0–3 (4 parallel shards)
# ----------------------------------------------------------
echo ""
echo "[5] Launching Exp A on GPUs 0-$((N_GPUS_PER_EXP-1)) (4 parallel shards)..."

for i in 0 1 2 3; do
    SHARD_FILE="$REPO/data/shards/minif2f_shard_${i}.jsonl"
    OUTPUT="$RESULTS/exp_a_shards/shard_${i}.json"
    LOG="$RESULTS/exp_a_shards/shard_${i}.log"

    CUDA_VISIBLE_DEVICES=$i nohup python $REPO/evaluate.py \
        --config  config_exp_a.yaml \
        --adapter yotsubian/qwen-exp-a \
        --dataset $SHARD_FILE \
        --output  $OUTPUT \
        > $LOG 2>&1 &

    echo "  GPU $i: shard $i  PID $!  → $LOG"
done

# ----------------------------------------------------------
# 6. Launch Exp B on GPUs 4–7 (4 parallel shards)
# ----------------------------------------------------------
echo ""
echo "[6] Launching Exp B on GPUs 4-7 (4 parallel shards)..."

for i in 0 1 2 3; do
    GPU_ID=$((i + 4))
    SHARD_FILE="$REPO/data/shards/minif2f_shard_${i}.jsonl"
    OUTPUT_DIR="$RESULTS/exp_b_shards/shard_${i}/"
    LOG="$RESULTS/exp_b_shards/shard_${i}.log"

    mkdir -p $OUTPUT_DIR

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python $REPO/eval_two_model.py \
        --config             config_corrector.yaml \
        --generator_adapter  yotsubian/qwen \
        --corrector_adapter  yotsubian/qwen-corrector \
        --dataset            $SHARD_FILE \
        --max_rounds         12 \
        --no_sampling_baseline \
        --output_dir         $OUTPUT_DIR \
        > $LOG 2>&1 &

    echo "  GPU $GPU_ID: shard $i  PID $!  → $LOG"
done

# ----------------------------------------------------------
# 7. Wait and merge
# ----------------------------------------------------------
echo ""
echo "[7] Waiting for all shards to complete..."
echo "    Monitor: tail -f $RESULTS/exp_a_shards/shard_0.log"
echo ""

wait  # Wait for all background jobs

echo ""
echo "[8] Merging results..."

# Merge Exp A
python $REPO/merge_results.py \
    --mode   exp_a \
    --inputs "$RESULTS/exp_a_shards/shard_*.json" \
    --output "$RESULTS/exp_a.json"

# Merge Exp B
python $REPO/merge_results.py \
    --mode       exp_b \
    --inputs     "$RESULTS/exp_b_shards/shard_*/per_problem.json" \
    --output_dir "$RESULTS/exp_b/"

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "================================================"
echo " ALL DONE  —  $(date)"
echo "================================================"
echo " Exp A results : $RESULTS/exp_a.json"
echo " Exp B results : $RESULTS/exp_b/"
echo ""
echo " Quick summary:"
python - << 'EOF'
import json
try:
    with open("/workspace/lean_prover_ft/results/exp_a.json") as f:
        a = json.load(f)
    print(f" Exp A — {a['adapter']}")
    for k, v in a['pass_at_k'].items():
        print(f"   {k}: {v:.4f}  ({v*100:.1f}%)")
except Exception as e:
    print(f" Exp A: {e}")

try:
    with open("/workspace/lean_prover_ft/results/exp_b/accuracy_by_round.json") as f:
        b = json.load(f)
    print(f" Exp B — self-correction accuracy by round:")
    for rnd, acc in sorted(b.items(), key=lambda x: int(x[0])):
        print(f"   Round {rnd}: {acc:.1f}%")
except Exception as e:
    print(f" Exp B: {e}")
EOF
echo "================================================"
