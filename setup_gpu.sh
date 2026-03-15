#!/bin/bash
# ============================================================
# GPU Machine Setup Script
# Run this after SSHing in and uploading lean_prover_ft/
#
# Usage:
#   bash setup_gpu.sh
#
# What it does:
#   1. Install Python dependencies
#   2. Install Lean v4.22.0-rc4 via elan
#   3. Clone mathlib4 + build Repl (~30-60 min)
#   4. Download APRIL dataset (~3 min)
#   5. Prepare training data
#   6. Start Lean server
#   7. Verify APRIL compiles (sanity check)
#   8. Launch both training jobs in parallel
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Step 1: Python dependencies"
echo "============================================================"
pip install torch transformers peft trl accelerate \
    datasets huggingface_hub requests pyyaml matplotlib -q
echo "Done."

echo "============================================================"
echo " Step 2: Install Lean v4.22.0-rc4 via elan"
echo "============================================================"
if ! command -v elan &>/dev/null; then
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    source "$HOME/.elan/env"
fi
elan install leanprover/lean4:v4.22.0-rc4
echo "Lean installed."

echo "============================================================"
echo " Step 3: Clone mathlib4 and build Repl (~30-60 min)"
echo "============================================================"
MATHLIB_DIR="$SCRIPT_DIR/mathlib4"
if [ ! -d "$MATHLIB_DIR" ]; then
    git clone https://github.com/leanprover-community/mathlib4 "$MATHLIB_DIR"
fi
echo "leanprover/lean4:v4.22.0-rc4" > "$MATHLIB_DIR/lean-toolchain"
cd "$MATHLIB_DIR"
~/.elan/bin/lake build Repl
cd "$SCRIPT_DIR"
echo "Repl built."

echo "============================================================"
echo " Step 4: Download APRIL dataset"
echo "============================================================"
if [ ! -f "data/april/train/thme_train.jsonl" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('uw-math-ai/APRIL', repo_type='dataset', local_dir='data/april')
print('APRIL downloaded.')
"
else
    echo "APRIL already present, skipping."
fi

echo "============================================================"
echo " Step 5: Prepare training data"
echo "============================================================"
if [ ! -f "data/april_train.jsonl" ]; then
    python prepare_mixed_dataset.py --april_only --output_prefix april
else
    echo "data/april_train.jsonl already exists, skipping."
fi

echo "============================================================"
echo " Step 6: Start Lean server in background"
echo "============================================================"
python lean_server.py --workspace "$MATHLIB_DIR" --port 8000 &
LEAN_SERVER_PID=$!
echo "Lean server PID: $LEAN_SERVER_PID"
echo "Waiting 10s for server to be ready..."
sleep 10

echo "============================================================"
echo " Step 7: Verify APRIL proofs compile"
echo "============================================================"
python verify_april.py
# Exits with code 1 if <90% compile — stops the script (set -e)

echo "============================================================"
echo " Step 8: Train both models in parallel"
echo "============================================================"
echo "Starting Exp A training (Exp1 + APRIL)..."
accelerate launch --num_processes 4 train_exp2.py --config config_exp_a.yaml &
EXP_A_PID=$!

echo "Starting Corrector training (base Qwen + APRIL)..."
accelerate launch --num_processes 4 train_exp2.py --config config_corrector.yaml &
CORRECTOR_PID=$!

echo "Both training jobs running."
echo "  Exp A PID     : $EXP_A_PID"
echo "  Corrector PID : $CORRECTOR_PID"
echo ""
echo "Monitor with:"
echo "  tail -f results/train_exp2.log"
echo ""

wait $EXP_A_PID
echo "Exp A training complete."
wait $CORRECTOR_PID
echo "Corrector training complete."

echo "============================================================"
echo " All done! Now run evaluation:"
echo "============================================================"
echo ""
echo "# Experiment A — pass@32 on miniF2F"
echo "python evaluate.py \\"
echo "    --adapter checkpoints/exp_a \\"
echo "    --dataset data/minif2f_test.jsonl \\"
echo "    --output results/exp_a.json"
echo ""
echo "# Experiment B — two-model self-correction"
echo "python eval_two_model.py \\"
echo "    --generator_adapter yotsubian/qwen \\"
echo "    --corrector_adapter checkpoints/corrector \\"
echo "    --output_dir results/exp_b"
