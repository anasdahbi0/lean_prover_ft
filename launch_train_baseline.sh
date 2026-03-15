#!/usr/bin/env bash
# ============================================================
# launch_train_baseline.sh
#
# Fine-tunes Qwen3-4B on lean_workbook_train.jsonl
# using LoRA rank=32 on a single H100 SXM.
#
# Expected time : ~1.5 hours (3 epochs, 29.7K examples)
#
# Usage:
#   bash /workspace/lean_prover_ft/launch_train_baseline.sh
# ============================================================

set -euo pipefail

WORKSPACE=/workspace
REPO=/workspace/lean_prover_ft
RESULTS=$REPO/results
LOG=$RESULTS/train_baseline.log

echo "================================================"
echo " Baseline Training: Qwen3-4B LoRA rank=32"
echo " $(date)"
echo "================================================"

# ----------------------------------------------------------
# 1. Python packages
# ----------------------------------------------------------
echo ""
echo "[1/4] Installing Python packages..."
pip install -q transformers peft torch accelerate requests pyyaml datasets huggingface_hub
echo "  OK"

# ----------------------------------------------------------
# 2. Download training data
# ----------------------------------------------------------
echo ""
echo "[2/4] Preparing lean_workbook_train.jsonl..."
cd $REPO
mkdir -p data results

if [ ! -f "data/lean_workbook_train.jsonl" ]; then
    echo "  Downloading Lean-workbook-proofs..."
    python prepare_data.py
else
    echo "  Already present: $(wc -l < data/lean_workbook_train.jsonl) examples"
fi

# ----------------------------------------------------------
# 3. GPU check
# ----------------------------------------------------------
echo ""
echo "[3/4] Checking GPU..."
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU found!'
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  GPU: {gpu}  ({mem:.0f} GB)')
"

# ----------------------------------------------------------
# 4. Train
# ----------------------------------------------------------
echo ""
echo "[4/4] Starting training..."
echo "  Config : config_baseline_r32.yaml"
echo "  Log    : $LOG"
echo ""

mkdir -p $RESULTS

python $REPO/train.py --config config_baseline_r32.yaml 2>&1 | tee $LOG

echo ""
echo "================================================"
echo " Training complete — $(date)"
echo " Adapter saved to: checkpoints/qwen-baseline-r32/"
echo " Log: $LOG"
echo "================================================"
