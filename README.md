# Qwen3-4B Lean 4 Theorem Proving

Fine-tunes [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) on Lean 4 theorem proving using LoRA. Training data from [Goedel-LM/Lean-workbook-proofs](https://huggingface.co/datasets/Goedel-LM/Lean-workbook-proofs) (~29K examples). Evaluated on [miniF2F](https://huggingface.co/datasets/cat-searcher/minif2f-lean4) (244 problems) using pass@k metrics.

Prompt format and Lean 4 header follow [Goedel-Prover](https://github.com/Goedel-LM/Goedel-Prover) exactly.

## Results

| Model | pass@1 | pass@8 | pass@32 | Adapter |
|---|---|---|---|---|
| Qwen3-4B (baseline) | 0.0% | 0.0% | 0.0% | — |
| Qwen3-4B + LoRA r=16 | 0.86% | 4.47% | **9.02%** | [yotsubian/qwen](https://huggingface.co/yotsubian/qwen) |
| Goedel-Prover-SFT (7B) | — | — | 57.6% | (reference) |
| DeepSeek-Prover-V2-7B | — | — | 73.4% | (reference) |

## Project Structure

```
config.yaml          # Training config (LoRA r=16)
prepare_data.py      # Download and format training/eval data
train.py             # LoRA fine-tuning with HuggingFace Trainer
lean_server.py       # Lean 4 REPL verification server (aiohttp)
evaluate.py          # Proof generation + pass@k evaluation
results_table.py     # Print comparison table from eval JSON files
test_lean_server.py  # Smoke tests for the Lean server
```

## Setup

```bash
pip install transformers peft datasets accelerate bitsandbytes aiohttp pyyaml
# For flash attention (recommended, requires matching CUDA/torch versions):
pip install flash-attn --no-build-isolation
export HF_HOME=/workspace/hf_cache  # optional
```

## Run Order

### 1. Prepare data

```bash
python prepare_data.py
# Outputs: data/lean_workbook_train.jsonl (~29.7K examples)
#          data/minif2f_test.jsonl (244 problems)
```

### 2. Train

```bash
# Single GPU
python train.py

# Multi-GPU (e.g. 3 GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=3 train.py
```

### 3. Start Lean verification server

```bash
# Requires: Lean 4 + Lake + Mathlib4 built
python lean_server.py --workspace /path/to/mathlib4 --port 8000

# Verify server is working
python test_lean_server.py
```

### 4. Evaluate

```bash
# Baseline (no adapter)
python evaluate.py \
    --dataset data/minif2f_test.jsonl \
    --output results/eval_baseline.json

# Fine-tuned model
python evaluate.py \
    --adapter checkpoints/qwen3_4b_lean \
    --dataset data/minif2f_test.jsonl \
    --output results/eval_finetuned.json
```

### 5. View results

```bash
python results_table.py results/eval_baseline.json results/eval_finetuned.json
```

### 6. Run demo

Generates a proof for a miniF2F problem and verifies it with the Lean REPL.

```bash
# Start the Lean server (in a separate terminal)
python lean_server.py --workspace /path/to/mathlib4 --port 8000

# Run the demo
python demo.py
```

## Configuration

Key hyperparameters (`config.yaml`):

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-4B-Instruct-2507 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o_proj |
| Dropout | 0.05 |
| Epochs | 3 |
| Batch size | 2 per GPU |
| Gradient accumulation | 3 steps |
| Learning rate | 2e-4 (cosine) |
| Max sequence length | 1024 |
| Precision | bfloat16 |
| Attention | flash_attention_2 |

Evaluation uses 32 samples per problem at temperature=1.0, verified by Lean REPL.

## Hardware

Trained on a single NVIDIA GPU with 24GB VRAM. Training takes ~3 hours for 3 epochs over 29.7K examples at max_length=1024 with flash attention.
