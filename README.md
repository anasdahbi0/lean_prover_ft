# Qwen3-4B Lean 4 Fine-Tuning

Fine-tunes [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on Lean 4 theorem proving using LoRA.
Training data from [Goedel-LM/Lean-workbook-proofs](https://huggingface.co/datasets/Goedel-LM/Lean-workbook-proofs) (~29K examples).
Evaluated on [miniF2F](https://huggingface.co/datasets/cat-searcher/minif2f-lean4) with pass@k metrics.

Prompt format and Lean header copied exactly from [Goedel-Prover](https://github.com/Goedel-LM/Goedel-Prover).

## Reference results (from paper)

| Model                        | pass@32 |
|------------------------------|---------|
| Goedel-Prover-SFT (7B)       | 0.576   |
| DeepSeek-Prover-V2-7B        | 0.734   |

## Project structure

```
config.yaml          # All hyperparameters and paths
prepare_data.py      # Download and format training data
train.py             # LoRA fine-tuning with HuggingFace Trainer
evaluate.py          # pass@k evaluation against Lean verifier
results_table.py     # Print comparison table from eval JSON files
```

## Setup

```bash
pip install transformers peft datasets accelerate bitsandbytes pyyaml
export HF_HOME=/workspace/hf_cache   # optional: redirect model cache
```

## Run order

```bash
# 1. Download and prepare data (~29.7K training + 244 miniF2F test)
python prepare_data.py

# 2. Train (single GPU)
python train.py

# 2. Train (multi-GPU, e.g. 3 GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=3 train.py

# 3. Evaluate baseline (no adapter)
python evaluate.py \
    --dataset data/minif2f_test.jsonl \
    --output results/eval_baseline.json

# 4. Evaluate fine-tuned
python evaluate.py \
    --adapter checkpoints/qwen3_4b_lean \
    --dataset data/minif2f_test.jsonl \
    --output results/eval_finetuned.json

# 5. Print comparison table
python results_table.py results/eval_baseline.json results/eval_finetuned.json
```

## Configuration

Key settings in `config.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| model | Qwen/Qwen3-4B | Base instruct model |
| LoRA r | 16 | Rank |
| LoRA alpha | 32 | Scaling |
| target modules | q/k/v/o_proj | Attention layers |
| batch size | 2 per GPU | With grad accum 3 |
| effective batch | ~18 | 2 × 3 GPUs × 3 accum |
| max length | 1024 | Truncates ~5% of examples |
| epochs | 3 | |
| learning rate | 2e-4 | Cosine schedule |

## Evaluation requirements

`evaluate.py` requires a running Lean 4 verification server at `http://localhost:8000`
(configurable via `lean_server.url` in `config.yaml`).
See [Goedel-Prover](https://github.com/Goedel-LM/Goedel-Prover) for server setup instructions.
