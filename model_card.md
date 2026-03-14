---
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
  - lean4
  - theorem-proving
  - lora
  - peft
---

# Qwen3-4B Lean Prover — Experiment 1

## Training
- Base model: Qwen/Qwen3-4B-Instruct-2507
- Training data: Goedel-LM/Lean-workbook-proofs (29,750 verified proof pairs)
- Fine-tuning method: LoRA (r=16, alpha=32, target: q/k/v/o_proj)
- Epochs: 3
- Per-device batch size: 2 × 3 GPUs × 3 grad accum = 18 effective
- Learning rate: 2e-4 (cosine schedule, 5% warmup)
- Max sequence length: 1024
- Hardware: 3× RTX 5090 32GB (RunPod)

## Results on miniF2F test (244 problems, pass@32)
| Model | Solved | pass@1 | pass@8 | pass@32 |
|-------|--------|--------|--------|---------|
| Baseline (no fine-tuning) | 0/244 | 0.0% | 0.0% | 0.0% |
| This checkpoint (29.7K SFT) | 22/244 | 0.86% | 4.47% | 9.02% |
| Goedel-Prover-SFT (paper) | 142/244 | — | — | 57.6% |
| DeepSeek-Prover-V2-7B (paper) | — | — | — | 73.4% |

## Reproducing
- Evaluation dataset: cat-searcher/minif2f-lean4 (test split)
- n_samples: 32 per problem
- Verifier: Lean 4 REPL server (Mathlib4)
- Code: https://github.com/anasdahbi0/lean_prover_ft
