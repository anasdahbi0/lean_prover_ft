#!/usr/bin/env python3
"""
Experiment 2: Continue training from Exp 1 LoRA on the mixed synthesis+repair dataset.

Strategy:
  1. Load base model (Qwen3-4B-Instruct-2507)
  2. Load Exp 1 LoRA checkpoint and MERGE into base weights
  3. Apply a fresh LoRA (r=32, alpha=64, all proj layers)
  4. Train on mixed_train.jsonl for 1 epoch with SFTTrainer
  5. Save final adapter

The merge-then-retrain approach preserves Exp 1's synthesis capability
as the new initialisation point, then adds repair capability without
touching the merged weights directly.

Run (single GPU, for testing):
    python train_exp2.py --config config_exp2_mixed.yaml

Run (2 GPUs with accelerate):
    accelerate launch --num_processes 2 train_exp2.py --config config_exp2_mixed.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    resolved = SCRIPT_DIR / path if not os.path.isabs(path) else Path(path)
    with open(resolved) as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(SCRIPT_DIR / val)
    return cfg


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_exp2_mixed.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    log_path = os.path.join(cfg["paths"]["results_dir"], "train_exp2.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    model_name      = cfg["model"]["name"]
    lora_checkpoint = cfg["model"].get("lora_checkpoint", "")
    output_dir      = cfg["paths"]["checkpoint_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Model: base → merge Exp 1 LoRA → apply fresh Exp 2 LoRA
    # ------------------------------------------------------------------
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    if lora_checkpoint:
        logger.info(f"Loading Exp 1 LoRA checkpoint: {lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint)
        logger.info("Merging Exp 1 LoRA into base weights ...")
        model = model.merge_and_unload()
        logger.info("Merge complete.")
    else:
        logger.warning(
            "No lora_checkpoint specified — training from base model (not recommended "
            "for Exp 2; set model.lora_checkpoint in config_exp2_mixed.yaml)."
        )

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resv  = torch.cuda.memory_reserved()  / 1e9
        logger.info(f"GPU memory after merge: {alloc:.1f}GB alloc, {resv:.1f}GB reserved")

    # Apply fresh LoRA for Exp 2
    lc = cfg["lora"]
    lora_config = LoraConfig(
        r             = lc["r"],
        lora_alpha    = lc["lora_alpha"],
        target_modules= lc["target_modules"],
        lora_dropout  = lc["lora_dropout"],
        bias          = lc["bias"],
        task_type     = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset  (each row must have a "messages" key)
    # ------------------------------------------------------------------
    train_path = cfg["paths"]["train_data"]
    logger.info(f"Loading training data: {train_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run: python prepare_mixed_dataset.py"
        )
    train_data = load_jsonl(train_path)
    logger.info(f"Training examples: {len(train_data):,}")

    # Verify schema
    assert "messages" in train_data[0], (
        f"Expected 'messages' key in training data, got: {list(train_data[0].keys())}"
    )
    train_dataset = Dataset.from_list(train_data)

    eval_dataset = None
    val_path = cfg["paths"].get("val_data", "")
    if val_path and os.path.exists(val_path):
        val_data     = load_jsonl(val_path)
        eval_dataset = Dataset.from_list(val_data)
        logger.info(f"Validation examples: {len(val_data):,}")
    else:
        logger.info("No validation data found — running without eval.")

    # ------------------------------------------------------------------
    # SFTConfig
    # ------------------------------------------------------------------
    tcfg    = cfg["training"]
    hf_cfg  = cfg.get("huggingface", {})
    hub_id  = hf_cfg.get("hub_model_id") if hf_cfg.get("push_to_hub") else None

    sft_config = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = tcfg["num_epochs"],
        per_device_train_batch_size = tcfg["per_device_batch_size"],
        gradient_accumulation_steps = tcfg["gradient_accumulation_steps"],
        learning_rate               = tcfg["learning_rate"],
        lr_scheduler_type           = tcfg["lr_scheduler"],
        warmup_ratio                = tcfg["warmup_ratio"],
        bf16                        = tcfg["bf16"],
        gradient_checkpointing      = tcfg.get("gradient_checkpointing", True),
        max_length                  = tcfg["max_length"],
        logging_steps               = tcfg["logging_steps"],
        eval_strategy               = "steps" if eval_dataset is not None else "no",
        **( {"eval_steps": tcfg.get("eval_steps", 500)} if eval_dataset is not None else {} ),
        save_strategy               = tcfg["save_strategy"],
        save_steps                  = tcfg.get("save_steps", 500),
        save_total_limit            = tcfg["save_total_limit"],
        # completion_only_loss=True: mask loss on system+user turns,
        # train only on assistant tokens. Works for both synthesis and repair.
        completion_only_loss        = True,
        dataset_text_field          = None,   # use "messages" column directly
        report_to                   = "none",
        dataloader_num_workers      = 0,
        hub_model_id                = hub_id,
        push_to_hub                 = bool(hub_id),
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model            = model,
        args             = sft_config,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        processing_class = tokenizer,   # TRL >= 0.9; older TRL: use tokenizer=tokenizer
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("\nStarting Experiment 2 training ...")
    logger.info(f"  Base model    : {model_name}")
    logger.info(f"  Exp 1 LoRA    : {lora_checkpoint or '(none)'}")
    logger.info(f"  Train examples: {len(train_data):,}")
    logger.info(f"  Max seq length: {tcfg['max_length']}")
    logger.info(f"  LoRA rank     : {lc['r']}, alpha: {lc['lora_alpha']}")
    logger.info(f"  LR            : {tcfg['learning_rate']}")
    logger.info(f"  Epochs        : {tcfg['num_epochs']}")
    logger.info(f"  Output dir    : {output_dir}\n")

    resume_ckpt = cfg.get("resume_from_checkpoint", False)
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info(f"Saving Exp 2 adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if hub_id:
        logger.info(f"Pushing adapter to HuggingFace: {hub_id}")
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        logger.info(f"Pushed to {hub_id}")

    logger.info("Training complete.")
    logger.info(
        f"\nNext steps:\n"
        f"  # Experiment A — standard synthesis eval:\n"
        f"  python evaluate.py --adapter {output_dir} "
        f"--dataset data/minif2f_test.jsonl --output results/exp_a.json\n\n"
        f"  # Experiment B — self-correction eval:\n"
        f"  python eval_self_correct.py --adapter {output_dir} "
        f"--max_rounds 8 --run_sampling_baseline --output_dir results/exp_b"
    )


if __name__ == "__main__":
    main()
