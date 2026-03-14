#!/usr/bin/env python3
"""
Fine-tune Qwen3-4B-Instruct on Lean 4 theorem proving with LoRA.

Run:
    python train.py
    python train.py --config config.yaml
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import yaml

# All relative paths in the config are resolved from the script's own directory,
# so the script works regardless of where it is invoked from.
SCRIPT_DIR = Path(__file__).parent.resolve()
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Copied exactly from Goedel-Prover/prover/utils.py
LEAN4_DEFAULT_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "\n"
    "set_option maxHeartbeats 0\n"
    "\n"
    "open BigOperators Real Nat Topology Rat\n"
    "\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    # Resolve relative to the script directory so invocation directory doesn't matter
    resolved = SCRIPT_DIR / path if not os.path.isabs(path) else Path(path)
    with open(resolved) as f:
        cfg = yaml.safe_load(f)
    # Make all config paths absolute, resolved relative to the script directory
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(SCRIPT_DIR / val)
    return cfg


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_user_content(statement):
    """
    Prompt format matching Goedel-Prover's non_cot_prompt, adapted for chat template.
    The user turn contains the full Lean 4 file header + theorem signature.
    """
    return (
        "Complete the following Lean 4 code:\n\n"
        "```lean4\n"
        + LEAN4_DEFAULT_HEADER
        + statement
        + "\n```"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProofDataset(torch.utils.data.Dataset):
    """
    Each training example is a (statement, proof) pair from lean_workbook_train.jsonl.

    Tokenization:
      - Tokenize the full conversation (user + assistant) → input_ids
      - Tokenize the user turn only (with add_generation_prompt=True) → prompt_ids
      - Set labels = input_ids, then mask the prompt prefix with -100
        so loss is only computed on the proof tokens.
    """

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        statement = row["statement"]
        proof = row["proof"]

        user_content = build_user_content(statement)

        # Full conversation: used to produce input_ids and labels
        messages_full = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": proof},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )

        # User-only turn: used to determine where the proof starts
        messages_prompt = [{"role": "user", "content": user_content}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_enc["input_ids"]
        labels = list(input_ids)

        # Mask all prompt tokens — loss is computed only on the proof
        prompt_len = len(prompt_enc["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    log_path = os.path.join(cfg["paths"]["results_dir"], "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    model_name = cfg["model"]["name"]
    # Use hub_model_id as checkpoint subdir name if pushing to hub, else default
    hf_cfg = cfg.get("huggingface", {})
    hub_model_id = hf_cfg.get("hub_model_id")
    ckpt_name = hub_model_id.split("/")[-1] if hub_model_id else "qwen3_4b_lean"
    checkpoint_dir = os.path.join(cfg["paths"]["checkpoint_dir"], ckpt_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

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
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        logger.info(
            f"GPU memory after model load: {allocated:.2f} GB allocated, "
            f"{reserved:.2f} GB reserved"
        )

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # config_exp2.yaml uses paths.train_data; fall back to lean_workbook_train.jsonl
    train_path = cfg["paths"].get("train_data") or os.path.join(
        cfg["paths"]["data_dir"], "lean_workbook_train.jsonl"
    )
    if not os.path.isabs(train_path):
        train_path = str(SCRIPT_DIR / train_path)
    logger.info(f"Loading training data: {train_path}")
    raw_data = load_jsonl(train_path)
    logger.info(f"Training examples: {len(raw_data)}")

    max_length = cfg["training"]["max_length"]
    train_dataset = ProofDataset(raw_data, tokenizer, max_length)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    tcfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        warmup_ratio=tcfg["warmup_ratio"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg.get("gradient_checkpointing", False),
        logging_steps=tcfg["logging_steps"],
        save_strategy=tcfg["save_strategy"],
        save_total_limit=tcfg["save_total_limit"],
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info(f"Saving adapter to {checkpoint_dir}")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    if hf_cfg.get("push_to_hub") and hub_model_id:
        logger.info(f"Pushing adapter to HuggingFace: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        logger.info(f"Pushed to {hub_model_id}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
