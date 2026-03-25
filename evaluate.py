#!/usr/bin/env python3
"""
Evaluate a base or LoRA-adapted Qwen3-4B-Instruct on Lean 4 theorem proving.

Generates n_samples proof candidates per theorem, verifies each with the
Lean server, and reports pass@k for k in {1, 2, 4, 8, 32}.

Run:
    # baseline (no adapter)
    python evaluate.py \\
        --dataset data/minif2f_test.jsonl \\
        --output results/eval_baseline.json

    # fine-tuned
    python evaluate.py \\
        --adapter checkpoints/qwen3_4b_lean \\
        --dataset data/minif2f_test.jsonl \\
        --output results/eval_finetuned.json
"""

import argparse
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import comb
from pathlib import Path

import requests
import torch
import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()

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
    resolved = SCRIPT_DIR / path if not os.path.isabs(path) else Path(path)
    with open(resolved) as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(SCRIPT_DIR / val)
    return cfg


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_user_content(formal_statement):
    """
    Prompt format matching Goedel-Prover's non_cot_prompt, adapted for chat template.
    """
    return (
        "Complete the following Lean 4 code:\n\n"
        "```lean4\n"
        + LEAN4_DEFAULT_HEADER
        + formal_statement
        + "\n```"
    )


# ---------------------------------------------------------------------------
# Lean verification
# Adapted from Goedel-Prover/prover/lean/verifier.py.
# Uses the same JSON request format; sends to HTTP server instead of subprocess.
# ---------------------------------------------------------------------------

def verify_lean4_http(code, server_url, timeout=60):
    """
    Verify a complete Lean 4 file via the HTTP REPL server.

    Request format is identical to Goedel-Prover's verify_lean4_file:
        {"cmd": code, "allTactics": false, "ast": false, ...}

    Response is parsed identically:
        pass     = no error messages
        complete = pass AND no sorries AND no 'declaration uses sorry' warnings
    """
    command = {
        "cmd": code,
        "allTactics": False,
        "ast": False,
        "tactics": False,
        "premises": False,
    }
    try:
        resp = requests.post(server_url, json=command, timeout=timeout)
        resp.raise_for_status()
        result_raw = resp.json()

        result = {
            "sorries":  result_raw.get("sorries", []),
            "tactics":  result_raw.get("tactics", []),
            "errors":   [m for m in result_raw.get("messages", []) if m["severity"] == "error"],
            "warnings": [m for m in result_raw.get("messages", []) if m["severity"] == "warning"],
            "infos":    [m for m in result_raw.get("messages", []) if m["severity"] == "info"],
            "system_messages": "",
            "system_errors":   None,
            "verified_code":   code,
        }
        result["pass"] = not result["errors"]
        result["complete"] = (
            result["pass"]
            and not result["sorries"]
            and not any(
                "declaration uses 'sorry'" in w["data"] or "failed" in w["data"]
                for w in result["warnings"]
            )
        )
    except Exception:
        result = {
            "pass": False,
            "complete": False,
            "errors": [],
            "sorries": [],
            "warnings": [],
            "system_messages": "",
            "system_errors": traceback.format_exc(),
        }
    return result


def assemble_lean_file(formal_statement, proof_body):
    """
    Assemble a complete Lean 4 file ready for verification.
    Normalises the statement to end with ':= by' so that bare tactic blocks
    (as produced by the model) are always syntactically valid.
    minif2f statements end with ':=' while lean_workbook ends with ':= by'.
    """
    stmt = formal_statement.rstrip()
    if stmt.endswith(":=") and not stmt.endswith(":= by"):
        stmt = stmt + " by"
    return LEAN4_DEFAULT_HEADER + stmt + "\n" + proof_body


# ---------------------------------------------------------------------------
# Proof extraction
# ---------------------------------------------------------------------------

def extract_proof(generated_text):
    """
    Clean up raw model output to extract the tactic proof body.

    Handles:
      - Qwen3 <think>...</think> blocks (chain-of-thought)
      - Markdown ```lean4 ... ``` fences
      - Leading/trailing whitespace
    """
    # Strip thinking blocks produced by Qwen3-4B-Instruct
    text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()

    # Strip ```lean4 or ``` fences if the model wrapped its output
    if text.startswith("```lean4"):
        text = text[len("```lean4"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ---------------------------------------------------------------------------
# pass@k  (unbiased estimator, same formula used in HumanEval paper)
# ---------------------------------------------------------------------------

def pass_at_k(n, c, k):
    """
    n : total samples generated per problem
    c : number of correct samples for this problem
    k : k value
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_proofs(model, tokenizer, prompt, n_samples, max_new_tokens, temperature):
    """
    Generate n_samples proof candidates for a single prompt.
    Returns a list of raw decoded strings (before proof extraction).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

    # outputs shape: [n_samples, seq_len] — strip the prompt prefix
    proofs = []
    for i in range(n_samples):
        generated_ids = outputs[i][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        proofs.append(text)
    return proofs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--dataset",  required=True,
                        help="Path to JSONL eval file (minif2f_test.jsonl etc.)")
    parser.add_argument("--adapter",  default=None,
                        help="Path to LoRA adapter directory (omit for baseline)")
    parser.add_argument("--output",   required=True,
                        help="Path to write results JSON")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Only evaluate the first N problems (smoke test)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override config n_samples (e.g. 1 for fast smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    log_path = os.path.join(cfg["paths"]["results_dir"], "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    model_name  = cfg["model"]["name"]
    n_samples   = args.n_samples if args.n_samples is not None else cfg["generation"]["n_samples"]
    max_new_tok = cfg["generation"]["max_new_tokens"]
    temperature = cfg["generation"]["temperature"]
    server_url  = cfg["lean_server"]["url"]
    verify_timeout = cfg["lean_server"]["timeout"]
    max_workers = cfg["lean_server"]["max_workers"]

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {model_name}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Model  (base or base + LoRA adapter)
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=cfg["model"]["trust_remote_code"],
        device_map="auto",
    )

    if args.adapter:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        logger.info(
            f"GPU memory after model load: {allocated:.2f} GB allocated, "
            f"{reserved:.2f} GB reserved"
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.dataset}")
    problems = load_jsonl(args.dataset)
    if args.limit is not None:
        problems = problems[:args.limit]
        logger.info(f"Problems: {len(problems)} (limited to {args.limit})")
    else:
        logger.info(f"Problems: {len(problems)}")

    # Normalise field names across datasets:
    # lean_workbook_train.jsonl uses 'statement'
    # cat-searcher/minif2f-lean4 uses 'formal_statement'
    def get_formal(prob):
        return prob.get("formal_statement", prob.get("formal", prob.get("statement", "")))

    def get_id(prob):
        return prob.get("id", prob.get("name", ""))

    # ------------------------------------------------------------------
    # Generate + verify
    # ------------------------------------------------------------------
    per_problem = []

    for prob_idx, prob in enumerate(problems):
        prob_id      = get_id(prob)
        formal_stmt  = get_formal(prob)

        if not formal_stmt:
            logger.warning(f"[{prob_idx}] {prob_id}: empty formal statement, skipping")
            per_problem.append({"id": prob_id, "n_correct": 0, "solved": False})
            continue

        # Build the inference prompt (user turn only, with generation prompt)
        user_content = build_user_content(formal_stmt)
        messages     = [{"role": "user", "content": user_content}]
        prompt       = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate n_samples raw outputs
        logger.info(f"[{prob_idx+1}/{len(problems)}] {prob_id}: generating {n_samples} samples ...")
        raw_outputs = generate_proofs(
            model, tokenizer, prompt, n_samples, max_new_tok, temperature
        )

        # Extract and filter proof bodies
        candidates = []
        for raw in raw_outputs:
            proof_body = extract_proof(raw)
            # Reject proofs that contain 'sorry'
            if "sorry" in proof_body:
                continue
            full_lean = assemble_lean_file(formal_stmt, proof_body)
            candidates.append(full_lean)

        logger.info(f"  {len(candidates)}/{n_samples} candidates after sorry filter")

        # Verify candidates concurrently
        n_correct = 0
        if candidates:
            def _verify(code):
                return verify_lean4_http(code, server_url, verify_timeout)

            with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as pool:
                futures = {pool.submit(_verify, c): c for c in candidates}
                for fut in as_completed(futures):
                    result = fut.result()
                    if result.get("complete", False):
                        n_correct += 1

        logger.info(f"  {n_correct} correct")
        per_problem.append({
            "id":        prob_id,
            "n_correct": n_correct,
            "solved":    n_correct > 0,
        })

    # ------------------------------------------------------------------
    # Aggregate pass@k
    # ------------------------------------------------------------------
    n_total  = len(per_problem)
    n_solved = sum(1 for r in per_problem if r["solved"])

    ks = [1, 2, 4, 8, 32]
    pass_at_k_values = {}
    for k in ks:
        if k > n_samples:
            continue
        avg = sum(
            pass_at_k(n_samples, r["n_correct"], k)
            for r in per_problem
        ) / max(n_total, 1)
        pass_at_k_values[f"pass@{k}"] = round(avg, 4)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results = {
        "model":       model_name,
        "adapter":     args.adapter,
        "dataset":     args.dataset,
        "n_samples":   n_samples,
        "n_total":     n_total,
        "n_solved":    n_solved,
        "pass_at_k":   pass_at_k_values,
        "per_problem": per_problem,
        "timestamp":   datetime.utcnow().isoformat() + "Z",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results written to {args.output}")
    logger.info(f"Solved: {n_solved}/{n_total}")
    for k_label, v in pass_at_k_values.items():
        logger.info(f"  {k_label}: {v:.4f}")


if __name__ == "__main__":
    main()
