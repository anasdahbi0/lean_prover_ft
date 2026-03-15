#!/usr/bin/env python3
"""
Experiment B: Self-correction eval loop.

For each miniF2F problem:
  Round 0:   generate one proof (synthesis mode, same format as Exp 1)
  Round 1-N: if Round k-1 failed, feed compiler error back and generate a repair

Compare against independent sampling baseline:
  N+1 independent synthesis proofs per problem (no feedback signal).

Outputs (in --output_dir):
  self_correction_results.json  — per-problem self-correction results
  accuracy_by_round.json        — {round: accuracy%} for plotting
  sampling_results.json         — per-problem sampling baseline (if run)
  accuracy_by_budget.json       — {budget: accuracy%} for sampling
  correction_vs_sampling.png    — the main figure

Run:
    # Experiment B only (self-correction):
    python eval_self_correct.py \\
        --adapter checkpoints/exp2_mixed \\
        --max_rounds 8 \\
        --output_dir results/exp_b

    # Self-correction + sampling baseline (recommended):
    python eval_self_correct.py \\
        --adapter checkpoints/exp2_mixed \\
        --max_rounds 8 \\
        --run_sampling_baseline \\
        --output_dir results/exp_b

    # Quick smoke test (5 problems, 2 rounds):
    python eval_self_correct.py \\
        --adapter checkpoints/exp2_mixed \\
        --max_rounds 2 \\
        --problems_limit 5 \\
        --output_dir results/exp_b_test
"""

import argparse
import json
import logging
import os
import re
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import comb
from pathlib import Path

import requests
import torch
import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()

LEAN4_DEFAULT_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "\n"
    "set_option maxHeartbeats 0\n"
    "\n"
    "open BigOperators Real Nat Topology Rat\n"
    "\n"
)

REPAIR_SYSTEM_PROMPT = (
    "A Lean 4 proof attempt failed to compile. "
    "Use the compiler error to write a corrected proof."
)


# ---------------------------------------------------------------------------
# Config + IO
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
# Lean verification
# ---------------------------------------------------------------------------

def verify_lean4_http(code: str, server_url: str, timeout: int = 60) -> dict:
    """
    Verify a complete Lean 4 file via the HTTP REPL server.

    Returns:
        complete:       bool — True iff no errors, no sorries, no failed warnings
        error_message:  str  — joined compiler error text (empty if success)
        raw_errors:     list — raw error message dicts from REPL
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

        all_msgs  = result_raw.get("messages", [])
        errors    = [m for m in all_msgs if m.get("severity") == "error"]
        sorries   = result_raw.get("sorries", [])
        bad_warns = [
            m for m in all_msgs
            if m.get("severity") == "warning"
            and (
                "declaration uses 'sorry'" in m.get("data", "")
                or "failed" in m.get("data", "")
            )
        ]

        complete      = not errors and not sorries and not bad_warns
        error_message = "\n".join(e.get("data", "") for e in errors)
        return {
            "complete":      complete,
            "error_message": error_message,
            "raw_errors":    errors,
        }
    except Exception:
        return {
            "complete":      False,
            "error_message": traceback.format_exc(),
            "raw_errors":    [],
        }


def assemble_lean_file(formal_statement: str, proof_body: str) -> str:
    """Assemble a verifiable Lean 4 file from statement + proof body."""
    return LEAN4_DEFAULT_HEADER + formal_statement + "\n" + proof_body


# ---------------------------------------------------------------------------
# Proof extraction (handles Qwen3 thinking tokens + fences)
# ---------------------------------------------------------------------------

def extract_proof(text: str) -> str:
    """
    Extract tactic proof body from model output.
    Strips <think>...</think> blocks, ```lean4 fences, and whitespace.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if text.startswith("```lean4"):
        text = text[len("```lean4") :].strip()
    elif text.startswith("```lean"):
        text = text[len("```lean") :].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate one response for the given messages list."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def synthesis_messages(formal_statement: str) -> list[dict]:
    """Synthesis prompt — no system message, matches Exp 1/2 training."""
    return [
        {
            "role": "user",
            "content": f"Complete the following Lean 4 proof:\n\n{formal_statement}",
        }
    ]


def repair_messages(
    formal_statement: str,
    failed_proof_body: str,
    error_message: str,
) -> list[dict]:
    """
    Repair prompt — matches the wo-exp APRIL training format.
    failed_proof_body: tactic body that failed (after ':= by').
    """
    user_content = (
        f"**Theorem:**\n```lean\n{formal_statement}\n```\n\n"
        f"**Failed proof:**\n```lean\n{failed_proof_body}\n```\n\n"
        f"**Compiler error:**\n{error_message}"
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# pass@k (unbiased estimator — HumanEval paper formula)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------------------
# Self-correction eval
# ---------------------------------------------------------------------------

def eval_self_correction(
    model,
    tokenizer,
    problems: list[dict],
    max_rounds: int,
    max_new_tokens: int,
    temperature: float,
    server_url: str,
    verify_timeout: int,
    logger: logging.Logger,
) -> list[dict]:
    """
    For each problem:
      Round 0   : synthesis mode → verify
      Rounds 1-N: repair mode (use previous error) → verify
    Returns per-problem result dicts.
    """
    results = []

    for prob_idx, prob in enumerate(problems):
        prob_id = prob.get("id", prob.get("name", str(prob_idx)))
        formal  = prob.get("formal", prob.get("statement", "")).strip()

        if not formal:
            logger.warning(f"[{prob_idx}] {prob_id}: empty formal statement, skipping")
            results.append({
                "id": prob_id, "solved": False, "round_solved": None,
                "rounds": [], "n_rounds": 0,
            })
            continue

        logger.info(f"[{prob_idx+1}/{len(problems)}] {prob_id}")

        solved       = False
        round_solved = None
        round_log    = []

        # ---- Round 0: synthesis ----
        raw        = generate_one(model, tokenizer, synthesis_messages(formal),
                                  max_new_tokens, temperature)
        proof_body = extract_proof(raw)
        lean_code  = assemble_lean_file(formal, proof_body)
        vr         = verify_lean4_http(lean_code, server_url, verify_timeout)

        round_log.append({
            "round": 0, "mode": "synthesis",
            "proof_body": proof_body,
            "complete":   vr["complete"],
            "error":      vr["error_message"],
        })
        logger.info(f"  Round 0 (synthesis): {'PASS' if vr['complete'] else 'FAIL'}")

        if vr["complete"]:
            solved       = True
            round_solved = 0
        else:
            cur_body  = proof_body
            cur_error = vr["error_message"]

            # ---- Rounds 1..max_rounds: repair ----
            for rnum in range(1, max_rounds + 1):
                if not cur_error.strip():
                    # No error message — can't form a meaningful repair prompt
                    logger.info(f"  Round {rnum}: no error message, stopping")
                    break

                raw        = generate_one(
                    model, tokenizer,
                    repair_messages(formal, cur_body, cur_error),
                    max_new_tokens, temperature,
                )
                proof_body = extract_proof(raw)
                lean_code  = assemble_lean_file(formal, proof_body)
                vr         = verify_lean4_http(lean_code, server_url, verify_timeout)

                round_log.append({
                    "round": rnum, "mode": "repair",
                    "proof_body": proof_body,
                    "complete":   vr["complete"],
                    "error":      vr["error_message"],
                })
                logger.info(f"  Round {rnum} (repair):    {'PASS' if vr['complete'] else 'FAIL'}")

                if vr["complete"]:
                    solved       = True
                    round_solved = rnum
                    break

                cur_body  = proof_body
                cur_error = vr["error_message"]

        results.append({
            "id":           prob_id,
            "solved":       solved,
            "round_solved": round_solved,
            "rounds":       round_log,
            "n_rounds":     len(round_log),
        })

    return results


# ---------------------------------------------------------------------------
# Independent sampling baseline
# ---------------------------------------------------------------------------

def eval_independent_sampling(
    model,
    tokenizer,
    problems: list[dict],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    server_url: str,
    verify_timeout: int,
    logger: logging.Logger,
) -> list[dict]:
    """
    For each problem, generate n_samples independent synthesis proofs.
    Returns per-problem result dicts with completions_per_sample list.
    """
    results = []

    for prob_idx, prob in enumerate(problems):
        prob_id = prob.get("id", prob.get("name", str(prob_idx)))
        formal  = prob.get("formal", prob.get("statement", "")).strip()

        if not formal:
            results.append({
                "id": prob_id, "n_correct": 0, "n_samples": n_samples,
                "completions_per_sample": [False] * n_samples,
            })
            continue

        logger.info(
            f"[{prob_idx+1}/{len(problems)}] {prob_id}: generating {n_samples} samples"
        )

        # Generate n_samples proofs
        candidates = []
        for _ in range(n_samples):
            raw  = generate_one(model, tokenizer, synthesis_messages(formal),
                                max_new_tokens, temperature)
            body = extract_proof(raw)
            if "sorry" in body:
                candidates.append(None)
            else:
                candidates.append(assemble_lean_file(formal, body))

        # Verify concurrently
        completions = [False] * n_samples

        def _verify(idx_code):
            idx, code = idx_code
            if code is None:
                return idx, False
            r = verify_lean4_http(code, server_url, verify_timeout)
            return idx, r["complete"]

        with ThreadPoolExecutor(max_workers=min(32, n_samples)) as pool:
            futures = {pool.submit(_verify, (i, c)): i for i, c in enumerate(candidates)}
            for fut in as_completed(futures):
                idx, ok = fut.result()
                completions[idx] = ok

        n_correct = sum(completions)
        logger.info(f"  {n_correct}/{n_samples} correct")

        results.append({
            "id":                     prob_id,
            "n_correct":              n_correct,
            "n_samples":              n_samples,
            "completions_per_sample": completions,
        })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def correction_accuracy_by_round(results: list[dict], max_rounds: int) -> dict:
    """
    For round budget B (total proofs = B+1), fraction solved using self-correction.
    Returns {B: accuracy_pct} for B in 0..max_rounds.
    """
    n   = len(results)
    acc = {}
    for budget in range(0, max_rounds + 1):
        n_solved = sum(
            1 for r in results
            if r["solved"]
            and r["round_solved"] is not None
            and r["round_solved"] <= budget
        )
        acc[budget] = round(100.0 * n_solved / max(n, 1), 2)
    return acc


def sampling_accuracy_by_budget(results: list[dict]) -> dict:
    """
    For proof budget k (1..n_samples), fraction solved by independent sampling.
    Returns {k: accuracy_pct}.
    """
    if not results:
        return {}
    n_problems = len(results)
    n_samples  = results[0]["n_samples"]
    acc = {}
    for k in range(1, n_samples + 1):
        n_solved = sum(
            1 for r in results
            if any(r["completions_per_sample"][:k])
        )
        acc[k] = round(100.0 * n_solved / max(n_problems, 1), 2)
    return acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_correction_vs_sampling(
    correction_acc: dict,
    sampling_acc: dict,
    output_path: str,
    n_problems: int,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. pip install matplotlib")
        return

    corr_x = sorted(correction_acc.keys())
    corr_y = [correction_acc[x] for x in corr_x]

    # Align sampling x-axis: budget k corresponds to k-1 additional proofs
    samp_x = sorted(sampling_acc.keys())
    samp_y = [sampling_acc[x] for x in samp_x]
    samp_x_aligned = [x - 1 for x in samp_x]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(corr_x, corr_y, "o-",
            label="Self-correction (compiler feedback)",
            color="#2196F3", linewidth=2.5, markersize=7)

    ax.plot(samp_x_aligned, samp_y, "s--",
            label="Independent sampling (no feedback)",
            color="#FF9800", linewidth=2.5, markersize=7)

    ax.set_xlabel("Additional attempts after first generation (N)", fontsize=12)
    ax.set_ylabel("% miniF2F problems solved", fontsize=12)
    ax.set_title(
        "Self-Correction vs Independent Sampling on miniF2F\n"
        f"(Qwen3-4B + APRIL mixed, n={n_problems} problems)",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(corr_x)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {output_path}")


# ---------------------------------------------------------------------------
# Additional analysis helpers
# ---------------------------------------------------------------------------

def loop_analysis(results: list[dict]) -> int:
    """Count problems where the model repeats the exact same proof body >= 3 times."""
    n_loops = 0
    for r in results:
        bodies = [rd["proof_body"] for rd in r.get("rounds", [])]
        if bodies:
            most_common_count = Counter(bodies).most_common(1)[0][1]
            if most_common_count >= 3:
                n_loops += 1
    return n_loops


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config_exp2_mixed.yaml")
    parser.add_argument("--dataset",  default="data/minif2f_test.jsonl",
                        help="Problems JSONL file (minif2f_test.jsonl)")
    parser.add_argument("--adapter",  default=None,
                        help="Path to LoRA adapter dir (omit for base model)")
    parser.add_argument("--max_rounds", type=int, default=8,
                        help="Max repair rounds per problem (0 = synthesis only)")
    parser.add_argument("--run_sampling_baseline", action="store_true",
                        help="Also run independent sampling baseline at same budget")
    parser.add_argument("--sampling_budget", type=int, default=None,
                        help="Total proofs for sampling baseline (default: max_rounds+1)")
    parser.add_argument("--output_dir", default="results/exp_b")
    parser.add_argument("--problems_limit", type=int, default=None,
                        help="Only eval first N problems (for testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "eval_self_correct.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    cfg            = load_config(args.config)
    model_name     = cfg["model"]["name"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    temperature    = cfg["generation"].get("temperature", 0.6)
    server_url     = cfg["lean_server"]["url"]
    verify_timeout = cfg["lean_server"]["timeout"]
    sampling_budget = args.sampling_budget or (args.max_rounds + 1)

    # ------------------------------------------------------------------
    # Tokenizer + model
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        alloc = torch.cuda.memory_allocated() / 1e9
        resv  = torch.cuda.memory_reserved()  / 1e9
        logger.info(f"GPU memory: {alloc:.1f}GB alloc, {resv:.1f}GB reserved")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_path = (
        SCRIPT_DIR / args.dataset
        if not os.path.isabs(args.dataset)
        else Path(args.dataset)
    )
    logger.info(f"Loading problems: {dataset_path}")
    problems = load_jsonl(str(dataset_path))
    if args.problems_limit:
        problems = problems[: args.problems_limit]
        logger.info(f"Limited to first {len(problems)} problems")
    logger.info(f"Total problems: {len(problems)}")

    # ------------------------------------------------------------------
    # Experiment B: self-correction
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT B: Self-correction (max_rounds={args.max_rounds})")
    logger.info(f"{'='*60}")

    sc_results = eval_self_correction(
        model, tokenizer, problems,
        max_rounds     = args.max_rounds,
        max_new_tokens = max_new_tokens,
        temperature    = temperature,
        server_url     = server_url,
        verify_timeout = verify_timeout,
        logger         = logger,
    )

    correction_acc = correction_accuracy_by_round(sc_results, args.max_rounds)
    n_solved_total = sum(1 for r in sc_results if r["solved"])

    logger.info(f"\n--- Self-correction accuracy by round ---")
    for budget, acc in sorted(correction_acc.items()):
        label = "synthesis only" if budget == 0 else f"synthesis + {budget} repair(s)"
        n_at  = int(round(acc * len(problems) / 100))
        logger.info(f"  N={budget:2d} ({label:35s}): {acc:5.1f}%  ({n_at}/{len(problems)})")

    # Save
    sc_output = {
        "experiment":       "self_correction",
        "adapter":          args.adapter,
        "max_rounds":       args.max_rounds,
        "n_problems":       len(problems),
        "n_solved_total":   n_solved_total,
        "accuracy_by_round": correction_acc,
        "per_problem":      sc_results,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
    }
    sc_path = os.path.join(args.output_dir, "self_correction_results.json")
    with open(sc_path, "w") as f:
        json.dump(sc_output, f, indent=2)
    logger.info(f"Saved → {sc_path}")

    acc_path = os.path.join(args.output_dir, "accuracy_by_round.json")
    with open(acc_path, "w") as f:
        json.dump(correction_acc, f, indent=2)

    # ------------------------------------------------------------------
    # Sampling baseline
    # ------------------------------------------------------------------
    sampling_acc = {}
    if args.run_sampling_baseline:
        logger.info(f"\n{'='*60}")
        logger.info(f"BASELINE: Independent sampling (n={sampling_budget} proofs/problem)")
        logger.info(f"{'='*60}")

        samp_results = eval_independent_sampling(
            model, tokenizer, problems,
            n_samples      = sampling_budget,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            server_url     = server_url,
            verify_timeout = verify_timeout,
            logger         = logger,
        )

        sampling_acc  = sampling_accuracy_by_budget(samp_results)
        n_samp_solved = sum(1 for r in samp_results if r["n_correct"] > 0)

        logger.info(f"\n--- Independent sampling accuracy by budget ---")
        for bgt, acc in sorted(sampling_acc.items()):
            n_at = int(round(acc * len(problems) / 100))
            logger.info(f"  budget={bgt:2d}: {acc:5.1f}%  ({n_at}/{len(problems)})")

        samp_output = {
            "experiment":       "independent_sampling",
            "adapter":          args.adapter,
            "n_samples":        sampling_budget,
            "n_problems":       len(problems),
            "n_solved":         n_samp_solved,
            "accuracy_by_budget": sampling_acc,
            "per_problem":      samp_results,
            "timestamp":        datetime.utcnow().isoformat() + "Z",
        }
        samp_path = os.path.join(args.output_dir, "sampling_results.json")
        with open(samp_path, "w") as f:
            json.dump(samp_output, f, indent=2)
        logger.info(f"Saved → {samp_path}")

        samp_acc_path = os.path.join(args.output_dir, "accuracy_by_budget.json")
        with open(samp_acc_path, "w") as f:
            json.dump(sampling_acc, f, indent=2)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if sampling_acc:
        plot_path = os.path.join(args.output_dir, "correction_vs_sampling.png")
        plot_correction_vs_sampling(
            correction_acc, sampling_acc, plot_path, n_problems=len(problems)
        )

    # ------------------------------------------------------------------
    # Summary + analysis
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(
        f"Synthesis only (round 0)     : {correction_acc.get(0, 0):.1f}%  "
        f"({int(round(correction_acc.get(0,0)*len(problems)/100))}/{len(problems)})"
    )
    if args.max_rounds > 0:
        logger.info(
            f"Self-correction (max rounds) : {correction_acc.get(args.max_rounds, 0):.1f}%  "
            f"({int(round(correction_acc.get(args.max_rounds,0)*len(problems)/100))}/{len(problems)})"
        )
    if sampling_acc:
        bgt = sampling_budget
        logger.info(
            f"Independent sampling (n={bgt})  : {sampling_acc.get(bgt, 0):.1f}%  "
            f"({int(round(sampling_acc.get(bgt,0)*len(problems)/100))}/{len(problems)})"
        )

    # Problems solved by repair but not by synthesis alone
    repair_only = [
        r for r in sc_results
        if r["solved"] and r["round_solved"] is not None and r["round_solved"] > 0
    ]
    logger.info(f"\nProblems solved by correction but not synthesis alone: {len(repair_only)}")
    for r in repair_only[:15]:
        logger.info(f"  {r['id']}  (solved at round {r['round_solved']})")

    # Round distribution
    round_hist = Counter(
        r["round_solved"]
        for r in sc_results
        if r["solved"] and r["round_solved"] is not None
    )
    logger.info(f"\nSolved at each round: {dict(sorted(round_hist.items()))}")

    # Loop detection
    n_loops = loop_analysis(sc_results)
    logger.info(f"Problems stuck in loops (same proof >= 3 times): {n_loops}")

    # Overlap analysis between correction and sampling
    if sampling_acc and args.run_sampling_baseline:
        sc_ids_solved   = {r["id"] for r in sc_results if r["solved"]}
        samp_results_out = json.load(open(samp_path))["per_problem"]
        samp_ids_solved  = {r["id"] for r in samp_results_out if r["n_correct"] > 0}
        both        = sc_ids_solved & samp_ids_solved
        sc_only     = sc_ids_solved - samp_ids_solved
        samp_only   = samp_ids_solved - sc_ids_solved
        logger.info(f"\nSolved by both: {len(both)}")
        logger.info(f"Solved by correction only: {len(sc_only)}")
        logger.info(f"Solved by sampling only:   {len(samp_only)}")

    logger.info(f"\nAll results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
