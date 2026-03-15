#!/usr/bin/env python3
"""
Experiment B: Two-model self-correction eval.

  Model 1 — Generator : the Exp 1 model (9% pass@32 on miniF2F, synthesis specialist)
  Model 2 — Corrector : Qwen trained on APRIL (repair specialist)

For each of 244 miniF2F problems:
  Round 0   : Generator produces one proof → compile
  Round 1-N : if failed, feed (statement + failed proof + compiler error) to
              Corrector → compile.  Corrector's own failure becomes input
              for the next round.
  Sweep N   : {1, 2, 3, 5, 8, 12}

Baseline (independent sampling):
  Generator produces 13 independent synthesis proofs (no feedback).
  Accuracy at budget k ∈ {1,2,...,13} reported for the same x-axis.

Outputs:
  {output_dir}/per_problem.json          per-problem round log
  {output_dir}/accuracy_by_round.json    {N: acc%} for self-correction
  {output_dir}/accuracy_by_budget.json   {k: acc%} for independent sampling
  {output_dir}/correction_vs_sampling.png  the main figure

PEFT multi-adapter trick:
  Both adapters are mounted on ONE copy of the base model.
  At inference time we call model.set_adapter("generator") or
  model.set_adapter("corrector") — no reloading, no extra VRAM.

Run:
    python eval_two_model.py \\
        --generator_adapter yotsubian/qwen \\
        --corrector_adapter checkpoints/corrector \\
        --output_dir results/exp_b

    # Smoke test (5 problems, 3 rounds):
    python eval_two_model.py \\
        --generator_adapter yotsubian/qwen \\
        --corrector_adapter checkpoints/corrector \\
        --max_rounds 3 --problems_limit 5 \\
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

# N values to report in the table/plot
ROUND_CHECKPOINTS = [0, 1, 2, 3, 5, 8, 12]


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
    Returns:
        complete:      bool
        error_message: str (compiler errors joined, empty if success)
    """
    command = {
        "cmd": code, "allTactics": False,
        "ast": False, "tactics": False, "premises": False,
    }
    try:
        resp = requests.post(server_url, json=command, timeout=timeout)
        resp.raise_for_status()
        raw      = resp.json()
        msgs     = raw.get("messages", [])
        errors   = [m for m in msgs if m.get("severity") == "error"]
        sorries  = raw.get("sorries", [])
        bad_warn = [m for m in msgs
                    if m.get("severity") == "warning"
                    and ("declaration uses 'sorry'" in m.get("data", "")
                         or "failed" in m.get("data", ""))]
        complete = not errors and not sorries and not bad_warn
        return {
            "complete":      complete,
            "error_message": "\n".join(e.get("data", "") for e in errors),
        }
    except Exception:
        return {"complete": False, "error_message": traceback.format_exc()}


def assemble_lean_file(statement: str, proof_body: str) -> str:
    return LEAN4_DEFAULT_HEADER + statement + "\n" + proof_body


# ---------------------------------------------------------------------------
# Proof extraction
# ---------------------------------------------------------------------------

def extract_proof(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for fence in ("```lean4", "```lean", "```"):
        if text.startswith(fence):
            text = text[len(fence):].strip()
            break
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(model, tokenizer, messages: list[dict],
                 max_new_tokens: int, temperature: float) -> str:
    prompt    = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def synthesis_messages(formal: str) -> list[dict]:
    """Prompt format must match Exp1 training (train.py / evaluate.py)."""
    return [{"role": "user",
             "content": (
                 "Complete the following Lean 4 code:\n\n"
                 "```lean4\n"
                 + LEAN4_DEFAULT_HEADER
                 + formal
                 + "\n```"
             )}]


def repair_messages(formal: str, failed_body: str, error: str) -> list[dict]:
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user",
         "content": (
             f"**Theorem:**\n```lean\n{formal}\n```\n\n"
             f"**Failed proof:**\n```lean\n{failed_body}\n```\n\n"
             f"**Compiler error:**\n{error}"
         )},
    ]


# ---------------------------------------------------------------------------
# Core: two-model self-correction
# ---------------------------------------------------------------------------

def eval_two_model(
    model, tokenizer,
    problems: list[dict],
    max_rounds: int,
    max_new_tokens: int,
    temperature: float,
    server_url: str,
    verify_timeout: int,
    logger: logging.Logger,
) -> list[dict]:
    """
    Round 0:   model.set_adapter("generator") → synthesis
    Rounds 1+: model.set_adapter("corrector") → repair
    Returns per-problem dicts with full round log.
    """
    results = []

    for idx, prob in enumerate(problems):
        prob_id = prob.get("id", prob.get("name", str(idx)))
        formal  = prob.get("formal", prob.get("statement", "")).strip()

        if not formal:
            logger.warning(f"[{idx}] {prob_id}: empty formal, skipping")
            results.append({
                "id": prob_id, "solved": False,
                "round_solved": None, "rounds": [],
            })
            continue

        logger.info(f"[{idx+1}/{len(problems)}] {prob_id}")

        solved       = False
        round_solved = None
        round_log    = []

        # ---- Round 0: Generator (synthesis) ----
        model.set_adapter("generator")
        raw        = generate_one(model, tokenizer, synthesis_messages(formal),
                                  max_new_tokens, temperature)
        proof_body = extract_proof(raw)
        vr         = verify_lean4_http(
            assemble_lean_file(formal, proof_body), server_url, verify_timeout
        )
        round_log.append({
            "round": 0, "model": "generator",
            "proof_body": proof_body,
            "complete":   vr["complete"],
            "error":      vr["error_message"],
        })
        logger.info(f"  Round 0 [generator]: {'PASS' if vr['complete'] else 'FAIL'}")

        if vr["complete"]:
            solved, round_solved = True, 0
        else:
            cur_body  = proof_body
            cur_error = vr["error_message"]

            # ---- Rounds 1..max_rounds: Corrector (repair) ----
            model.set_adapter("corrector")

            for rnum in range(1, max_rounds + 1):
                if not cur_error.strip():
                    logger.info(f"  Round {rnum}: no error message, stopping")
                    break

                raw        = generate_one(
                    model, tokenizer,
                    repair_messages(formal, cur_body, cur_error),
                    max_new_tokens, temperature,
                )
                proof_body = extract_proof(raw)
                vr         = verify_lean4_http(
                    assemble_lean_file(formal, proof_body), server_url, verify_timeout
                )
                round_log.append({
                    "round": rnum, "model": "corrector",
                    "proof_body": proof_body,
                    "complete":   vr["complete"],
                    "error":      vr["error_message"],
                })
                logger.info(
                    f"  Round {rnum} [corrector]: {'PASS' if vr['complete'] else 'FAIL'}"
                )
                if vr["complete"]:
                    solved, round_solved = True, rnum
                    break
                cur_body  = proof_body
                cur_error = vr["error_message"]

        results.append({
            "id": prob_id, "solved": solved,
            "round_solved": round_solved, "rounds": round_log,
        })

    return results


# ---------------------------------------------------------------------------
# Baseline: independent sampling (Generator only)
# ---------------------------------------------------------------------------

def eval_independent_sampling(
    model, tokenizer,
    problems: list[dict],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    server_url: str,
    verify_timeout: int,
    logger: logging.Logger,
) -> list[dict]:
    """Generator produces n_samples independent synthesis proofs per problem."""
    results = []

    for idx, prob in enumerate(problems):
        prob_id = prob.get("id", prob.get("name", str(idx)))
        formal  = prob.get("formal", prob.get("statement", "")).strip()

        if not formal:
            results.append({
                "id": prob_id, "n_correct": 0, "n_samples": n_samples,
                "completions": [False] * n_samples,
            })
            continue

        logger.info(
            f"[{idx+1}/{len(problems)}] {prob_id}: sampling {n_samples} proofs"
        )

        model.set_adapter("generator")
        candidates = []
        for _ in range(n_samples):
            raw  = generate_one(model, tokenizer, synthesis_messages(formal),
                                max_new_tokens, temperature)
            body = extract_proof(raw)
            candidates.append(
                assemble_lean_file(formal, body) if "sorry" not in body else None
            )

        completions = [False] * n_samples

        def _verify(item):
            i, code = item
            if code is None:
                return i, False
            return i, verify_lean4_http(code, server_url, verify_timeout)["complete"]

        with ThreadPoolExecutor(max_workers=min(32, n_samples)) as pool:
            futs = {pool.submit(_verify, (i, c)): i for i, c in enumerate(candidates)}
            for fut in as_completed(futs):
                i, ok = fut.result()
                completions[i] = ok

        n_correct = sum(completions)
        logger.info(f"  {n_correct}/{n_samples} correct")
        results.append({
            "id": prob_id, "n_correct": n_correct,
            "n_samples": n_samples, "completions": completions,
        })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def correction_accuracy_by_round(results: list[dict], max_rounds: int) -> dict:
    """Fraction solved by self-correction at each round budget 0..max_rounds."""
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
    """Fraction solved by independent sampling at each budget k=1..n_samples."""
    if not results:
        return {}
    n_prob    = len(results)
    n_samples = results[0]["n_samples"]
    acc = {}
    for k in range(1, n_samples + 1):
        n_solved = sum(1 for r in results if any(r["completions"][:k]))
        acc[k] = round(100.0 * n_solved / max(n_prob, 1), 2)
    return acc


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(
    correction_acc: dict,
    sampling_acc: dict,
    checkpoints: list[int],
    output_path: str,
    n_problems: int,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — pip install matplotlib")
        return

    # Self-correction: x = round number (cumulative attempts = x+1)
    corr_x = sorted(k for k in correction_acc if k in checkpoints or k == 0)
    corr_y = [correction_acc[x] for x in corr_x]

    # Independent sampling: budget k → k total proofs → k-1 additional proofs
    # Align to same x-axis: x = k-1
    samp_x = sorted(sampling_acc.keys())
    samp_y = [sampling_acc[k] for k in samp_x]
    samp_x_aligned = [k - 1 for k in samp_x]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(corr_x, corr_y, "o-",
            label="Two-model self-correction\n(Generator → Corrector)",
            color="#2196F3", linewidth=2.5, markersize=8)
    ax.plot(samp_x_aligned, samp_y, "s--",
            label="Independent sampling\n(Generator only, no feedback)",
            color="#FF9800", linewidth=2.5, markersize=8)

    ax.set_xlabel("Additional attempts after first generation (N)", fontsize=12)
    ax.set_ylabel("% miniF2F problems solved", fontsize=12)
    ax.set_title(
        "Two-Model Self-Correction vs Independent Sampling on miniF2F\n"
        f"Qwen3-4B  |  n={n_problems} problems",
        fontsize=11,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(corr_x)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config_corrector.yaml")
    parser.add_argument("--dataset",   default="data/minif2f_test.jsonl")
    parser.add_argument("--generator_adapter", required=True,
                        help="Generator LoRA path/HF repo (Exp 1 checkpoint)")
    parser.add_argument("--corrector_adapter", required=True,
                        help="Corrector LoRA path/HF repo (APRIL-trained)")
    parser.add_argument("--max_rounds", type=int, default=12,
                        help="Max correction rounds (default 12 to cover all checkpoints)")
    parser.add_argument("--no_sampling_baseline", action="store_true",
                        help="Skip independent sampling baseline")
    parser.add_argument("--output_dir",     default="results/exp_b")
    parser.add_argument("--problems_limit", type=int, default=None)
    args = parser.parse_args()

    max_rounds = args.max_rounds

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "eval_two_model.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    cfg            = load_config(args.config)
    model_name     = cfg["model"]["name"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    temperature    = cfg["generation"].get("temperature", 0.6)
    server_url     = cfg["lean_server"]["url"]
    verify_timeout = cfg["lean_server"]["timeout"]

    # ------------------------------------------------------------------
    # Load base model + mount BOTH adapters (one base, two adapters = ~8GB VRAM)
    # ------------------------------------------------------------------
    logger.info(f"Loading base model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=cfg["model"]["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=cfg["model"]["trust_remote_code"],
        device_map="auto",
    )

    # Mount generator adapter
    logger.info(f"Loading generator adapter: {args.generator_adapter}")
    model = PeftModel.from_pretrained(
        base_model, args.generator_adapter, adapter_name="generator"
    )

    # Mount corrector adapter onto the same base
    logger.info(f"Loading corrector adapter: {args.corrector_adapter}")
    model.load_adapter(args.corrector_adapter, adapter_name="corrector")

    model.eval()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resv  = torch.cuda.memory_reserved()  / 1e9
        logger.info(f"GPU memory: {alloc:.1f}GB alloc, {resv:.1f}GB reserved")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_path = (
        SCRIPT_DIR / args.dataset if not os.path.isabs(args.dataset)
        else Path(args.dataset)
    )
    problems = load_jsonl(str(dataset_path))
    if args.problems_limit:
        problems = problems[:args.problems_limit]
    logger.info(f"Problems: {len(problems)}")

    # ------------------------------------------------------------------
    # Experiment B: two-model self-correction
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT B: Two-model self-correction (max_rounds={max_rounds})")
    logger.info(f"  Generator : {args.generator_adapter}")
    logger.info(f"  Corrector : {args.corrector_adapter}")
    logger.info(f"  Checkpoints: {ROUND_CHECKPOINTS}")
    logger.info(f"{'='*60}")

    sc_results     = eval_two_model(
        model, tokenizer, problems,
        max_rounds=max_rounds, max_new_tokens=max_new_tokens,
        temperature=temperature, server_url=server_url,
        verify_timeout=verify_timeout, logger=logger,
    )
    correction_acc = correction_accuracy_by_round(sc_results, max_rounds)

    # Print results table for the requested checkpoints
    logger.info(f"\n--- Self-correction accuracy ---")
    logger.info(f"  {'N':>3}  {'Description':30}  {'Solved':>6}  {'Acc%':>6}")
    logger.info(f"  {'-'*55}")
    for n in sorted(correction_acc):
        if n not in ROUND_CHECKPOINTS and n != max_rounds:
            continue
        n_sol = int(round(correction_acc[n] * len(problems) / 100))
        desc  = "synthesis only" if n == 0 else f"+{n} correction round(s)"
        logger.info(f"  {n:>3}  {desc:30}  {n_sol:>4}/{len(problems)}  {correction_acc[n]:>5.1f}%")

    # Save
    sc_out_path = os.path.join(args.output_dir, "per_problem.json")
    with open(sc_out_path, "w") as f:
        json.dump({
            "experiment":        "two_model_self_correction",
            "generator_adapter": args.generator_adapter,
            "corrector_adapter": args.corrector_adapter,
            "max_rounds":        max_rounds,
            "n_problems":        len(problems),
            "n_solved_total":    sum(1 for r in sc_results if r["solved"]),
            "accuracy_by_round": correction_acc,
            "per_problem":       sc_results,
            "timestamp":         datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)
    logger.info(f"Saved → {sc_out_path}")

    acc_path = os.path.join(args.output_dir, "accuracy_by_round.json")
    with open(acc_path, "w") as f:
        json.dump(correction_acc, f, indent=2)

    # ------------------------------------------------------------------
    # Baseline: independent sampling
    # ------------------------------------------------------------------
    sampling_acc = {}
    if not args.no_sampling_baseline:
        n_samp = max_rounds + 1   # same total-proof budget as max correction rounds
        logger.info(f"\n{'='*60}")
        logger.info(f"BASELINE: Independent sampling (n={n_samp} proofs/problem)")
        logger.info(f"{'='*60}")

        samp_results = eval_independent_sampling(
            model, tokenizer, problems,
            n_samples=n_samp, max_new_tokens=max_new_tokens,
            temperature=temperature, server_url=server_url,
            verify_timeout=verify_timeout, logger=logger,
        )
        sampling_acc = sampling_accuracy_by_budget(samp_results)

        logger.info(f"\n--- Independent sampling accuracy ---")
        for k in sorted(sampling_acc):
            if k - 1 not in ROUND_CHECKPOINTS and k != n_samp:
                continue
            n_sol = int(round(sampling_acc[k] * len(problems) / 100))
            logger.info(f"  budget={k:>2}  (+{k-1} additional)  "
                        f"{n_sol:>4}/{len(problems)}  {sampling_acc[k]:>5.1f}%")

        samp_path = os.path.join(args.output_dir, "sampling_results.json")
        with open(samp_path, "w") as f:
            json.dump({
                "experiment":     "independent_sampling",
                "n_samples":      n_samp,
                "n_problems":     len(problems),
                "accuracy_by_budget": sampling_acc,
                "per_problem":    samp_results,
                "timestamp":      datetime.utcnow().isoformat() + "Z",
            }, f, indent=2)

        bgt_path = os.path.join(args.output_dir, "accuracy_by_budget.json")
        with open(bgt_path, "w") as f:
            json.dump(sampling_acc, f, indent=2)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if sampling_acc:
        plot_path = os.path.join(args.output_dir, "correction_vs_sampling.png")
        make_plot(
            correction_acc, sampling_acc,
            checkpoints=ROUND_CHECKPOINTS,
            output_path=plot_path,
            n_problems=len(problems),
        )

    # ------------------------------------------------------------------
    # Summary table (paper-style)
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*60}")
    header = f"  {'N':>3}  {'Method':35}  {'Solved':>10}  {'Acc%':>6}"
    logger.info(header)
    logger.info(f"  {'-'*60}")

    for n in ROUND_CHECKPOINTS:
        if n > max_rounds:
            break
        sc_n    = correction_acc.get(n, 0)
        sc_sol  = int(round(sc_n * len(problems) / 100))
        sc_desc = "synthesis only (Generator)" if n == 0 else f"Generator+{n}×Corrector"
        logger.info(f"  {n:>3}  {sc_desc:35}  {sc_sol:>4}/{len(problems)}  {sc_n:>5.1f}%")

    if sampling_acc:
        logger.info(f"  {'-'*60}")
        for n in ROUND_CHECKPOINTS:
            k    = n + 1
            if k > max_rounds + 1:
                break
            s_k  = sampling_acc.get(k, 0)
            s_sol = int(round(s_k * len(problems) / 100))
            logger.info(f"  {n:>3}  {'Independent sampling (k=' + str(k) + ')':35}  "
                        f"{s_sol:>4}/{len(problems)}  {s_k:>5.1f}%")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    repair_only = [r for r in sc_results
                   if r["solved"] and r.get("round_solved", 0) > 0]
    logger.info(f"\nProblems solved by correction but not synthesis alone: "
                f"{len(repair_only)}")
    for r in repair_only[:10]:
        logger.info(f"  {r['id']}  (round {r['round_solved']})")

    round_hist = Counter(
        r["round_solved"] for r in sc_results
        if r["solved"] and r["round_solved"] is not None
    )
    logger.info(f"\nSolved-at-round distribution: {dict(sorted(round_hist.items()))}")

    n_loops = sum(
        1 for r in sc_results
        if len(r["rounds"]) >= 3
        and Counter(rd["proof_body"] for rd in r["rounds"]).most_common(1)[0][1] >= 3
    )
    logger.info(f"Problems where Corrector loops (same proof ≥3 times): {n_loops}")

    logger.info(f"\nAll results written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
