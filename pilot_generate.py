#!/usr/bin/env python3
"""
Phases 2 & 3 – Generate proof candidates with Goedel-Formalizer-V2-8B.

Phase 2: micro-batch (50 rows × 3 templates × 8 samples) to pick best template.
Phase 3: full pilot (2000 rows × 4 samples) with the winning template.

Run:
    python pilot_generate.py
    python pilot_generate.py --skip-micro  # skip to Phase 3 using template A
    python pilot_generate.py --template B  # force a specific template for Phase 3
"""

import argparse
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR / "data"

MODEL_ID = "Goedel-LM/Goedel-Formalizer-V2-8B"

LEAN4_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "\n"
    "set_option maxHeartbeats 0\n"
    "\n"
    "open BigOperators Real Nat Topology Rat\n"
    "\n"
)

# Tactic keywords: if the output contains one of these it "looks like" a proof
TACTIC_KEYWORDS = [
    "simp", "ring", "omega", "intro", "apply", "exact", "have", "calc",
    "rw", "norm_num", "linarith", "tauto", "decide", "constructor", "use",
    "cases", "induction", "aesop", "field_simp", "push_neg", "contrapose",
    "contradiction", "trivial", "assumption", "refine", "congr", "ext",
]

BAD_KEYWORDS = ["theorem", "import", "#check", "#eval"]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def strip_sorry(s: str) -> str:
    s = s.strip()
    for suf in [":= by sorry", "sorry"]:
        if s.endswith(suf):
            s = s[: -len(suf)].rstrip()
            break
    return s


def build_prompt_A(row: dict) -> list[dict]:
    """
    Template A – direct proof formalization request.
    Closest to how the formalizer was trained but repurposed for proof bodies.
    """
    stmt = strip_sorry(row["formal_statement"])
    sol  = row["informal_solution"]
    content = (
        "Please formalize the following informal proof into a Lean 4 tactic proof body.\n\n"
        "The theorem to prove is:\n"
        f"```lean4\n{stmt}\n```\n\n"
        "The informal proof is:\n"
        f"{sol}\n\n"
        "Write ONLY the tactic proof body – the tactics that go after `:= by`. "
        "Do NOT include the theorem statement, import lines, or `sorry`."
    )
    return [{"role": "user", "content": content}]


def build_prompt_B(row: dict) -> list[dict]:
    """
    Template B – fill-in-the-blank with a hint.
    """
    stmt = row["formal_statement"].strip()
    if not stmt.endswith("sorry"):
        stmt = stmt.rstrip() + " sorry"
    sol = row["informal_solution"]
    content = (
        "Complete the following Lean 4 proof by replacing `sorry` with a valid tactic proof.\n\n"
        f"```lean4\n{stmt}\n```\n\n"
        f"Hint (informal reasoning): {sol}\n\n"
        "Write the complete tactic proof body (what replaces `sorry`)."
    )
    return [{"role": "user", "content": content}]


def build_prompt_C_text(row: dict) -> str:
    """
    Template C – raw code completion (no informal solution).
    Returns a plain text prompt for llm.generate() rather than chat.
    """
    stmt = strip_sorry(row["formal_statement"])
    # If the statement already contains imports, use it directly; else prepend header
    if "import " in stmt:
        base = stmt
        if "maxHeartbeats" not in base:
            base = base.replace(
                "set_option maxHeartbeats 0",
                "set_option maxHeartbeats 400000",
            )
    else:
        base = LEAN4_HEADER + stmt
    return base + "\n  "   # two-space indent to continue the tactic block


# ---------------------------------------------------------------------------
# Heuristic quality check
# ---------------------------------------------------------------------------

def looks_like_tactics(text: str) -> bool:
    """Return True if the text looks like a Lean 4 tactic proof body."""
    if not text or len(text.strip()) < 2:
        return False
    has_tactic = any(kw in text for kw in TACTIC_KEYWORDS)
    has_bad    = any(kw in text for kw in BAD_KEYWORDS)
    return has_tactic and not has_bad


def is_reasonable_length(text: str, max_tokens: int = 2048) -> bool:
    return 2 <= len(text.strip()) <= max_tokens * 4   # rough char estimate


def assess_output(text: str) -> dict:
    return {
        "looks_like_tactics": looks_like_tactics(text),
        "reasonable_length":  is_reasonable_length(text),
        "length":             len(text),
    }


# ---------------------------------------------------------------------------
# Proof extraction (mirrors evaluate.py logic)
# ---------------------------------------------------------------------------

def extract_proof(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```lean4"):
        text = text[len("```lean4"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def load_model(tensor_parallel: int = 2):
    from vllm import LLM, SamplingParams   # noqa: imported lazily
    print(f"Loading {MODEL_ID} with tensor_parallel_size={tensor_parallel} ...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=tensor_parallel,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=False,
    )
    return llm


def generate_chat(llm, messages_batch: list[list[dict]], n: int,
                  temperature: float = 0.6, top_p: float = 0.95,
                  max_new_tokens: int = 2048) -> list[list[str]]:
    """
    Generate n completions per conversation.
    Returns list of lists: [[completion, ...], ...] (one inner list per row).
    """
    from vllm import SamplingParams
    params = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                            max_tokens=max_new_tokens, stop=["```\n\n"])
    outputs = llm.chat(messages_batch, params)
    return [[c.text for c in o.outputs] for o in outputs]


def generate_text(llm, prompts: list[str], n: int,
                  temperature: float = 0.6, top_p: float = 0.95,
                  max_new_tokens: int = 2048) -> list[list[str]]:
    """
    Generate n completions per raw-text prompt (Template C).
    """
    from vllm import SamplingParams
    params = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                            max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    return [[c.text for c in o.outputs] for o in outputs]


# ---------------------------------------------------------------------------
# Phase 2 – micro-batch
# ---------------------------------------------------------------------------

def run_micro_batch(llm, rows: list[dict]) -> str:
    """
    Test 3 templates on `rows` (50 rows, 8 samples each).
    Returns the name of the best template ('A', 'B', or 'C').
    """
    print(f"\n=== Phase 2: Micro-batch ({len(rows)} rows × 3 templates × 8 samples) ===")

    results = {}

    # Template A
    print("  Template A ...")
    msgs = [build_prompt_A(r) for r in rows]
    outs = generate_chat(llm, msgs, n=8)
    scores_A = [looks_like_tactics(extract_proof(c)) for row_outs in outs for c in row_outs]
    pct_A = sum(scores_A) / len(scores_A) * 100
    results["A"] = pct_A
    print(f"    A: {pct_A:.1f}% look like tactics ({sum(scores_A)}/{len(scores_A)})")

    # Template B
    print("  Template B ...")
    msgs = [build_prompt_B(r) for r in rows]
    outs = generate_chat(llm, msgs, n=8)
    scores_B = [looks_like_tactics(extract_proof(c)) for row_outs in outs for c in row_outs]
    pct_B = sum(scores_B) / len(scores_B) * 100
    results["B"] = pct_B
    print(f"    B: {pct_B:.1f}% look like tactics ({sum(scores_B)}/{len(scores_B)})")

    # Template C (raw completion)
    print("  Template C (raw completion) ...")
    texts = [build_prompt_C_text(r) for r in rows]
    outs = generate_text(llm, texts, n=8)
    scores_C = [looks_like_tactics(c) for row_outs in outs for c in row_outs]
    pct_C = sum(scores_C) / len(scores_C) * 100
    results["C"] = pct_C
    print(f"    C: {pct_C:.1f}% look like tactics ({sum(scores_C)}/{len(scores_C)})")

    best = max(results, key=results.get)
    print(f"\n  Best template: {best} ({results[best]:.1f}%)")
    return best


# ---------------------------------------------------------------------------
# Phase 3 – full pilot generation
# ---------------------------------------------------------------------------

def run_full_generation(llm, rows: list[dict], template: str,
                        n_samples: int = 4) -> list[dict]:
    """
    Generate n_samples proof candidates per row using the given template.
    Returns a list of candidate dicts for pilot_candidates.jsonl.
    """
    print(f"\n=== Phase 3: Full generation ({len(rows)} rows × {n_samples} samples, template {template}) ===")

    if template == "C":
        prompts = [build_prompt_C_text(r) for r in rows]
        all_outputs = generate_text(llm, prompts, n=n_samples)
    else:
        if template == "A":
            msgs = [build_prompt_A(r) for r in rows]
        else:
            msgs = [build_prompt_B(r) for r in rows]
        all_outputs = generate_chat(llm, msgs, n=n_samples)

    candidates = []
    n_tactic = 0
    for row, row_outs in zip(rows, all_outputs):
        for idx, raw in enumerate(row_outs):
            proof = extract_proof(raw) if template != "C" else raw
            tactic = looks_like_tactics(proof)
            if tactic:
                n_tactic += 1
            candidates.append({
                "row_id":             row["problem_id"],
                "formal_statement":   row["formal_statement"],
                "informal_solution":  row["informal_solution"],
                "stratum":            row.get("stratum", ""),
                "template_used":      template,
                "sample_idx":         idx,
                "generated_proof":    proof,
                "looks_like_tactics": tactic,
            })

    print(f"Generated {len(candidates)} candidates")
    print(f"  {n_tactic}/{len(candidates)} look like tactics "
          f"({n_tactic/max(len(candidates),1)*100:.1f}%)")

    lengths = [len(c["generated_proof"]) for c in candidates]
    if lengths:
        import statistics
        print(f"  Output length: min={min(lengths)}  "
              f"median={statistics.median(lengths):.0f}  max={max(lengths)}")

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-micro", action="store_true",
                        help="Skip Phase 2 micro-batch; go straight to Phase 3")
    parser.add_argument("--template", choices=["A", "B", "C"],
                        help="Force a specific template for Phase 3")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Samples per row in Phase 3 (default 4)")
    parser.add_argument("--tensor-parallel", type=int, default=2,
                        help="Number of GPUs for tensor parallelism (default 2)")
    parser.add_argument("--micro-n", type=int, default=50,
                        help="Rows to use in micro-batch (default 50)")
    args = parser.parse_args()

    # Load pilot subset
    subset_path = DATA_DIR / "pilot_subset.jsonl"
    if not subset_path.exists():
        print("ERROR: data/pilot_subset.jsonl not found. Run pilot_data.py first.")
        return
    rows = load_jsonl(subset_path)
    print(f"Loaded {len(rows)} rows from pilot_subset.jsonl")

    # Check for existing output
    candidates_path = DATA_DIR / "pilot_candidates.jsonl"
    if candidates_path.exists():
        existing = sum(1 for l in open(candidates_path, encoding="utf-8") if l.strip())
        if existing > 0:
            print(f"pilot_candidates.jsonl already exists ({existing} rows). Delete to re-run.")
            return

    # Load model
    llm = load_model(tensor_parallel=args.tensor_parallel)

    # Phase 2: pick template
    if args.template:
        best_template = args.template
        print(f"Template forced: {best_template}")
    elif args.skip_micro:
        best_template = "A"
        print("Skipping micro-batch; using template A")
    else:
        micro_rows = rows[:args.micro_n]
        best_template = run_micro_batch(llm, micro_rows)

    # Phase 3: full generation
    candidates = run_full_generation(llm, rows, best_template, n_samples=args.n_samples)
    save_jsonl(candidates_path, candidates)
    print(f"\nSaved {len(candidates)} candidates → {candidates_path}")


if __name__ == "__main__":
    main()
