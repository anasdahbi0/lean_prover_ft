#!/usr/bin/env python3
"""
Download and prepare training/evaluation data for Lean 4 theorem proving.

Run:
    python prepare_data.py
"""

import json
import logging
import os
import random
import re
from pathlib import Path

from datasets import load_dataset

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

DATA_DIR = str(SCRIPT_DIR / "data")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Lean-workbook-proofs
# ---------------------------------------------------------------------------

def prepare_lean_workbook(output_path):
    """
    Download Goedel-LM/Lean-workbook-proofs and extract (statement, proof) pairs.

    Each row has a full_proof field that is a complete Lean 4 file, e.g.:
        import Mathlib
        ...
        theorem lean_workbook_N ... := by
          <tactic proof>

    Strategy: locate the first 'theorem' or 'lemma' keyword in the file,
    take everything from there, then split on the last ':= by'.
    This is robust regardless of whether the header matches exactly.

    Output: data/lean_workbook_train.jsonl
        {"id": "lean_workbook_12345", "statement": "theorem ... := by", "proof": "simp"}
    """
    if os.path.exists(output_path):
        existing = load_jsonl(output_path)
        if len(existing) > 0:
            print(f"lean_workbook_train.jsonl : {len(existing)} rows  (already exists, skipping)")
            return
        # File exists but is empty — fall through and re-generate

    print("Downloading Goedel-LM/Lean-workbook-proofs ...")
    ds = load_dataset("Goedel-LM/Lean-workbook-proofs", split="train")

    # Print the column names of the first row to aid debugging
    if len(ds) > 0:
        print(f"  Dataset columns: {list(ds.features.keys())}")
        sample = ds[0]
        for k, v in sample.items():
            preview = str(v)[:120].replace("\n", "\\n")
            print(f"  [{k}] {preview}")

    rows = []
    skipped = 0
    for i, example in enumerate(ds):
        # The field name is 'full_proof' per the dataset card;
        # fall back to 'proof' in case of a schema variant.
        full_proof = example.get("full_proof") or example.get("proof", "")
        if not full_proof:
            skipped += 1
            continue

        # Find the first 'theorem' or 'lemma' declaration in the file.
        # This is robust to any header format — we don't need to strip it.
        thm_match = re.search(r"^(theorem|lemma)\s+", full_proof, re.MULTILINE)
        if not thm_match:
            skipped += 1
            continue

        # Body = from the theorem/lemma keyword to end of file
        body = full_proof[thm_match.start():]

        # Split on the last ':= by' to separate signature from tactic proof.
        # rfind avoids splitting on ':= by' that may appear in type annotations.
        idx = body.rfind(":= by")
        if idx == -1:
            skipped += 1
            continue  # term-mode proof — skip

        statement = body[: idx + len(":= by")]
        proof = body[idx + len(":= by") :].strip()

        if not proof:
            skipped += 1
            continue

        row_id = f"lean_workbook_{i}"
        rows.append({
            "id": row_id,
            "statement": statement,
            "proof": proof,
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    save_jsonl(output_path, rows)
    print(f"lean_workbook_train.jsonl : {len(rows)} rows  ({skipped} skipped)")


# ---------------------------------------------------------------------------
# Goedel-Pset-v1-solved
# ---------------------------------------------------------------------------

def _extract_lean_block(text: str) -> str:
    """Return the content of the last ```lean4 ... ``` fenced block in text."""
    matches = list(re.finditer(r"```lean4\s*\n(.*?)```", text, re.DOTALL))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _parse_sft_row(example):
    """
    Parse one row from Goedel-LM/SFT_dataset_v2.
    Each row has a 'messages' list: [{role, content}, {role, content}].

    User content:
        "Complete the following Lean 4 code:\n\n```lean4\n...theorem ... := by sorry```\n\n..."
        → extract the lean4 block, strip trailing 'sorry' to get statement ending in ':= by'

    Assistant content:
        Long reasoning trace ending with a ```lean4\ntheorem ... := by\n  <proof>\n``` block.
        → extract the last lean4 block, then split on ':= by' to get just the tactic proof.

    Returns (statement, proof) or (None, None) on failure.
    """
    messages = example.get("messages", [])
    if len(messages) < 2:
        return None, None

    user_content = messages[0].get("content", "")
    assistant_content = messages[1].get("content", "")

    # --- Statement from user content ---
    user_lean = _extract_lean_block(user_content)
    if not user_lean:
        return None, None

    # Find theorem/lemma declaration
    thm_match = re.search(r"^(theorem|lemma)\s+", user_lean, re.MULTILINE)
    if not thm_match:
        return None, None

    body = user_lean[thm_match.start():]
    idx = body.rfind(":= by")
    if idx == -1:
        return None, None

    # Strip everything after ':= by' (which is ' sorry' in the user prompt)
    statement = body[: idx + len(":= by")]

    # --- Proof from assistant content ---
    asst_lean = _extract_lean_block(assistant_content)
    if not asst_lean:
        return None, None

    # Split on last ':= by' to isolate tactic proof body
    proof_idx = asst_lean.rfind(":= by")
    if proof_idx == -1:
        return None, None

    proof = asst_lean[proof_idx + len(":= by"):].strip()
    if not proof or "sorry" in proof:
        return None, None

    return statement, proof


def prepare_goedel_pset(output_path: str, n: int = 100_000, seed: int = 42):
    """
    Download Goedel-LM/SFT_dataset_v2 via streaming and reservoir-sample n rows.
    Extract (statement, proof) pairs from the messages format.
    Idempotent: skips if output_path already exists and is non-empty.
    """
    if os.path.exists(output_path):
        existing_count = sum(1 for _ in open(output_path, encoding="utf-8") if _.strip())
        if existing_count > 0:
            print(f"goedel_pset_100k.jsonl    : {existing_count} rows  (already exists, skipping)")
            return

    print(f"Downloading Goedel-LM/SFT_dataset_v2 (streaming, sampling {n:,}) ...")
    ds = load_dataset("Goedel-LM/SFT_dataset_v2", split="train", streaming=True)

    rng = random.Random(seed)
    reservoir = []
    for i, row in enumerate(ds):
        if len(reservoir) < n:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = row
        if (i + 1) % 100_000 == 0:
            print(f"  streamed {i + 1:,} rows, reservoir={len(reservoir):,} ...")

    print(f"Sampled:   {len(reservoir):,}")

    rows = []
    skipped = 0
    for i, example in enumerate(reservoir):
        statement, proof = _parse_sft_row(example)
        if statement is None:
            skipped += 1
            continue
        rows.append({"id": f"goedel_pset_{i}", "statement": statement, "proof": proof})

    os.makedirs(DATA_DIR, exist_ok=True)
    save_jsonl(output_path, rows)
    print(f"Extracted: {len(rows):,}")
    print(f"Skipped:   {skipped:,}")
    print(f"Saved to:  {output_path}")


# ---------------------------------------------------------------------------
# miniF2F
# ---------------------------------------------------------------------------

def prepare_minif2f(output_test, output_valid):
    """
    Download cat-searcher/minif2f-lean4 and extract test/validation splits.

    Each row has:
        formal_statement : ends with ':= by sorry'  →  we strip 'sorry'
        informal_statement (or informal) : natural language description
        id (or name) : problem identifier

    Output: data/minif2f_test.jsonl, data/minif2f_valid.jsonl
        {"id": "...", "formal": "theorem ... := by", "informal": "..."}
    """
    test_exists = os.path.exists(output_test)
    valid_exists = os.path.exists(output_valid)

    if test_exists and valid_exists:
        print(f"minif2f_test.jsonl        : {len(load_jsonl(output_test))} rows  (already exists, skipping)")
        print(f"minif2f_valid.jsonl       : {len(load_jsonl(output_valid))} rows  (already exists, skipping)")
        return

    print("Downloading cat-searcher/minif2f-lean4 ...")
    ds_test = load_dataset("cat-searcher/minif2f-lean4", split="test")
    ds_valid = load_dataset("cat-searcher/minif2f-lean4", split="validation")

    def process_split(ds):
        rows = []
        for example in ds:
            formal = example.get("formal_statement", "").strip()
            # informal field name varies by dataset version
            informal = example.get(
                "informal_statement",
                example.get("informal", example.get("problem", "")),
            )
            row_id = example.get("id", example.get("name", ""))

            # Strip ':= by sorry' → ':= by'
            # Handle both 'sorry' alone and ':= by sorry' suffixes
            if formal.endswith(":= by sorry"):
                formal = formal[: -len("sorry")].rstrip()
            elif formal.endswith("sorry"):
                # e.g. already stripped ':= by' present elsewhere — just drop sorry
                formal = formal[: -len("sorry")].rstrip()

            rows.append({
                "id": row_id,
                "formal": formal,
                "informal": informal,
            })
        return rows

    os.makedirs(DATA_DIR, exist_ok=True)

    if not test_exists:
        rows = process_split(ds_test)
        save_jsonl(output_test, rows)
        print(f"minif2f_test.jsonl        : {len(rows)} rows")
    else:
        print(f"minif2f_test.jsonl        : {len(load_jsonl(output_test))} rows  (already exists, skipping)")

    if not valid_exists:
        rows = process_split(ds_valid)
        save_jsonl(output_valid, rows)
        print(f"minif2f_valid.jsonl       : {len(rows)} rows")
    else:
        print(f"minif2f_valid.jsonl       : {len(load_jsonl(output_valid))} rows  (already exists, skipping)")


# ---------------------------------------------------------------------------
# NuminaMath-Lean
# ---------------------------------------------------------------------------

def prepare_numina(output_path, split="test"):
    """
    Download AI-MO/NuminaMath-Lean and extract formal statements for evaluation.

    The dataset has a 'formal_statement' field ending with ':= by sorry'.
    We strip 'sorry' to get the statement ending in ':= by', matching miniF2F format.

    Output: data/numina_test.jsonl
        {"id": "...", "formal": "theorem ... := by", "informal": "..."}
    """
    if os.path.exists(output_path):
        existing = load_jsonl(output_path)
        if existing:
            print(f"numina_{split}.jsonl          : {len(existing)} rows  (already exists, skipping)")
            return

    print(f"Downloading AI-MO/NuminaMath-Lean ({split} split) ...")
    ds = load_dataset("AI-MO/NuminaMath-Lean", split=split)

    if len(ds) > 0:
        print(f"  Columns: {list(ds.features.keys())}")

    rows = []
    skipped = 0
    for example in ds:
        formal = (
            example.get("formal_statement")
            or example.get("formal")
            or example.get("statement", "")
        ).strip()

        if not formal:
            skipped += 1
            continue

        # Strip trailing sorry (same logic as miniF2F)
        if formal.endswith(":= by sorry"):
            formal = formal[: -len("sorry")].rstrip()
        elif formal.endswith("sorry"):
            formal = formal[: -len("sorry")].rstrip()

        # Skip if no ':= by' — can't evaluate tactic proof generation
        if ":= by" not in formal:
            skipped += 1
            continue

        informal = (
            example.get("informal_statement")
            or example.get("informal")
            or example.get("problem", "")
        )
        row_id = example.get("id", example.get("name", example.get("problem_id", "")))

        rows.append({"id": row_id, "formal": formal, "informal": informal})

    os.makedirs(DATA_DIR, exist_ok=True)
    save_jsonl(output_path, rows)
    print(f"numina_{split}.jsonl          : {len(rows)} rows  ({skipped} skipped)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goedel-pset", action="store_true",
                        help="Only run prepare_goedel_pset()")
    parser.add_argument("--numina", action="store_true",
                        help="Only run prepare_numina()")
    parser.add_argument("--n", type=int, default=100_000,
                        help="Number of rows to sample for goedel-pset (default 100000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.goedel_pset:
        prepare_goedel_pset(
            os.path.join(DATA_DIR, "goedel_pset_100k.jsonl"),
            n=args.n,
            seed=args.seed,
        )
    elif args.numina:
        prepare_numina(os.path.join(DATA_DIR, "numina_test.jsonl"), split="train")
    else:
        prepare_lean_workbook(os.path.join(DATA_DIR, "lean_workbook_train.jsonl"))
        prepare_minif2f(
            os.path.join(DATA_DIR, "minif2f_test.jsonl"),
            os.path.join(DATA_DIR, "minif2f_valid.jsonl"),
        )


if __name__ == "__main__":
    main()
