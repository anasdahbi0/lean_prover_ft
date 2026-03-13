#!/usr/bin/env python3
"""
Download and prepare training/evaluation data for Lean 4 theorem proving.

Run:
    python prepare_data.py
"""

import json
import os
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
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    prepare_lean_workbook(os.path.join(DATA_DIR, "lean_workbook_train.jsonl"))
    prepare_minif2f(
        os.path.join(DATA_DIR, "minif2f_test.jsonl"),
        os.path.join(DATA_DIR, "minif2f_valid.jsonl"),
    )


if __name__ == "__main__":
    main()
