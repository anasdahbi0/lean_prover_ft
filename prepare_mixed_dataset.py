#!/usr/bin/env python3
"""
Phase 1: Prepare the mixed training dataset for Experiment 2.

Combines:
  - Synthesis examples (our 29.7K verified proofs, upsampled 3x)
  - APRIL repair examples (249K, wo-exp format: no explanation)

Output:
  data/mixed_train.jsonl   ~338K examples (synthesis upsampled + APRIL train)
  data/mixed_val.jsonl     ~9.7K examples (APRIL val + 500 synthesis)

Message formats:
  Synthesis (no system prompt):
    user:      "Complete the following Lean 4 proof:\n\n{statement}"
    assistant: "{proof_body}"

  Repair wo-exp (system prompt discriminates this task):
    system:    "A Lean 4 proof attempt failed to compile. Use the compiler
               error to write a corrected proof."
    user:      "**Theorem:**\n```lean\n{statement}\n```\n\n
               **Failed proof:**\n```lean\n{incorrect_proof}\n```\n\n
               **Compiler error:**\n{error}
               [**Goal state:**\n{state_at_error}]  (omitted if empty)"
    assistant: "{correct_proof_body}"   # just tactics after ':= by'

Run:
    # Mixed dataset (Exp A control — synthesis + repair, joint training):
    python prepare_mixed_dataset.py \\
        --synthesis_path data/lean_workbook_train.jsonl \\
        --april_dir ~/data/april \\
        --upsample 3

    # APRIL-only dataset (Corrector training for Exp B):
    python prepare_mixed_dataset.py \\
        --april_dir ~/data/april \\
        --april_only \\
        --output_prefix april
    # → writes data/april_train.jsonl and data/april_val.jsonl
"""

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

REPAIR_SYSTEM_PROMPT = (
    "A Lean 4 proof attempt failed to compile. "
    "Use the compiler error to write a corrected proof."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Lean proof parsing
# ---------------------------------------------------------------------------

def extract_statement(lean_code: str) -> str | None:
    """
    Extract 'theorem foo ... := by' from a full Lean 4 file.
    Returns the declaration up to and including ':= by', or None.
    """
    match = re.search(r"^(theorem|lemma)\s+", lean_code, re.MULTILINE)
    if not match:
        return None
    body = lean_code[match.start():]
    idx = body.rfind(":= by")
    if idx == -1:
        return None
    return body[: idx + len(":= by")]


def extract_proof_body(lean_code: str) -> str | None:
    """
    Extract the tactic proof body from a full Lean 4 proof
    (everything after the last ':= by').
    """
    idx = lean_code.rfind(":= by")
    if idx == -1:
        return None
    return lean_code[idx + len(":= by") :].strip()


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def format_synthesis(statement: str, proof_body: str) -> dict:
    """Type 1: synthesis — no system prompt, matches Exp 1 training format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Complete the following Lean 4 proof:\n\n{statement}",
            },
            {
                "role": "assistant",
                "content": proof_body,
            },
        ]
    }


def format_repair(
    statement: str,
    incorrect_proof: str,
    error: str,
    state_at_error: str,
    correct_proof_body: str,
) -> dict:
    """
    Type 2: repair (wo-exp) — system prompt marks repair task.
    Assistant outputs only the corrected proof body (no explanation).
    """
    user_content = (
        f"**Theorem:**\n```lean\n{statement}\n```\n\n"
        f"**Failed proof:**\n```lean\n{incorrect_proof}\n```\n\n"
        f"**Compiler error:**\n{error}"
    )
    if state_at_error and state_at_error.strip():
        user_content += f"\n\n**Goal state:**\n{state_at_error}"

    return {
        "messages": [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": correct_proof_body},
        ]
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_synthesis(path: str) -> list[dict]:
    """Load synthesis JSONL ({statement, proof}) and format as Type 1 messages."""
    logger.info(f"Loading synthesis data: {path}")
    raw     = load_jsonl(path)
    rows    = []
    skipped = 0

    for row in raw:
        statement  = (row.get("statement") or "").strip()
        proof_body = (row.get("proof")     or "").strip()
        if not statement or not proof_body:
            skipped += 1
            continue
        if "sorry" in proof_body:
            skipped += 1
            continue
        rows.append(format_synthesis(statement, proof_body))

    logger.info(f"  Synthesis formatted: {len(rows):,}  skipped: {skipped}")
    return rows


def load_april(april_dir: Path, filenames: list[str]) -> list[dict]:
    """
    Load one or more APRIL JSONL files and format as Type 2 repair messages.

    APRIL correct_proof / incorrect_proof fields contain full Lean 4 files
    (with 'import Mathlib' headers).  We extract:
      - statement      = theorem signature up to ':= by'  (from correct_proof)
      - correct_body   = tactic proof body after ':= by'  (from correct_proof)
      - incorrect_proof = the full broken file             (shown to model as-is)
    """
    rows    = []
    skipped = 0
    schema_logged = False

    for fname in filenames:
        fpath = april_dir / fname
        if not fpath.exists():
            logger.warning(f"  APRIL file not found, skipping: {fpath}")
            continue

        logger.info(f"  Loading {fpath.name} ...")
        raw = load_jsonl(str(fpath))
        logger.info(f"    {len(raw):,} rows")

        # Log schema once so we can verify the field names
        if not schema_logged and raw:
            schema_logged = True
            logger.info(f"    Columns: {list(raw[0].keys())}")
            sample_cp = raw[0].get("correct_proof", "")
            logger.info(f"    correct_proof[:300]: {sample_cp[:300]!r}")

        for row in raw:
            incorrect_proof = (row.get("incorrect_proof") or "").strip()
            correct_proof   = (row.get("correct_proof")   or "").strip()
            error           = (row.get("error")           or "").strip()
            state_at_error  = (row.get("state_at_error")  or "").strip()

            if not incorrect_proof or not correct_proof or not error:
                skipped += 1
                continue

            statement   = extract_statement(correct_proof)
            correct_body = extract_proof_body(correct_proof)

            if not statement or not correct_body:
                skipped += 1
                continue

            if "sorry" in correct_body:
                skipped += 1
                continue

            rows.append(format_repair(
                statement       = statement,
                incorrect_proof = incorrect_proof,
                error           = error,
                state_at_error  = state_at_error,
                correct_proof_body = correct_body,
            ))

        logger.info(f"    Formatted so far: {len(rows):,}")

    logger.info(f"  APRIL total formatted: {len(rows):,}  skipped: {skipped}")
    return rows


# ---------------------------------------------------------------------------
# Statistics + sample printer
# ---------------------------------------------------------------------------

def print_stats(rows: list[dict], label: str, rng: random.Random, n_samples: int = 3):
    n_repair = sum(1 for r in rows if r["messages"][0]["role"] == "system")
    n_synth  = len(rows) - n_repair

    total_chars_synth  = sum(
        sum(len(m["content"]) for m in r["messages"])
        for r in rows if r["messages"][0]["role"] != "system"
    )
    total_chars_repair = sum(
        sum(len(m["content"]) for m in r["messages"])
        for r in rows if r["messages"][0]["role"] == "system"
    )
    avg_synth_tok  = total_chars_synth  / max(n_synth,  1) / 4
    avg_repair_tok = total_chars_repair / max(n_repair, 1) / 4

    empty_asst = sum(
        1 for r in rows if not r["messages"][-1].get("content", "").strip()
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  {label}: {len(rows):,} total examples")
    logger.info(f"    Synthesis : {n_synth:,}  ({100*n_synth/max(len(rows),1):.1f}%)")
    logger.info(f"    Repair    : {n_repair:,}  ({100*n_repair/max(len(rows),1):.1f}%)")
    logger.info(f"    Avg tokens (est) synthesis : {avg_synth_tok:.0f}")
    logger.info(f"    Avg tokens (est) repair    : {avg_repair_tok:.0f}")
    logger.info(f"    Empty assistant messages   : {empty_asst}")

    # Print random samples
    synth_pool  = [r for r in rows if r["messages"][0]["role"] != "system"]
    repair_pool = [r for r in rows if r["messages"][0]["role"] == "system"]

    if synth_pool:
        logger.info(f"\n  --- {min(n_samples, len(synth_pool))} synthesis samples ---")
        for row in rng.sample(synth_pool, min(n_samples, len(synth_pool))):
            for m in row["messages"]:
                logger.info(f"    [{m['role']:10s}] {m['content'][:200]!r}")
            logger.info("")

    if repair_pool:
        logger.info(f"\n  --- {min(n_samples, len(repair_pool))} repair samples ---")
        for row in rng.sample(repair_pool, min(n_samples, len(repair_pool))):
            for m in row["messages"]:
                logger.info(f"    [{m['role']:10s}] {m['content'][:200]!r}")
            logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthesis_path",
        default="data/lean_workbook_train.jsonl",
        help="Path to synthesis JSONL (statement + proof pairs). "
             "Default: data/lean_workbook_train.jsonl",
    )
    parser.add_argument(
        "--april_dir",
        default="data/april",
        help="Directory containing APRIL JSONL files (from HuggingFace)",
    )
    parser.add_argument(
        "--output_dir", default="data",
        help="Where to write mixed_train.jsonl and mixed_val.jsonl",
    )
    parser.add_argument(
        "--upsample", type=int, default=3,
        help="How many times to repeat synthesis data (default 3 → ~1:2.8 ratio)",
    )
    parser.add_argument(
        "--val_synthesis_n", type=int, default=500,
        help="Synthesis examples to include in val set (default 500)",
    )
    parser.add_argument(
        "--april_only", action="store_true",
        help="Skip synthesis data entirely — output APRIL repair examples only. "
             "Use this for Corrector training in Exp B.",
    )
    parser.add_argument(
        "--output_prefix", default="mixed",
        help="Prefix for output files: {prefix}_train.jsonl, {prefix}_val.jsonl "
             "(default 'mixed'). Use 'april' for the Corrector dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Resolve paths relative to script dir
    synthesis_path = (
        SCRIPT_DIR / args.synthesis_path
        if not os.path.isabs(args.synthesis_path)
        else Path(args.synthesis_path)
    )
    april_dir  = Path(args.april_dir)
    output_dir = SCRIPT_DIR / args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load synthesis (skip if --april_only)
    # ------------------------------------------------------------------
    synth_rows = []
    if not args.april_only:
        if not synthesis_path.exists():
            raise FileNotFoundError(
                f"Synthesis data not found: {synthesis_path}\n"
                "Run: python prepare_data.py"
            )
        synth_rows = load_synthesis(str(synthesis_path))
        if not synth_rows:
            raise RuntimeError(f"No synthesis examples loaded from {synthesis_path}")
        logger.info(f"Synthesis examples: {len(synth_rows):,}")
    else:
        logger.info("--april_only set: skipping synthesis data")

    # ------------------------------------------------------------------
    # Load APRIL train (4 files, in train/ subdirectory)
    # ------------------------------------------------------------------
    april_train_dir = april_dir / "train"
    logger.info(f"\nLoading APRIL train data from: {april_train_dir}")
    april_train_rows = load_april(
        april_train_dir,
        ["thme_train.jsonl", "tme_train.jsonl", "mlme_train.jsonl", "lme_train.jsonl"],
    )
    if not april_train_rows:
        raise RuntimeError(
            f"No APRIL train examples loaded from {april_train_dir}\n"
            "Expected files: thme_train.jsonl, tme_train.jsonl, mlme_train.jsonl, lme_train.jsonl\n"
            "Download with:\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('uw-math-ai/APRIL', repo_type='dataset', local_dir='~/data/april')\""
        )

    # ------------------------------------------------------------------
    # Load APRIL val (4 files, in val/ subdirectory)
    # ------------------------------------------------------------------
    april_val_dir = april_dir / "val"
    logger.info(f"\nLoading APRIL val data from: {april_val_dir}")
    april_val_rows = load_april(
        april_val_dir,
        ["thme_val.jsonl", "tme_val.jsonl", "mlme_val.jsonl", "lme_val.jsonl"],
    )
    logger.info(f"APRIL val examples: {len(april_val_rows):,}")

    # ------------------------------------------------------------------
    # Build train set
    # ------------------------------------------------------------------
    if args.april_only:
        train_rows = april_train_rows
        logger.info(f"\nAPRIL-only train: {len(train_rows):,}")
    else:
        synth_upsampled = synth_rows * args.upsample
        ratio = len(april_train_rows) / max(len(synth_upsampled), 1)
        logger.info(f"\nSynthesis ×{args.upsample}: {len(synth_upsampled):,}")
        logger.info(f"APRIL train:         {len(april_train_rows):,}")
        logger.info(f"Synthesis:repair ratio: 1:{ratio:.1f}")
        train_rows = synth_upsampled + april_train_rows

    rng.shuffle(train_rows)
    logger.info(f"Total train: {len(train_rows):,}")

    # ------------------------------------------------------------------
    # Build val set
    # ------------------------------------------------------------------
    if args.april_only:
        val_rows = april_val_rows
    else:
        val_synth = rng.sample(synth_rows, min(args.val_synthesis_n, len(synth_rows)))
        val_rows  = val_synth + april_val_rows

    rng.shuffle(val_rows)
    logger.info(f"Total val:   {len(val_rows):,}")

    # ------------------------------------------------------------------
    # Stats + samples
    # ------------------------------------------------------------------
    print_stats(train_rows, "TRAIN", rng, n_samples=3)
    print_stats(val_rows,   "VAL",   rng, n_samples=1)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    prefix    = args.output_prefix
    train_out = str(output_dir / f"{prefix}_train.jsonl")
    val_out   = str(output_dir / f"{prefix}_val.jsonl")

    save_jsonl(train_out, train_rows)
    save_jsonl(val_out,   val_rows)

    logger.info(f"\nSaved train → {train_out}  ({len(train_rows):,} examples)")
    logger.info(f"Saved val   → {val_out}  ({len(val_rows):,} examples)")

    if args.april_only:
        logger.info("\nNext: train_exp2.py --config config_corrector.yaml")
    else:
        logger.info("\nNext: verify_april.py, then train_exp2.py --config config_exp2_mixed.yaml")


if __name__ == "__main__":
    main()
