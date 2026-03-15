#!/usr/bin/env python3
"""
Phase 0.4 sanity check: verify APRIL correct_proof fields compile on our Lean version.

Picks N random rows from APRIL train files, compiles correct_proof via the
Lean HTTP server, and reports the pass rate.

Pass criteria: >=18/20 (90%) compile.
If <90% compile, STOP — likely a Lean/Mathlib version mismatch.
APRIL was built against leanprover/lean4:v4.22.0-rc4.

Run (lean server must be running first):
    python verify_april.py --april_dir ~/data/april
    python verify_april.py --april_dir ~/data/april --n 50 --server_url http://localhost:8000
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_lean4_http(code: str, server_url: str, timeout: int = 120) -> tuple[bool, list[str]]:
    """
    Send Lean code to the HTTP REPL server.
    Returns (success, list_of_error_messages).
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

        errors   = [m for m in result_raw.get("messages", []) if m.get("severity") == "error"]
        sorries  = result_raw.get("sorries", [])
        bad_warn = [m for m in result_raw.get("messages", [])
                    if m.get("severity") == "warning"
                    and ("declaration uses 'sorry'" in m.get("data", "")
                         or "failed" in m.get("data", ""))]

        success = not errors and not sorries and not bad_warn
        msgs    = [e.get("data", "") for e in errors]
        return success, msgs

    except requests.exceptions.ConnectionError:
        return False, [
            "CONNECTION ERROR — is the Lean server running?\n"
            "  python lean_server.py --workspace /workspace/Goedel-Prover/mathlib4"
        ]
    except Exception as exc:
        return False, [str(exc)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify APRIL correct_proof fields compile on our Lean version."
    )
    parser.add_argument("--april_dir", default="data/april",
                        help="Directory containing APRIL JSONL files")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of proofs to sample (default 20)")
    parser.add_argument("--server_url", default="http://localhost:8000",
                        help="Lean REPL HTTP server URL")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Compilation timeout per proof in seconds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    april_dir  = Path(args.april_dir)
    train_dir  = april_dir / "train"   # files live in train/ subdirectory

    # ------------------------------------------------------------------
    # Load a pool of APRIL examples (stop after we have enough to sample from)
    # ------------------------------------------------------------------
    all_rows = []
    for fname in ["thme_train.jsonl", "tme_train.jsonl", "mlme_train.jsonl", "lme_train.jsonl"]:
        fpath = train_dir / fname
        if not fpath.exists():
            logger.warning(f"Not found, skipping: {fpath}")
            continue
        rows = load_jsonl(str(fpath))
        all_rows.extend(rows)
        logger.info(f"Loaded {len(rows):,} rows from {fname}  (pool: {len(all_rows):,})")
        if len(all_rows) >= max(args.n * 20, 5_000):
            break

    if not all_rows:
        logger.error(f"No APRIL JSONL files found in: {train_dir}")
        logger.error(
            "Expected ~/data/april/train/thme_train.jsonl etc.\n"
            "Re-download with:\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('uw-math-ai/APRIL', repo_type='dataset', local_dir='~/data/april')\""
        )
        sys.exit(1)

    rng    = random.Random(args.seed)
    sample = rng.sample(all_rows, min(args.n, len(all_rows)))

    # ------------------------------------------------------------------
    # Print schema + 3 correct_proof previews so we can inspect the format
    # ------------------------------------------------------------------
    logger.info("\n=== First row schema ===")
    for k, v in sample[0].items():
        preview = str(v)[:160].replace("\n", "\\n")
        logger.info(f"  {k!r}: {preview!r}")

    logger.info("\n=== 3 correct_proof previews (first 400 chars each) ===")
    for i, row in enumerate(sample[:3]):
        cp = row.get("correct_proof", "")
        logger.info(f"\n  [{i+1}]\n{cp[:400]}")

    # ------------------------------------------------------------------
    # Preflight: check server is reachable before wasting time on 20 proofs
    # ------------------------------------------------------------------
    logger.info(f"\nChecking Lean server at {args.server_url} ...")
    try:
        requests.post(args.server_url, json={"cmd": "#check Nat", "allTactics": False,
                      "ast": False, "tactics": False, "premises": False}, timeout=10)
        logger.info("  Server reachable.")
    except requests.exceptions.ConnectionError:
        logger.error(
            f"\nCannot reach Lean server at {args.server_url}\n\n"
            "Start it first:\n"
            "  python lean_server.py --workspace /workspace/Goedel-Prover/mathlib4\n\n"
            "Then re-run:\n"
            "  python verify_april.py"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compile each sample
    # ------------------------------------------------------------------
    logger.info(f"\nCompiling {len(sample)} proofs via {args.server_url} ...")
    passed     = 0
    failed_log = []

    for i, row in enumerate(sample):
        cp = row.get("correct_proof", "").strip()

        if not cp:
            logger.info(f"  [{i+1:3d}/{len(sample)}] SKIP — empty correct_proof")
            failed_log.append({"idx": i, "reason": "empty", "errors": []})
            continue

        # Prepend Mathlib import if the file doesn't start with one
        if not cp.startswith("import"):
            cp = "import Mathlib\n\n" + cp

        ok, errors = verify_lean4_http(cp, args.server_url, args.timeout)

        if ok:
            passed += 1
            logger.info(f"  [{i+1:3d}/{len(sample)}] PASS")
        else:
            err_preview = errors[0][:120] if errors else "unknown error"
            logger.info(f"  [{i+1:3d}/{len(sample)}] FAIL  {err_preview!r}")
            failed_log.append({"idx": i, "reason": "compile_error", "errors": errors})

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    threshold = int(len(sample) * 0.90)
    print(f"\n{'='*55}")
    print(f"  Result: {passed}/{len(sample)} correct_proof fields compiled")
    print(f"{'='*55}")

    if passed >= threshold:
        print("\n  PASS — Lean version is compatible. Proceed to training.")
        sys.exit(0)
    else:
        print(f"\n  FAIL — Only {passed}/{len(sample)} compiled (need >= {threshold}).")
        print("  This indicates a Lean or Mathlib version mismatch.")
        print()
        print("  APRIL was built with: leanprover/lean4:v4.22.0-rc4")
        print("  Check your workspace toolchain:")
        print("    cat /workspace/Goedel-Prover/mathlib4/lean-toolchain")
        print()
        print("  To fix: update your lean-toolchain to v4.22.0-rc4")
        print("    elan install leanprover/lean4:v4.22.0-rc4")
        print()
        print("  First failures:")
        for entry in failed_log[:5]:
            errs = entry["errors"]
            msg  = errs[0][:200] if errs else entry["reason"]
            print(f"    [idx={entry['idx']}] {msg!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
