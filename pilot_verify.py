#!/usr/bin/env python3
"""
Phase 4 – Lean verification of pilot candidates.

Reads data/pilot_candidates.jsonl, assembles a complete Lean 4 file for each
candidate, verifies via the running lean_server.py (HTTP), and writes
data/pilot_verified.jsonl with `compiles` and `lean_stderr` fields.

Prerequisites:
    python lean_server.py &   # must be running

Run:
    python pilot_verify.py
    python pilot_verify.py --workers 16 --timeout 90
"""

import argparse
import json
import re
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR / "data"

CANDIDATES_PATH = DATA_DIR / "pilot_candidates.jsonl"
OUTPUT_PATH     = DATA_DIR / "pilot_verified.jsonl"

LEAN4_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "\n"
    "set_option maxHeartbeats 400000\n"
    "\n"
    "open BigOperators Real Nat Topology Rat\n"
    "\n"
)


# ---------------------------------------------------------------------------
# Helpers
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


def strip_sorry(s: str) -> str:
    s = s.strip()
    for suf in [":= by sorry", "sorry"]:
        if s.endswith(suf):
            s = s[: -len(suf)].rstrip()
            break
    return s


def assemble_lean_file(formal_statement: str, proof_body: str) -> str | None:
    """
    Assemble a complete Lean 4 file for verification.
    Returns None if we can't construct a valid file.
    """
    stmt = strip_sorry(formal_statement)

    if "import " in stmt:
        # Formal statement already includes imports.
        # Ensure maxHeartbeats is set reasonably.
        stmt = re.sub(
            r"set_option maxHeartbeats \d+",
            "set_option maxHeartbeats 400000",
            stmt,
        )
        if "maxHeartbeats" not in stmt:
            # Insert after the last preamble line
            lines = stmt.split("\n")
            insert_at = 0
            for i, line in enumerate(lines):
                if line.startswith(("import ", "open ", "set_option")):
                    insert_at = i
            lines.insert(insert_at + 1, "set_option maxHeartbeats 400000")
            stmt = "\n".join(lines)
    else:
        stmt = LEAN4_HEADER + stmt

    # stmt must end with ':= by'
    if not stmt.rstrip().endswith(":= by"):
        return None

    # Indent the proof body (two spaces per line)
    proof_lines = proof_body.strip().split("\n")
    indented = "\n".join("  " + l if l.strip() else l for l in proof_lines)
    return stmt + "\n" + indented + "\n"


def verify_one(candidate: dict, server_url: str, timeout: int) -> dict:
    """
    Verify a single candidate. Returns the candidate dict augmented with
    `compiles`, `lean_stderr`, and `wall_time`.
    """
    result = dict(candidate)
    proof = candidate.get("generated_proof", "")

    if not proof.strip() or "sorry" in proof:
        result.update(compiles=False, lean_stderr="rejected: empty or sorry",
                      wall_time=0.0)
        return result

    lean_code = assemble_lean_file(candidate["formal_statement"], proof)
    if lean_code is None:
        result.update(compiles=False, lean_stderr="rejected: could not assemble",
                      wall_time=0.0)
        return result

    command = {
        "cmd":         lean_code,
        "allTactics":  False,
        "ast":         False,
        "tactics":     False,
        "premises":    False,
    }
    t0 = time.time()
    try:
        resp = requests.post(server_url, json=command, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()
        errors   = [m for m in raw.get("messages", []) if m["severity"] == "error"]
        warnings = [m for m in raw.get("messages", []) if m["severity"] == "warning"]
        sorries  = raw.get("sorries", [])
        compiles = (
            not errors
            and not sorries
            and not any(
                "declaration uses 'sorry'" in w.get("data", "") or
                "failed" in w.get("data", "")
                for w in warnings
            )
        )
        stderr = "; ".join(m.get("data", "") for m in errors)[:300] if errors else ""
    except Exception:
        compiles = False
        stderr = traceback.format_exc()[-200:]

    result.update(
        compiles=compiles,
        lean_stderr=stderr,
        wall_time=round(time.time() - t0, 2),
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",  default="http://localhost:8000",
                        help="Lean server URL (default http://localhost:8000)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel verification workers (default 32)")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Per-candidate timeout seconds (default 90)")
    args = parser.parse_args()

    if not CANDIDATES_PATH.exists():
        print("ERROR: data/pilot_candidates.jsonl not found. Run pilot_generate.py first.")
        return

    # Check lean server is reachable
    try:
        requests.get(args.server, timeout=3)
    except Exception:
        # lean_server.py only handles POST; a GET error is expected but proves it's up
        pass
    try:
        requests.post(args.server, json={"cmd": "#check Nat", "allTactics": False,
                                          "ast": False, "tactics": False, "premises": False},
                      timeout=10)
    except Exception as e:
        print(f"WARNING: lean_server may not be running ({e}). "
              "Start it with: python lean_server.py")

    if OUTPUT_PATH.exists():
        existing = sum(1 for l in open(OUTPUT_PATH, encoding="utf-8") if l.strip())
        if existing > 0:
            print(f"pilot_verified.jsonl already exists ({existing} rows). Delete to re-run.")
            return

    candidates = load_jsonl(CANDIDATES_PATH)
    print(f"Verifying {len(candidates)} candidates with {args.workers} workers ...")

    verified = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(verify_one, c, args.server, args.timeout): i
            for i, c in enumerate(candidates)
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            verified.append(result)
            if done % 100 == 0 or done == len(candidates):
                n_ok = sum(1 for r in verified if r["compiles"])
                elapsed = time.time() - t_start
                print(f"  [{done}/{len(candidates)}] "
                      f"compiled so far: {n_ok} "
                      f"({n_ok/done*100:.1f}%)  "
                      f"elapsed: {elapsed:.0f}s")

    # Sort back to original order
    order = {id(c): i for i, c in enumerate(candidates)}
    # (order is approximate since we used id before submit—just save as-is)

    save_jsonl(OUTPUT_PATH, verified)

    # Quick summary
    n_total  = len(verified)
    n_ok     = sum(1 for r in verified if r["compiles"])
    elapsed  = time.time() - t_start

    print(f"\nVerification complete in {elapsed:.0f}s")
    print(f"Compiled: {n_ok}/{n_total} ({n_ok/max(n_total,1)*100:.1f}%)")

    # Most common errors
    errors = [r["lean_stderr"] for r in verified if not r["compiles"] and r["lean_stderr"]]
    if errors:
        # Classify by first keyword
        def classify_error(e):
            for kw in ["timeout", "TIMEOUT", "type mismatch", "unknown tactic",
                       "unknown identifier", "failed to synthesize", "parse error",
                       "rejected", "application type mismatch"]:
                if kw.lower() in e.lower():
                    return kw
            return "other"
        counts = Counter(classify_error(e) for e in errors)
        print("\nTop error types:")
        for err, cnt in counts.most_common(8):
            print(f"  {cnt:5d}  {err}")

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
