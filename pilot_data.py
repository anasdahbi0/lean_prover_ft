#!/usr/bin/env python3
"""
Phase 1 – Explore minimario/Goedel-Pset-v1-Solutions and create a stratified
pilot subset of 2,000 rows for Option A.

Run:
    python pilot_data.py
"""

import json
import random
import statistics
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT = DATA_DIR / "pilot_subset.jsonl"

PILOT_N = 2000
SHORT_N = 500
MEDIUM_N = 1000
LONG_N = 500
SEED = 42

MIN_SOL_LEN = 20   # chars – skip trivially empty solutions


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_histogram(values, n_bins=8, label="chars"):
    lo, hi = min(values), max(values)
    step = (hi - lo) / n_bins or 1
    bins = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / step), n_bins - 1)
        bins[idx] += 1
    print(f"  Solution length histogram ({label}):")
    for i, count in enumerate(bins):
        lo_b = lo + i * step
        hi_b = lo_b + step
        bar = "█" * (count * 40 // max(bins))
        print(f"    {lo_b:6.0f}-{hi_b:6.0f}: {bar} ({count:,})")


def main():
    if OUTPUT.exists():
        existing = sum(1 for l in open(OUTPUT, encoding="utf-8") if l.strip())
        if existing > 0:
            print(f"pilot_subset.jsonl already exists ({existing} rows). Delete to re-run.")
            return

    print("Loading minimario/Goedel-Pset-v1-Solutions (streaming) ...")
    ds = load_dataset("minimario/Goedel-Pset-v1-Solutions", split="train", streaming=True)

    # Print schema from first row
    first = next(iter(ds))
    print(f"\nColumns: {list(first.keys())}")
    print("Sample row:")
    for k, v in first.items():
        preview = str(v)[:120].replace("\n", "\\n")
        print(f"  [{k}] {preview}")

    # Collect eligible rows in a single streaming pass
    print(f"\nStreaming all rows (filtering: is_proof_problem=True, len(informal_solution) > {MIN_SOL_LEN}) ...")
    eligible = []
    total = 0
    n_proof = 0

    for row in ds:
        total += 1
        if total % 200_000 == 0:
            print(f"  scanned {total:,}, eligible so far: {len(eligible):,}")

        if not row.get("is_proof_problem"):
            continue
        n_proof += 1

        sol = (row.get("informal_solution") or "").strip()
        if len(sol) <= MIN_SOL_LEN:
            continue

        eligible.append({
            "problem_id":         row["problem_id"],
            "formal_statement":   row["formal_statement"],
            "informal_statement": row["informal_statement"],
            "informal_solution":  sol,
            "sol_len":            len(sol),
        })

    print(f"\nTotal rows scanned  : {total:,}")
    print(f"is_proof_problem=True: {n_proof:,}")
    print(f"Eligible (non-empty sol): {len(eligible):,}")

    if len(eligible) == 0:
        print("ERROR: no eligible rows found. Check dataset schema.")
        return

    # Stats
    lengths = [r["sol_len"] for r in eligible]
    q = statistics.quantiles(lengths, n=4)
    q25, q75 = q[0], q[2]
    print(f"\nSolution length stats:")
    print(f"  min={min(lengths):,}  q25={q25:,.0f}  median={statistics.median(lengths):,.0f}  "
          f"q75={q75:,.0f}  max={max(lengths):,}")
    print_histogram(lengths)

    # Stratified sample
    short  = [r for r in eligible if r["sol_len"] <= q25]
    long_  = [r for r in eligible if r["sol_len"] >= q75]
    medium = [r for r in eligible if q25 < r["sol_len"] < q75]

    print(f"\nStrata: short={len(short):,}  medium={len(medium):,}  long={len(long_):,}")

    rng = random.Random(SEED)
    sn = min(SHORT_N, len(short))
    mn = min(MEDIUM_N, len(medium))
    ln = min(LONG_N, len(long_))

    subset = (
        rng.sample(short,  sn) +
        rng.sample(medium, mn) +
        rng.sample(long_,  ln)
    )
    rng.shuffle(subset)

    for row in subset:
        row["stratum"] = (
            "short"  if row["sol_len"] <= q25 else
            "long"   if row["sol_len"] >= q75 else
            "medium"
        )

    save_jsonl(OUTPUT, subset)

    print(f"\nSaved {len(subset)} rows → {OUTPUT}")
    print(f"  short: {sn}  medium: {mn}  long: {ln}")


if __name__ == "__main__":
    main()
