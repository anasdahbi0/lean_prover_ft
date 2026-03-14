#!/usr/bin/env python3
"""
Phase 5 – Analyze pilot verification results and print the decision report.

Run:
    python pilot_report.py
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR / "data"

VERIFIED_PATH  = DATA_DIR / "pilot_verified.jsonl"
MINIF2F_PATH   = DATA_DIR / "minif2f_test.jsonl"


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt(n, d, pct=True):
    if pct:
        return f"{n}/{d} ({n/max(d,1)*100:.1f}%)"
    return f"{n}/{d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    verified = load_jsonl(VERIFIED_PATH)
    if not verified:
        print("ERROR: data/pilot_verified.jsonl not found. Run pilot_verify.py first.")
        return

    minif2f = load_jsonl(MINIF2F_PATH)

    # ------------------------------------------------------------------ #
    # Basic counts                                                        #
    # ------------------------------------------------------------------ #
    n_candidates = len(verified)
    n_compiled   = sum(1 for r in verified if r["compiles"])

    # By template
    by_template = defaultdict(lambda: {"total": 0, "ok": 0})
    for r in verified:
        t = r.get("template_used", "?")
        by_template[t]["total"] += 1
        if r["compiles"]:
            by_template[t]["ok"] += 1

    # By stratum
    by_stratum = defaultdict(lambda: {"total": 0, "ok": 0})
    for r in verified:
        s = r.get("stratum", "?")
        by_stratum[s]["total"] += 1
        if r["compiles"]:
            by_stratum[s]["ok"] += 1

    # Per-problem pass@4: problem solved if ≥1 sample compiles
    by_problem = defaultdict(lambda: {"total": 0, "ok": 0})
    for r in verified:
        pid = r["row_id"]
        by_problem[pid]["total"] += 1
        if r["compiles"]:
            by_problem[pid]["ok"] += 1

    n_problems       = len(by_problem)
    n_problems_solved = sum(1 for v in by_problem.values() if v["ok"] > 0)
    problem_yield    = n_problems_solved / max(n_problems, 1)

    # Heuristic filter stats
    n_heuristic = sum(1 for r in verified if r.get("looks_like_tactics"))

    # Wall-time stats
    times = [r["wall_time"] for r in verified if r.get("wall_time", 0) > 0]

    # ------------------------------------------------------------------ #
    # Overlap with miniF2F                                                #
    # ------------------------------------------------------------------ #
    pilot_stmts  = {r["formal_statement"].strip()[:200] for r in verified}
    minif2f_stmts = {prob.get("formal", "").strip()[:200] for prob in minif2f}
    overlap = pilot_stmts & minif2f_stmts
    if overlap:
        print(f"WARNING: {len(overlap)} pilot statement(s) overlap with miniF2F test set!")
        for s in list(overlap)[:3]:
            print(f"  {s[:80]}")

    # ------------------------------------------------------------------ #
    # Examples                                                            #
    # ------------------------------------------------------------------ #
    successes = [r for r in verified if r["compiles"]]
    failures  = [r for r in verified if not r["compiles"] and r.get("lean_stderr")]

    # ------------------------------------------------------------------ #
    # Extrapolation                                                       #
    # ------------------------------------------------------------------ #
    # Full dataset has ~1.2M rows; assume similar filtering as pilot
    FULL_SCALE    = 200_000        # hypothetical full run size
    N_SAMPLES_FULL = 4
    expected_pairs = int(FULL_SCALE * problem_yield)

    # GPU time estimate: pilot used n_candidates / n_problems * n_problems generations
    # Formalizer on 2×H200 at roughly 1000 tok/s × 2048 max → ~2s per sample
    # Verification: ~30s per candidate on average (Lean compile)
    avg_gen_s    = 2.0   # seconds per candidate generation (rough)
    avg_verify_s = statistics.mean(times) if times else 30.0
    gpu_h_full   = (FULL_SCALE * N_SAMPLES_FULL * avg_gen_s) / 3600
    verify_h_full = (FULL_SCALE * N_SAMPLES_FULL * avg_verify_s / 32) / 3600  # 32 workers

    # ------------------------------------------------------------------ #
    # Decision                                                            #
    # ------------------------------------------------------------------ #
    if problem_yield >= 0.05:
        decision = "SCALE UP  →  proceed to full 200K run"
    elif problem_yield >= 0.02:
        decision = "CONSIDER  →  try more samples (pass@8) before committing"
    else:
        decision = "ABORT     →  Option A yield too low; proceed with Option B (direct SFT)"

    # ------------------------------------------------------------------ #
    # Print report                                                        #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 62)
    print("        Option A Pilot Results")
    print("=" * 62)
    print(f"Dataset:            minimario/Goedel-Pset-v1-Solutions")
    print(f"Model:              Goedel-LM/Goedel-Formalizer-V2-8B")
    print(f"Pilot size:         {n_problems} problems × {n_candidates // max(n_problems,1)} samples "
          f"= {n_candidates} candidates")
    templates_used = ", ".join(by_template.keys())
    print(f"Template(s) used:   {templates_used}")
    print()
    print(f"Heuristic filter:   {fmt(n_heuristic, n_candidates)} look like tactics")
    print(f"Lean verification:  {fmt(n_compiled, n_candidates)}")
    print(f"Problem yield:      {fmt(n_problems_solved, n_problems)}  (pass@{n_candidates // max(n_problems,1)})")
    print()

    if by_template:
        print("By template:")
        for t, v in sorted(by_template.items()):
            print(f"  Template {t}: {fmt(v['ok'], v['total'])}")
        print()

    if by_stratum:
        print("By solution length stratum:")
        for s, v in sorted(by_stratum.items()):
            print(f"  {s:8s}: {fmt(v['ok'], v['total'])}")
        print()

    if times:
        print(f"Avg verify time:    {statistics.mean(times):.1f}s/candidate")
        print()

    print("Extrapolation to full {:,} run:".format(FULL_SCALE))
    print(f"  Expected verified pairs: ~{expected_pairs:,}")
    print(f"  Est. GPU time (gen):     ~{gpu_h_full:.1f}h")
    print(f"  Est. verify time:        ~{verify_h_full:.1f}h")
    print()
    print(f"Recommendation: {decision}")
    print("=" * 62)

    # Examples
    if successes:
        print(f"\n--- 5 successful (statement, informal_solution, proof) ---")
        for r in successes[:5]:
            print(f"\n  [{r['row_id']}]")
            print(f"  Theorem : {r['formal_statement'][:120].replace(chr(10), ' ')} ...")
            print(f"  Informal: {r['informal_solution'][:100].replace(chr(10), ' ')} ...")
            print(f"  Proof   : {r['generated_proof'][:120].replace(chr(10), ' ')} ...")

    if failures:
        print(f"\n--- 5 failures with Lean error ---")
        for r in failures[:5]:
            print(f"\n  [{r['row_id']}]")
            print(f"  Theorem : {r['formal_statement'][:100].replace(chr(10), ' ')} ...")
            print(f"  Error   : {r['lean_stderr'][:200]}")

    # Overlap warning recap
    if overlap:
        print(f"\nWARNING: {len(overlap)} overlap(s) with miniF2F test set found.")
    else:
        print(f"\nNo overlap with miniF2F test set.")


if __name__ == "__main__":
    main()
