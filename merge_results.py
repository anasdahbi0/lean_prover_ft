#!/usr/bin/env python3
"""
Merge parallel evaluation results from multiple GPU shards.

Usage:
  # Merge Exp A shards
  python merge_results.py --mode exp_a \
      --inputs results/exp_a_shard_*.json \
      --output results/exp_a.json

  # Merge Exp B shards
  python merge_results.py --mode exp_b \
      --inputs results/exp_b_shard_*/per_problem.json \
      --output_dir results/exp_b/
"""

import argparse
import glob
import json
from math import comb
from pathlib import Path
from datetime import datetime


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def merge_exp_a(input_paths, output_path):
    all_per_problem = []
    meta = None

    for path in sorted(input_paths):
        with open(path) as f:
            data = json.load(f)
        all_per_problem.extend(data["per_problem"])
        if meta is None:
            meta = data

    n_samples = meta["n_samples"]
    n_total   = len(all_per_problem)
    n_solved  = sum(1 for r in all_per_problem if r["solved"])

    ks = [1, 2, 4, 8, 32]
    pass_at_k_values = {}
    for k in ks:
        if k > n_samples:
            continue
        avg = sum(
            pass_at_k(n_samples, r["n_correct"], k)
            for r in all_per_problem
        ) / max(n_total, 1)
        pass_at_k_values[f"pass@{k}"] = round(avg, 4)

    result = {
        "model":       meta["model"],
        "adapter":     meta["adapter"],
        "dataset":     meta["dataset"],
        "n_samples":   n_samples,
        "n_total":     n_total,
        "n_solved":    n_solved,
        "pass_at_k":   pass_at_k_values,
        "per_problem": all_per_problem,
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "note":        f"Merged from {len(input_paths)} shards",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Merged {n_total} problems from {len(input_paths)} shards")
    print(f"Solved: {n_solved}/{n_total}")
    for k_label, v in pass_at_k_values.items():
        print(f"  {k_label}: {v:.4f}  ({v*100:.1f}%)")
    print(f"Written: {output_path}")


def merge_exp_b(input_paths, output_dir):
    all_per_problem = []
    meta = None

    for path in sorted(input_paths):
        with open(path) as f:
            data = json.load(f)
        all_per_problem.extend(data["per_problem"])
        if meta is None:
            meta = data

    max_rounds = meta.get("max_rounds", 12)
    n_problems = len(all_per_problem)

    # Recompute accuracy_by_round
    accuracy_by_round = {}
    for budget in range(0, max_rounds + 1):
        n_solved = sum(
            1 for r in all_per_problem
            if r["solved"]
            and r.get("round_solved") is not None
            and r["round_solved"] <= budget
        )
        accuracy_by_round[budget] = round(100.0 * n_solved / max(n_problems, 1), 2)

    result = {
        "experiment":        "two_model_self_correction",
        "generator_adapter": meta.get("generator_adapter"),
        "corrector_adapter": meta.get("corrector_adapter"),
        "max_rounds":        max_rounds,
        "n_problems":        n_problems,
        "n_solved_total":    sum(1 for r in all_per_problem if r["solved"]),
        "accuracy_by_round": accuracy_by_round,
        "per_problem":       all_per_problem,
        "timestamp":         datetime.utcnow().isoformat() + "Z",
        "note":              f"Merged from {len(input_paths)} shards",
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "per_problem.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    acc_path = Path(output_dir) / "accuracy_by_round.json"
    with open(acc_path, "w") as f:
        json.dump(accuracy_by_round, f, indent=2)

    print(f"Merged {n_problems} problems from {len(input_paths)} shards")
    print(f"Accuracy by round:")
    for rnd, acc in sorted(accuracy_by_round.items()):
        n_sol = int(round(acc * n_problems / 100))
        print(f"  Round {rnd}: {n_sol}/{n_problems}  ({acc:.1f}%)")
    print(f"Written: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    required=True, choices=["exp_a", "exp_b"])
    parser.add_argument("--inputs",  required=True, nargs="+")
    parser.add_argument("--output",  default=None, help="For exp_a: output JSON path")
    parser.add_argument("--output_dir", default=None, help="For exp_b: output directory")
    args = parser.parse_args()

    # Expand glob patterns
    paths = []
    for p in args.inputs:
        expanded = glob.glob(p)
        paths.extend(expanded if expanded else [p])
    paths = [p for p in paths if Path(p).exists()]

    if not paths:
        print(f"ERROR: No input files found matching: {args.inputs}")
        return

    print(f"Found {len(paths)} shard files")

    if args.mode == "exp_a":
        merge_exp_a(paths, args.output or "results/exp_a.json")
    else:
        merge_exp_b(paths, args.output_dir or "results/exp_b/")


if __name__ == "__main__":
    main()
