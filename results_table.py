#!/usr/bin/env python3
"""
Print a comparison table from one or more eval result JSON files.

Run:
    python results_table.py results/eval_baseline.json results/eval_finetuned.json
"""

import json
import sys


# ---------------------------------------------------------------------------
# Reference results from published papers (hardcoded)
# ---------------------------------------------------------------------------

REFERENCE_ROWS = [
    {
        "model":     "Goedel-Prover-SFT (paper)",
        "n_solved":  142,
        "n_total":   244,
        "pass@1":    None,
        "pass@8":    None,
        "pass@32":   0.576,
    },
    {
        "model":     "DeepSeek-Prover-V2-7B (paper)",
        "n_solved":  None,
        "n_total":   244,
        "pass@1":    None,
        "pass@8":    None,
        "pass@32":   0.734,
    },
]


def load_result(path):
    with open(path) as f:
        return json.load(f)


def fmt_solved(n_solved, n_total):
    if n_solved is None:
        return "--"
    return f"{n_solved}/{n_total}"


def fmt_float(v):
    if v is None:
        return "--"
    return f"{v:.3f}"


def model_label(result, path):
    """
    Build a display name from the result dict.
    Uses model name + '+ adapter' suffix when an adapter was used.
    """
    name = result.get("model", path)
    # Shorten the HF model path to the model name only
    name = name.split("/")[-1]
    adapter = result.get("adapter")
    if adapter:
        adapter_label = adapter.rstrip("/").split("/")[-1]
        name = f"{name} + {adapter_label}"
    return name


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python results_table.py results/eval_baseline.json [results/eval_finetuned.json ...]")
        sys.exit(1)

    # Load all results
    rows = []
    for path in paths:
        r = load_result(path)
        pak = r.get("pass_at_k", {})
        rows.append({
            "model":    model_label(r, path),
            "n_solved": r.get("n_solved"),
            "n_total":  r.get("n_total", 244),
            "pass@1":   pak.get("pass@1"),
            "pass@8":   pak.get("pass@8"),
            "pass@32":  pak.get("pass@32"),
        })

    # Column widths
    col_model   = max(34, max(len(r["model"]) for r in rows + REFERENCE_ROWS))
    col_solved  = 9
    col_pass1   = 8
    col_pass8   = 8
    col_pass32  = 8

    sep_model  = "-" * col_model
    sep_solved = "-" * col_solved
    sep_pass   = "-" * col_pass1

    header = (
        f"{'Model':<{col_model}} | {'Solved':<{col_solved}} | "
        f"{'pass@1':<{col_pass1}} | {'pass@8':<{col_pass8}} | {'pass@32':<{col_pass32}}"
    )
    divider = (
        f"{sep_model}-|-{sep_solved}-|-"
        f"{sep_pass}-|-{sep_pass}-|-{sep_pass}"
    )

    print()
    print(header)
    print(divider)

    for r in rows:
        print(
            f"{r['model']:<{col_model}} | "
            f"{fmt_solved(r['n_solved'], r['n_total']):<{col_solved}} | "
            f"{fmt_float(r['pass@1']):<{col_pass1}} | "
            f"{fmt_float(r['pass@8']):<{col_pass8}} | "
            f"{fmt_float(r['pass@32']):<{col_pass32}}"
        )

    # Reference separator
    ref_sep = f"{'── reference ──':<{col_model}}-{'─' * (col_solved + col_pass1 * 3 + 9)}"
    print(ref_sep)

    for r in REFERENCE_ROWS:
        print(
            f"{r['model']:<{col_model}} | "
            f"{fmt_solved(r['n_solved'], r['n_total']):<{col_solved}} | "
            f"{fmt_float(r['pass@1']):<{col_pass1}} | "
            f"{fmt_float(r['pass@8']):<{col_pass8}} | "
            f"{fmt_float(r['pass@32']):<{col_pass32}}"
        )

    print()


if __name__ == "__main__":
    main()
