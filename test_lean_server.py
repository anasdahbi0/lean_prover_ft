#!/usr/bin/env python3
"""
Smoke-test the Lean verification server against a handful of known examples.

Run:
    # In one terminal:
    python lean_server.py --workspace /workspace/lean_prover_ft/mathlib4 --port 8000

    # In another:
    python test_lean_server.py
"""

import json
import sys
import requests

SERVER_URL = "http://localhost:8000"

HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "\n"
    "set_option maxHeartbeats 0\n"
    "\n"
    "open BigOperators Real Nat Topology Rat\n"
    "\n"
)

CASES = [
    {
        "name":     "valid: norm_num closes 1+1=2",
        "code":     HEADER + "theorem test1 : 1 + 1 = 2 := by norm_num",
        "expect_complete": True,
    },
    {
        "name":     "valid: ring closes simple algebra",
        "code":     HEADER + "theorem test2 (x : ℝ) : (x + 1) ^ 2 = x ^ 2 + 2 * x + 1 := by ring",
        "expect_complete": True,
    },
    {
        "name":     "invalid: wrong goal (1+1=3)",
        "code":     HEADER + "theorem test3 : 1 + 1 = 3 := by norm_num",
        "expect_complete": False,
    },
    {
        "name":     "sorry: should be rejected",
        "code":     HEADER + "theorem test4 : 1 + 1 = 3 := by sorry",
        "expect_complete": False,
    },
    {
        "name":     "valid: omega closes nat arithmetic",
        "code":     HEADER + "theorem test5 (n : ℕ) (h : n > 3) : n ≥ 4 := by omega",
        "expect_complete": True,
    },
    {
        "name":     "invalid: tactic does not close goal",
        "code":     HEADER + "theorem test6 (a b : ℝ) (h : a < b) : a ≤ b := by ring",
        "expect_complete": False,
    },
]


def verify(code: str) -> tuple:
    resp = requests.post(SERVER_URL, json={
        "cmd": code,
        "allTactics": False,
        "ast": False,
        "tactics": False,
        "premises": False,
    }, timeout=180)
    resp.raise_for_status()
    raw = resp.json()

    errors   = [m for m in raw.get("messages", []) if m["severity"] == "error"]
    warnings = [m for m in raw.get("messages", []) if m["severity"] == "warning"]
    sorries  = raw.get("sorries", [])

    passed   = not errors
    complete = (
        passed
        and not sorries
        and not any(
            "declaration uses 'sorry'" in w["data"] or "failed" in w["data"]
            for w in warnings
        )
    )
    return {"pass": passed, "complete": complete, "errors": errors, "sorries": sorries, "_raw": raw}


def main():
    # Health check first
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        info = r.json()
        print(f"Server OK — max_concurrent={info['max_concurrent']}  timeout={info['timeout']}s\n")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {SERVER_URL}: {e}")
        sys.exit(1)

    passed_all = True
    for case in CASES:
        try:
            result = verify(case["code"])
        except Exception as e:
            print(f"FAIL  {case['name']}")
            print(f"      exception: {e}")
            passed_all = False
            continue

        ok = result["complete"] == case["expect_complete"]
        status = "OK  " if ok else "FAIL"
        if not ok:
            passed_all = False

        expect_str = "complete=True" if case["expect_complete"] else "complete=False"
        got_str    = f"complete={result['complete']}  pass={result['pass']}"
        print(f"{status}  {case['name']}")
        print(f"      expected: {expect_str}")
        print(f"      got:      {got_str}")
        if result["errors"]:
            print(f"      errors:   {result['errors'][0]['data'][:120]}")
        print(f"      raw:      {json.dumps(result['_raw'])[:300]}")
        print()

    if passed_all:
        print("All cases passed.")
    else:
        print("Some cases FAILED — check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
