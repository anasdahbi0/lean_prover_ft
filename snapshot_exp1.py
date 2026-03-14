#!/usr/bin/env python3
"""
Upload Experiment 1 artifacts to HuggingFace.
Usage:
    HF_TOKEN=hf_... python snapshot_exp1.py
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ID = "yotsubian/qwen"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN environment variable")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)
    ok = []
    fail = []

    uploads = [
        (SCRIPT_DIR / "model_card.md",               "README.md"),
        (SCRIPT_DIR / "results" / "eval_baseline.json",  "results/eval_baseline.json"),
        (SCRIPT_DIR / "results" / "eval_finetuned.json", "results/eval_finetuned.json"),
        (SCRIPT_DIR / "config_exp1.yaml",             "config_exp1.yaml"),
    ]

    for local_path, repo_path in uploads:
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            ok.append(repo_path)
            print(f"  uploaded: {repo_path}")
        except Exception as e:
            fail.append(repo_path)
            print(f"  FAILED:   {repo_path} — {e}")

    print()
    for f in ok:
        print(f"✓ {f}")
    for f in fail:
        print(f"✗ {f}")

    if not fail:
        print(f"\nExperiment 1 frozen. Safe to start Experiment 2.")

if __name__ == "__main__":
    main()
