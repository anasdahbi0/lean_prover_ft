#!/usr/bin/env python3
"""
Minimal HTTP wrapper around `lake exe repl`.

Accepts POST requests with the same JSON format as Goedel-Prover's verifier:
    {"cmd": "<lean code>", "allTactics": false, ...}

Returns the raw JSON output from the Lean REPL.

Run:
    python lean_server.py --workspace /workspace/Goedel-Prover/mathlib4 --port 8000
"""

import argparse
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a separate thread."""
    daemon_threads = True

LAKE_PATH = os.path.expanduser("~/.elan/bin/lake")


def run_lean_repl(command: dict, workspace: str, timeout: int = 120) -> dict:
    message = json.dumps(command, ensure_ascii=False) + "\r\n\r\n"
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as f:
        f.write(message)
        f.seek(0)
        result = subprocess.run(
            [LAKE_PATH, "exe", "repl"],
            stdin=f,
            capture_output=True,
            text=True,
            cwd=workspace,
            timeout=timeout,
        )
    return json.loads(result.stdout) if result.stdout.strip() else {"messages": [], "errors": []}


class LeanHandler(BaseHTTPRequestHandler):
    workspace = None
    timeout = 120

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            command = json.loads(body)
            result = run_lean_repl(command, self.workspace, self.timeout)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        pass  # suppress per-request logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace/Goedel-Prover/mathlib4")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    LeanHandler.workspace = args.workspace
    LeanHandler.timeout = args.timeout

    print(f"Lean REPL server listening on port {args.port}")
    print(f"Workspace: {args.workspace}")
    ThreadedHTTPServer(("0.0.0.0", args.port), LeanHandler).serve_forever()


if __name__ == "__main__":
    main()
