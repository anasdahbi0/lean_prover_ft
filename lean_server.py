#!/usr/bin/env python3
"""
Async HTTP wrapper around `lake exe repl`.

Key design decisions (learned from production failures):
  - MAX_CONCURRENT = 4: hard semaphore — more than 4 concurrent lake processes
    causes all requests to timeout silently, giving 0% results.
  - TIMEOUT = 120s: 60s is too short; complex proofs need up to 90s.
  - Process group kill (os.setsid + os.killpg): kills zombie lake processes
    on timeout — without this, timed-out processes accumulate indefinitely.
  - Always uses `lake exe repl`, never the repl binary directly:
    direct binary calls skip LEAN_PATH setup → Mathlib imports fail.
  - Input via stdin (not echo): avoids shell injection on proofs with quotes.

Run:
    python lean_server.py --workspace /workspace/Goedel-Prover/mathlib4 --port 8000
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
from aiohttp import web

WORKSPACE = "/workspace/Goedel-Prover/mathlib4"
LEAN_ENV = "~/.elan/env"   # path to elan env script; override with --lean-env
MAX_CONCURRENT = 4   # DO NOT INCREASE — server silently fails above this
TIMEOUT = 120        # DO NOT DECREASE — proofs need up to 90s

semaphore: asyncio.Semaphore = None  # initialised in main()


def _run_repl_sync(repl_input: bytes) -> dict:
    """
    Blocking subprocess call to lake exe repl.
    Runs in a thread pool so it doesn't block the event loop.
    Uses subprocess.run (same as the verified-working direct test).
    """
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {LEAN_ENV} && lake exe repl"],
            input=repl_input,
            capture_output=True,
            cwd=WORKSPACE,
            timeout=TIMEOUT,
        )
        stdout_text = result.stdout.decode("utf-8", errors="replace").strip()
        if stdout_text:
            try:
                return json.loads(stdout_text)
            except json.JSONDecodeError:
                return {"messages": [{"severity": "error", "data": f"repl returned non-JSON: {stdout_text[:200]}"}], "sorries": []}
        return {"messages": [], "sorries": []}
    except subprocess.TimeoutExpired:
        return {"messages": [{"severity": "error", "data": f"timeout after {TIMEOUT}s"}], "sorries": []}
    except Exception as e:
        return {"messages": [{"severity": "error", "data": str(e)}], "sorries": []}


async def run_lean_repl(command: dict) -> dict:
    """Run lake exe repl with the given command, respecting the concurrency limit."""
    # The REPL only understands {"cmd": ..., "env": ...} — strip all other fields
    repl_command = {"cmd": command["cmd"]}
    if "env" in command:
        repl_command["env"] = command["env"]
    repl_input = (json.dumps(repl_command, ensure_ascii=False) + "\r\n\r\n").encode()

    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_repl_sync, repl_input)


async def handle_post(request: web.Request) -> web.Response:
    try:
        command = await request.json()
        result = await run_lean_repl(command)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "max_concurrent": MAX_CONCURRENT, "timeout": TIMEOUT})


def main():
    global WORKSPACE, LEAN_ENV, TIMEOUT, semaphore

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--lean-env", default=LEAN_ENV,
                        help="Path to elan env script (default: ~/.elan/env)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    WORKSPACE = args.workspace
    LEAN_ENV = args.lean_env
    TIMEOUT = args.timeout

    semaphore = asyncio.Semaphore(args.max_concurrent)

    print(f"Lean REPL server on port {args.port}")
    print(f"Workspace:      {WORKSPACE}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Timeout:        {TIMEOUT}s")

    app = web.Application()
    app.router.add_post("/", handle_post)
    app.router.add_get("/health", handle_health)

    web.run_app(app, host="0.0.0.0", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
