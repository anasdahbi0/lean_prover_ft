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
from aiohttp import web

WORKSPACE = "/workspace/Goedel-Prover/mathlib4"
MAX_CONCURRENT = 4   # DO NOT INCREASE — server silently fails above this
TIMEOUT = 120        # DO NOT DECREASE — proofs need up to 90s

semaphore: asyncio.Semaphore = None  # initialised in main()


async def run_lean_repl(command: dict) -> dict:
    """Run lake exe repl with the given command, respecting the concurrency limit."""
    repl_input = (json.dumps(command, ensure_ascii=False) + "\r\n\r\n").encode()

    async with semaphore:
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", "source ~/.elan/env && lake exe repl",
                cwd=WORKSPACE,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,  # new process group → clean kill on timeout
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=repl_input),
                    timeout=TIMEOUT,
                )
            except asyncio.TimeoutError:
                # Kill the entire process group so no zombie lake processes remain
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
                return {"messages": [{"severity": "error", "data": f"timeout after {TIMEOUT}s"}], "sorries": []}

        except Exception as e:
            return {"messages": [{"severity": "error", "data": str(e)}], "sorries": []}

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    if stdout_text:
        try:
            return json.loads(stdout_text)
        except json.JSONDecodeError:
            return {"messages": [{"severity": "error", "data": f"repl returned non-JSON: {stdout_text[:200]}"}], "sorries": []}
    return {"messages": [], "sorries": []}


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
    global WORKSPACE, semaphore

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    WORKSPACE = args.workspace
    global TIMEOUT
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
