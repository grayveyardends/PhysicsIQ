#!/usr/bin/env python3
"""claude_debug_wrapper.py — temp debug proxy for the `claude` CLI.

Stands in for the real `claude` binary, forwards the call unchanged, and
writes one text file per cloud request showing what PhysicsIQ sent (system
prompt + prompt payload) and what came back.

Enable:  point the CloudBin setting at this file's absolute path. From
         FreeCAD's Python console:
    FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/PhysicsIQ"
    ).SetString("CloudBin", "<abs path>/scripts/claude_debug_wrapper.py")
Disable: set CloudBin back to "claude".

Logs go to runs/claude_debug/ (override with PIQ_CLAUDE_DEBUG_DIR). The real
binary is found on PATH, or set PIQ_REAL_CLAUDE.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
LOG_DIR = os.environ.get("PIQ_CLAUDE_DEBUG_DIR",
                         os.path.join(REPO, "runs", "claude_debug"))

# long base64 runs are attached images — elide them so the log stays readable
_B64 = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def real_claude():
    override = os.environ.get("PIQ_REAL_CLAUDE")
    if override:
        return override
    found = shutil.which("claude")
    if found and os.path.realpath(found) != os.path.realpath(__file__):
        return found
    sys.exit("claude_debug_wrapper: real 'claude' not found "
             "(set PIQ_REAL_CLAUDE)")


def arg_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def clean_answer(raw_lines):
    """Pull the readable answer out of the CLI's stream-json output."""
    streamed, final, cost = [], None, None
    for line in raw_lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "stream_event":
            delta = (ev.get("event") or {}).get("delta") or {}
            if delta.get("type") == "text_delta":
                streamed.append(delta.get("text") or "")
        elif kind == "result":
            cost = ev.get("total_cost_usd")
            if ev.get("subtype") == "success":
                final = ev.get("result")
    return (final if final is not None else "".join(streamed)), cost


def main():
    argv = sys.argv[1:]
    stdin_bytes = sys.stdin.buffer.read()   # app writes the payload, then EOF
    started = time.time()

    child = subprocess.Popen(
        [real_claude(), *argv],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    err_chunks = []

    def pump_err():
        for chunk in iter(lambda: child.stderr.read(4096), b""):
            err_chunks.append(chunk)
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()

    err_thread = threading.Thread(target=pump_err, daemon=True)
    err_thread.start()

    child.stdin.write(stdin_bytes)
    child.stdin.close()

    # tee the child's stdout to ours line by line so the app still streams
    raw_lines = []
    for line in iter(child.stdout.readline, b""):
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        raw_lines.append(line.decode("utf-8", "replace"))
    code = child.wait()
    err_thread.join(timeout=2)

    try:
        write_log(argv, stdin_bytes, raw_lines, err_chunks, started, code)
    except Exception as exc:  # never let logging break the real call
        sys.stderr.write(f"claude_debug_wrapper: log failed: {exc}\n")

    sys.exit(code)


def write_log(argv, stdin_bytes, raw_lines, err_chunks, started, code):
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(LOG_DIR, f"claude-{stamp}-{os.getpid()}.txt")

    answer, cost = clean_answer(raw_lines)
    system = arg_value(argv, "--system-prompt") or "(none)"
    payload = _B64.sub(lambda m: f"<{len(m.group(0))} base64 chars elided>",
                       stdin_bytes.decode("utf-8", "replace"))
    err = b"".join(err_chunks).decode("utf-8", "replace").strip()

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"time      {stamp}\n")
        fh.write(f"model     {arg_value(argv, '--model') or '?'}\n")
        fh.write(f"duration  {time.time() - started:.1f}s   exit {code}")
        fh.write(f"   cost ${cost}\n" if cost is not None else "\n")

        fh.write("\n----- system prompt sent to claude -----\n")
        fh.write(system + "\n")
        fh.write("\n----- prompt payload (stdin) -----\n")
        fh.write(payload + "\n")
        fh.write("\n----- answer from claude -----\n")
        fh.write((answer or "(empty)") + "\n")
        if err:
            fh.write("\n----- stderr -----\n" + err + "\n")


if __name__ == "__main__":
    main()
