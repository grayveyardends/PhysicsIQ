"""store.py — dead-simple long-term memory: one .md file per fact.

Files live in memory/files/ next to this module. Every chat turn, all of
them (trimmed to a character budget) are appended to the system prompt as
"Remembered notes". The model saves new facts by calling
piq_tools.remember("the drone frame uses M3 bolts").

Plain files on purpose: you can read, edit, and delete your assistant's
memory with a text editor. No database to corrupt.
"""

import os
import re
import time

_FILES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "files")
_CHAR_BUDGET = 1500   # memory must never crowd out the actual task


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "note"


def remember(text: str) -> str:
    """Save one fact. Returns the filename (the model echoes it back)."""
    os.makedirs(_FILES_DIR, exist_ok=True)
    name = f"{int(time.time())}-{_slug(text)}.md"
    with open(os.path.join(_FILES_DIR, name), "w", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")
    return f"remembered: {name}"


def render_for_prompt() -> str:
    """All memories, newest first, cut to budget. Empty string if none —
    the caller then skips the section entirely (no token waste)."""
    if not os.path.isdir(_FILES_DIR):
        return ""
    chunks = []
    used = 0
    for name in sorted(os.listdir(_FILES_DIR), reverse=True):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(_FILES_DIR, name), encoding="utf-8") as fh:
            text = fh.read().strip()
        if used + len(text) > _CHAR_BUDGET:
            break
        chunks.append(f"- {text}")
        used += len(text)
    return "\n".join(chunks)
