"""Load prompt .md files. Keeping prompts as files (not python strings)
means you can tune the model's behaviour with a text editor and git-diff
the changes — the 'master prompt as versioned file' idea."""

import os

_DIR = os.path.dirname(os.path.realpath(__file__))


def load(name: str) -> str:
    with open(os.path.join(_DIR, name), encoding="utf-8") as fh:
        return fh.read()
