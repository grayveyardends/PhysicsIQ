"""inferio — the Python package behind the Inferio workbench.

Layout map (read docs/ARCHITECTURE.md in the repo for the full story):

    llm/      talk to llama-server (start it, stream chat from it)
    rag/      find relevant FreeCAD scripting recipes for a prompt
    tools/    things the LLM (or you) can run inside FreeCAD
    agent/    the chat loop that glues llm + rag + tools together
    ui/       Qt widgets (dock panel, viewers) + the threading pattern
    memory/   long-term .md-file memory

This file only holds path helpers, because every other module needs to
answer "where am I installed?" and "where is the repo?".
"""

import os

# __file__ = .../Inferio/inferio/__init__.py
# realpath() matters: the addon is usually a SYMLINK from FreeCAD's Mod dir
# into the git repo (see scripts/dev_install.sh). realpath follows it so we
# can find repo-level folders like models/ and pinn/.
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))   # .../Inferio/inferio
ADDON_DIR = os.path.dirname(PACKAGE_DIR)                    # .../Inferio
REPO_DIR = os.path.dirname(ADDON_DIR)                       # repo root


def icon_path() -> str:
    return os.path.join(ADDON_DIR, "icons", "inferio.svg")


def default_models_dir() -> str:
    """Where the .gguf files live. In a dev checkout that's <repo>/models."""
    return os.path.join(REPO_DIR, "models")


def runs_dir() -> str:
    """Working folder for PINN jobs (exported STEP, bcs.json, results).

    Lives in the repo so you can inspect results, and is .gitignore'd.
    """
    path = os.path.join(REPO_DIR, "runs")
    os.makedirs(path, exist_ok=True)
    return path
