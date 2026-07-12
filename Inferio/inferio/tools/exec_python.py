"""exec_python.py — runs LLM-generated Python inside FreeCAD, safely-ish.

"Safely-ish" means three concrete guarantees, not a sandbox:

1. UNDOABLE. Everything runs inside one document transaction, so a bad
   script is one Ctrl+Z away from gone. This is why users can dare to
   enable auto-run.
2. CONTAINED. Exceptions are caught and returned as data (traceback
   string) instead of exploding the GUI. The traceback is exactly what
   the agent loop feeds back to the model for self-correction.
3. OBSERVED. stdout is captured, so `print()` in generated code becomes
   visible output in the chat panel.

MUST run on the main thread (it touches the document — see workers.py
for the ONE RULE). The agent loop guarantees that: it only calls this
from Qt slots.
"""

import contextlib
import io
import traceback
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    error: str          # short one-liner ("" when ok)
    traceback: str      # full traceback for the model ("" when ok)


def _build_globals():
    """The namespace generated code runs in. Everything the master prompt
    promises ('already imported') must actually be here."""
    import math

    import FreeCAD as App
    import FreeCADGui as Gui
    import Part
    from inferio.tools import registry

    g = {
        "App": App,
        "Gui": Gui,
        "Part": Part,
        "math": math,
        "inferio_tools": registry.make_namespace(),
    }
    try:
        import Draft
        g["Draft"] = Draft
    except ImportError:
        pass  # Draft workbench not available in some minimal builds
    return g


def run_freecad_python(code: str, transaction_name="Inferio AI edit") -> ExecResult:
    import FreeCAD as App
    import FreeCADGui as Gui

    # The master prompt says "App.ActiveDocument always exists" — we make
    # that true here so generated code never has to handle the None case.
    doc = App.ActiveDocument
    if doc is None:
        doc = App.newDocument("Inferio")

    out = io.StringIO()
    doc.openTransaction(transaction_name)  # ← the undo checkpoint
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(code, "<inferio-generated>", "exec"), _build_globals())
        doc.commitTransaction()
        doc.recompute()
        try:
            Gui.SendMsgToActiveView("ViewFit")  # zoom so new parts are visible
        except Exception:
            pass  # no active 3D view (e.g. fresh doc) — cosmetic only
        return ExecResult(True, out.getvalue(), "", "")
    except Exception as exc:  # noqa: BLE001 — LLM code can raise anything
        doc.abortTransaction()  # roll the document back to the checkpoint
        return ExecResult(
            ok=False,
            stdout=out.getvalue(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=6),
        )
