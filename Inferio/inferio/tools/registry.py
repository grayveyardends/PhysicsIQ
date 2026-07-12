"""registry.py — the catalogue of helpers the LLM may call.

Design decision (important for a 2B model): we do NOT use JSON function-
calling. Small models fumble JSON tool schemas constantly. Instead the
model writes ordinary Python and calls helpers on a pre-imported module:

    inferio_tools.annotate("Gear1")
    inferio_tools.arrange_views("tile")

One mechanism (python code blocks) = one thing to get right.

`make_namespace()` builds that module-like object for exec_python.py.
`render_docs()` writes the {TOOL_DOCS} section of the master prompt, so
the prompt NEVER goes out of sync with what's actually callable — the
docs are generated from this table.
"""

from types import SimpleNamespace


def _tool_table():
    # Imported lazily: some of these touch FreeCADGui, which does not
    # exist under headless freecadcmd (used by tests).
    from inferio.tools import annotate as _annotate
    from inferio.tools import ui_control

    return [
        # (callable, signature shown to the model, one-line description)
        (_annotate.annotate,
         'annotate(label, text=None)',
         "put a floating text note next to the object with this Label; "
         "text=None writes the object's dimensions"),
        (ui_control.arrange_views,
         "arrange_views(mode)",
         "rearrange open windows: mode is 'tile', 'cascade' or 'tabbed'"),
        (ui_control.split_model_and_sketch,
         "split_model_and_sketch(sketch_label=None)",
         "3D viewport left + top-down 2D view right, side by side"),
        (ui_control.set_view,
         "set_view(direction)",
         "snap camera: 'iso','front','top','right','left','rear','bottom'"),
    ]


def make_namespace() -> SimpleNamespace:
    ns = SimpleNamespace()
    for fn, _sig, _doc in _tool_table():
        setattr(ns, fn.__name__, fn)
    return ns


def render_docs() -> str:
    lines = []
    for _fn, sig, doc in _tool_table():
        lines.append(f"- inferio_tools.{sig} — {doc}")
    return "\n".join(lines)
