"""registry.py — the catalogue of helpers the LLM may call.

Design decision (important for a 2B model): we do NOT use JSON function-
calling. Small models fumble JSON tool schemas constantly. Instead the
model writes ordinary Python and calls helpers on a pre-imported module:

    piq_tools.annotate("Gear1")
    piq_tools.arrange_views("tile")

One mechanism (python code blocks) = one thing to get right.

`make_namespace()` builds that module-like object for exec_python.py.
`render_docs()` writes the {TOOL_DOCS} section of the master prompt, so
the prompt NEVER goes out of sync with what's actually callable — the
docs are generated from this table.
"""

from types import SimpleNamespace


def _run_stress_analysis(label=None, force_N=(0, 0, -500),
                         fixed="zmin", load="zmax"):
    """Kick off the break-point PINN through the chat controller so its
    results flow back into the conversation (the closed loop)."""
    from physicsiq import panel_manager
    panel = panel_manager.chat_panel_or_none()
    if panel is None:
        raise RuntimeError("chat panel is not open")
    panel.controller.request_pinn_analysis(
        target_label=label, force_N=force_N,
        fixed_rule=fixed, load_rule=load)
    return "stress analysis started — results will appear in the chat"


def _remember(text):
    from physicsiq.memory import store
    return store.remember(text)


def _tool_table():
    # Imported lazily: some of these touch FreeCADGui, which does not
    # exist under headless freecadcmd (used by tests).
    from physicsiq.tools import annotate as _annotate
    from physicsiq.tools import ui_control, ui_widgets

    return [
        (ui_widgets.popup,
         "popup(message, title='PhysicsIQ')",
         "show a non-blocking info popup window to the user"),
        (ui_widgets.ask_user,
         "ask_user(question, placeholder='')",
         "show a small input box; the user's answer arrives as the NEXT "
         "chat message — never wait for it inside the script"),
        (ui_widgets.side_panel,
         "side_panel(title, markdown)",
         "open/update a temporary side tab showing markdown (notes, BOM, "
         "results); same title = same tab refreshed"),
        (ui_widgets.close_panels,
         "close_panels()",
         "close all popups and side tabs created earlier"),
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
        (_run_stress_analysis,
         "run_stress_analysis(label=None, force_N=(0,0,-500), "
         "fixed='zmin', load='zmax')",
         "physics: train the elasticity PINN on the part, overlay a stress "
         "heatmap, report weak points. fixed/load pick faces: 'zmin' = "
         "bottom, 'zmax' = top, also xmin/xmax/ymin/ymax. force in Newtons"),
        (_remember,
         "remember(text)",
         "save a fact to long-term memory (materials, conventions, "
         "decisions the user states)"),
    ]


def make_namespace() -> SimpleNamespace:
    ns = SimpleNamespace()
    for fn, sig, _doc in _tool_table():
        # The name the model calls is whatever the signature says — parse
        # it from there so table and prompt can never disagree.
        setattr(ns, sig.split("(", 1)[0], fn)
    return ns


def render_docs() -> str:
    lines = []
    for _fn, sig, doc in _tool_table():
        lines.append(f"- piq_tools.{sig} — {doc}")
    return "\n".join(lines)
