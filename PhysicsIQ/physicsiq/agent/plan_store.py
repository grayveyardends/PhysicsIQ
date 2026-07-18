"""plan_store.py — plans live on DISK as .md files, not in the context window.

The problem this solves: a reasoning model (DeepSeek-R1) is great at
figuring out HOW to build a part, and terrible at doing it in one shot —
it spends its whole token budget thinking. And a long chat history where
everything happened is exactly what a small model drowns in.

So we split the two, the way a human engineer does:

    PLAN once   -> the model thinks hard, writes a plan to files
    RUN a step  -> a FRESH, tiny context: the plan's goal, one-line
                  summaries of the finished steps, and the ONE step to do

Nothing else from the conversation is carried in. Step 7 costs the same
context as step 1. That is the whole idea.

On disk (plans/<slug>/):

    plan.md        the goal + an index of every step   (read this)
    step-1.md      what step 1 must do                 (EDIT THIS — it is
    step-2.md      ...                                  the source of truth;
    state.json     status + one-line result per step    the model re-reads
                                                        it at run time)

Files, not a database: you can open a step in vim, fix the model's dumb
idea, and run it. That is the point of plain markdown.
"""

import json
import os
import re
import time

from physicsiq import REPO_DIR

PLANS_DIR = os.path.join(REPO_DIR, "plans")

# "## Step 3 — drill the bolt holes"  (also tolerates ':' or '-' or nothing)
_STEP_RE = re.compile(r"^##\s*step\s*(\d+)\s*[-—:.]?\s*(.*)$", re.I)
_GOAL_RE = re.compile(r"^#\s*(?:goal|plan)?\s*[:—-]?\s*(.*)$", re.I)

TODO, DONE, FAILED = "todo", "done", "failed"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "plan"


# parsing the model's markdown into steps
def parse_plan(markdown: str):
    """(goal, [(n, title, body), ...]) from the planner's reply.

    We are deliberately forgiving: models miscount, skip numbers and
    wander off format. We take the headings in the order they appear and
    RE-NUMBER them 1..N, so 'Step 2, Step 2, Step 5' still becomes a
    sane three-step plan.
    """
    goal_lines, steps = [], []
    current = None
    for line in markdown.splitlines():
        m = _STEP_RE.match(line.strip())
        if m:
            current = {"title": m.group(2).strip() or "step", "body": []}
            steps.append(current)
            continue
        if current is not None:
            current["body"].append(line)
        else:
            g = _GOAL_RE.match(line.strip())
            goal_lines.append(g.group(1) if g else line)

    goal = "\n".join(goal_lines).strip()
    out = [(i, s["title"], "\n".join(s["body"]).strip())
           for i, s in enumerate(steps, 1)]
    return goal, out


# writing / reading
def save_plan(markdown: str, title: str = None) -> dict:
    """Write a fresh plan to disk. Returns the loaded plan dict.
    Raises ValueError if the model produced no recognizable steps —
    the caller then shows the raw reply instead of pretending."""
    goal, steps = parse_plan(markdown)
    if not steps:
        raise ValueError("no '## Step N' headings found in the plan")

    title = title or (goal.splitlines()[0] if goal else "plan")
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(title)}"
    plan_dir = os.path.join(PLANS_DIR, name)
    os.makedirs(plan_dir, exist_ok=True)

    for n, step_title, body in steps:
        with open(os.path.join(plan_dir, f"step-{n}.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"# Step {n} — {step_title}\n\n{body}\n")

    index = [f"# {title}", "", goal, "", "## Steps", ""]
    index += [f"{n}. **{t}**  -> `step-{n}.md`" for n, t, _ in steps]
    with open(os.path.join(plan_dir, "plan.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(index) + "\n")

    _write_state(plan_dir, {
        "title": title, "goal": goal, "created": time.time(),
        "steps": {str(n): {"status": TODO, "result": ""} for n, _, _ in steps},
    })
    _set_active(name)
    return load(name)


def _state_path(plan_dir):
    return os.path.join(plan_dir, "state.json")


def _write_state(plan_dir, state):
    with open(_state_path(plan_dir), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _read_state(plan_dir):
    with open(_state_path(plan_dir), encoding="utf-8") as fh:
        return json.load(fh)


# The "active" plan is just a pointer file, so `next step` needs no argument
# and survives a FreeCAD restart.
def _active_path():
    return os.path.join(PLANS_DIR, "ACTIVE")


def _set_active(name):
    os.makedirs(PLANS_DIR, exist_ok=True)
    with open(_active_path(), "w", encoding="utf-8") as fh:
        fh.write(name)


set_active = _set_active     # public name — /plan use <name>


def clear_active():
    """Stop working on the current plan. Deletes ONLY the ACTIVE pointer —
    every step-N.md and its results stay exactly where they are. A plan can
    cost a real cloud prompt to produce; we do not throw that away because
    someone typed /clear-plan. `/plan use <name>` brings it straight back."""
    try:
        os.remove(_active_path())
    except OSError:
        pass                 # no active plan — clearing is already true


def active_name():
    try:
        with open(_active_path(), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def list_plans():
    if not os.path.isdir(PLANS_DIR):
        return []
    return sorted((d for d in os.listdir(PLANS_DIR)
                   if os.path.isdir(os.path.join(PLANS_DIR, d))),
                  reverse=True)


def load(name: str = None):
    """Load a plan. name=None -> the active one. Returns None if there is
    none. Step BODIES are re-read from step-N.md every time, so editing a
    step file in your editor changes what the next run does."""
    name = name or active_name()
    if not name:
        return None
    plan_dir = os.path.join(PLANS_DIR, name)
    if not os.path.isfile(_state_path(plan_dir)):
        return None
    state = _read_state(plan_dir)

    steps = []
    for key in sorted(state["steps"], key=int):
        n = int(key)
        path = os.path.join(plan_dir, f"step-{n}.md")
        try:
            with open(path, encoding="utf-8") as fh:
                body = fh.read()
        except OSError:
            body = ""            # file deleted by hand — step becomes a no-op
        # strip the "# Step N — title" heading back off; keep the title
        lines = body.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else f"step {n}"
        title = re.sub(r"^step\s*\d+\s*[-—:.]?\s*", "", title, flags=re.I)
        steps.append({
            "n": n, "title": title or f"step {n}",
            "body": "\n".join(lines[1:]).strip(),
            "status": state["steps"][key]["status"],
            "result": state["steps"][key]["result"],
        })

    return {"name": name, "dir": plan_dir, "title": state["title"],
            "goal": state.get("goal", ""), "steps": steps}


def set_status(name: str, n: int, status: str, result: str = ""):
    plan_dir = os.path.join(PLANS_DIR, name)
    state = _read_state(plan_dir)
    state["steps"][str(n)] = {"status": status, "result": result[:400]}
    _write_state(plan_dir, state)


def next_todo(plan) -> int:
    """Number of the first unfinished step, or 0 when the plan is done."""
    for s in plan["steps"]:
        if s["status"] != DONE:
            return s["n"]
    return 0


# the context the model actually sees when running a step
def render_for_prompt(plan, current_n: int) -> str:
    """The ENTIRE plan-related context for one step run — kept deliberately
    tiny. Finished steps contribute ONE line each (their result), never
    their transcript. This is what stops step 9 from costing 9 steps of
    tokens."""
    lines = [f"## Active plan: {plan['title']}"]
    if plan["goal"]:
        lines.append(f"Goal: {plan['goal']}")
    lines.append("")
    for s in plan["steps"]:
        if s["n"] == current_n:
            mark = "▶ NOW"
        elif s["status"] == DONE:
            mark = "done"
        elif s["status"] == FAILED:
            mark = "FAILED"
        else:
            mark = "todo"
        line = f"  {s['n']}. [{mark}] {s['title']}"
        if s["status"] == DONE and s["result"]:
            line += f" -> {s['result']}"
        lines.append(line)

    current = next((s for s in plan["steps"] if s["n"] == current_n), None)
    if current:
        lines += [
            "",
            f"### Your task right now: step {current_n} — {current['title']}",
            current["body"],
            "",
            "Do ONLY this step. The earlier steps are already built (see "
            "their results above and the scene description). Do not redo "
            "them, and do not run ahead to later steps.",
        ]
    return "\n".join(lines)
