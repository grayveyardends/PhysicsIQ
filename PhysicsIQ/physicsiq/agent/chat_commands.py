"""chat_commands.py — slash commands you type in the chat box.

These never reach the model: `send_user_message` sees a leading "/" and hands
the line here instead of spending a turn (or, if cloud is ticked, a cloud prompt)
on it. Same idea as Claude Code's own slash commands.

    /help                what you're reading
    /plan                show the active plan and where it's up to
    /plan list           every plan on disk
    /plan use <name>     make an old plan active again
    /clear-plan          stop working on the plan (the .md FILES ARE KEPT)
    /next                run the next unfinished step
    /run 3               run step 3
    /cloud               what the cloud button will call, and what it has cost

Design note: a command returns TEXT, which the panel prints as a system
message. Commands that change state do it through the controller, so the plan
card and the status bar update through the same signals as everything else.

NOT the same thing as physicsiq/commands.py — that file is FreeCAD's toolbar
buttons. This one is the chat's own little command line.
"""

HELP = """**Chat commands**

| command | what it does |
|---|---|
| `/plan` | show the active plan and its progress |
| `/plan list` | list every plan saved on disk |
| `/plan use <name>` | make an older plan active again |
| `/clear-plan` | stop working on the plan — **the .md files are kept** |
| `/next` | run the next unfinished step |
| `/run <n>` | run step n |
| `/cloud` | which brain you're on, and what it has cost this session |
| `/help` | this table |

Tick **plan mode** and describe a whole job to get a plan. Pick a **cloud Claude**
entry in the model dropdown to have Claude write it instead of the local
model — that's the combo worth paying for: one prompt buys a whole plan, and
the steps then run on your GPU for free."""


def is_command(text: str) -> bool:
    return text.strip().startswith("/")


def handle(text: str, controller) -> str:
    """Run one command. Returns the text to show in the chat.
    Never raises: a typo in the chat box must not take the panel down."""
    from physicsiq.agent import plan_store

    parts = text.strip().split()
    cmd, args = parts[0].lower().lstrip("/"), parts[1:]

    try:
        if cmd in ("help", "?"):
            return HELP

        if cmd in ("clear-plan", "clearplan"):
            return _clear_plan(controller, plan_store)

        if cmd == "plan":
            if not args:
                return _show_plan(plan_store)
            if args[0] == "list":
                return _list_plans(plan_store)
            if args[0] == "use" and len(args) > 1:
                return _use_plan(controller, plan_store, args[1])
            if args[0] == "clear":
                return _clear_plan(controller, plan_store)
            return f"unknown: `/plan {args[0]}` — try `/help`"

        if cmd == "next":
            controller.run_next_step()
            return ""          # the step's own status/stream says the rest

        if cmd == "run":
            if not args or not args[0].isdigit():
                return "usage: `/run <step number>`, e.g. `/run 2`"
            controller.run_step(int(args[0]))
            return ""

        if cmd == "cloud":
            return _cloud_status(controller)

        return f"unknown command `/{cmd}` — try `/help`"
    except Exception as exc:  # noqa: BLE001 — a chat typo is not a crash
        return f"command failed: {type(exc).__name__}: {exc}"


def _clear_plan(controller, plan_store) -> str:
    plan = plan_store.load()
    if plan is None:
        return "no active plan."
    controller.abandon_plan()          # cancels an in-flight step, hides card
    return (f"✓ cleared **{plan['title']}** — no longer the active plan.\n\n"
            f"Nothing was deleted: `plans/{plan['name']}/` still holds "
            f"{len(plan['steps'])} step file(s). Bring it back with "
            f"`/plan use {plan['name']}`.")


def _show_plan(plan_store) -> str:
    plan = plan_store.load()
    if plan is None:
        return ("no active plan. Tick **plan mode**, describe the job, and "
                "the model will write one.")
    lines = [f"**{plan['title']}**  (`plans/{plan['name']}/`)"]
    for s in plan["steps"]:
        mark = {"done": "✓", "failed": "✗"}.get(s["status"], "○")
        line = f"{mark} **{s['n']}.** {s['title']}"
        if s["result"]:
            line += f" — *{s['result']}*"
        lines.append(line)
    nxt = plan_store.next_todo(plan)
    lines.append(f"\nNext: step {nxt} (`/next`)" if nxt else "\nAll steps done.")
    return "\n".join(lines)


def _list_plans(plan_store) -> str:
    names = plan_store.list_plans()
    if not names:
        return "no plans on disk yet."
    active = plan_store.active_name()
    lines = ["**Plans on disk** (newest first):"]
    for name in names:
        lines.append(f"- `{name}`" + ("  <- active" if name == active else ""))
    lines.append("\n`/plan use <name>` to switch.")
    return "\n".join(lines)


def _use_plan(controller, plan_store, name: str) -> str:
    matches = [n for n in plan_store.list_plans() if name in n]
    if not matches:
        return f"no plan matching `{name}` — try `/plan list`."
    if len(matches) > 1 and name not in matches:
        return ("that matches several plans:\n"
                + "\n".join(f"- `{m}`" for m in matches))
    chosen = name if name in matches else matches[0]
    controller.adopt_plan(chosen)
    plan = plan_store.load(chosen)
    return f"✓ active plan is now **{plan['title']}**.\n\n" + _show_plan(plan_store)


def _cloud_status(controller) -> str:
    from physicsiq.llm.server import MANAGER
    from physicsiq.settings import S

    spent, cost = controller.cloud_calls, controller.cloud_cost_usd
    picked = controller.cloud_model
    lines = [
        f"Brain: **{'cloud Claude ' + picked if picked else 'local gguf'}**"
        + (f"  (`{S.cloud_bin()}`)" if picked else ""),
        f"Spent **{spent}** cloud prompt(s) this session"
        + (f" (~${cost:.3f})" if cost else "") + ".",
        "",
    ]
    if picked and MANAGER.is_running():
        lines.append(
            "A local model is ALSO loaded, so retries, the visual check and "
            "the PINN loop run on it — free. Only the messages you send cost "
            "a prompt. This is the cheap way to work.")
    elif picked:
        lines.append(
            "warning: no local model is loaded, so the agent's own retries and "
            "visual checks fall back to cloud too (still capped at "
            f"{S.max_fix_attempts()} retries and {S.max_visual_fixes()} visual "
            "fixes). Press **Load** on a gguf to make those free.")
    else:
        lines.append(
            "Pick a **cloud Claude** entry in the model dropdown to send your "
            "messages to Claude instead. Available: "
            + ", ".join(f"`{m}`" for m in S.cloud_models()) + ".")
    return "\n".join(lines)
