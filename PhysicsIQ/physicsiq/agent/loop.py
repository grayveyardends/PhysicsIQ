"""loop.py — the agent brain. It runs in one of three modes.

CHAT (the default) — one turn:

    user text --> system prompt (master.md + tool docs + scene + RAG + memory)
              --> LlmStreamWorker streams the reply (worker thread)
              --> reply parsed for ```python blocks (main thread)
              --> auto-run ON:  execute now, inside a transaction
                  auto-run OFF: panel shows a [Run] button per block
              --> code failed?   traceback goes BACK to the model
                                 ("self-correction"), max S.max_fix_attempts()
              --> code worked?   screenshot -> the VISION model checks its
                                 own work (max S.max_visual_fixes() fixes)
              --> PINN results re-enter the same way (the closed loop)

PLAN -> STEP (for anything bigger than one part):

    plan mode ON --> the model THINKS and writes plans/<slug>/step-*.md.
                     No code runs. This is the one call worth spending cloud on.
    "run step N" --> a FRESH context: goal + one-line results of finished
                     steps + THIS step. The chat history is NOT carried in,
                     so step 7 costs the same context as step 1.

cloud CLOUD — picked in the model dropdown, costs a prompt per message:

    Choosing "cloud Claude sonnet" instead of a .gguf sets `cloud_model`, and the
    messages YOU send go to the subscription. But everything the agent does on
    its OWN — self-correction retries, visual checks, PINN fixes — calls
    _generate() without a `cloud` argument, which means _loop_uses_cloud():

        a local model is loaded  ->  loops run on it, and cost you nothing
        no local model at all    ->  loops fall back to the cloud, still capped
                                    (max_fix_attempts, max_visual_fixes)

    So the cheap way to work is: load a gguf AND pick cloud for the hard question.
    You get the good brain where you asked for it and free loops everywhere
    else. Nothing can silently run away with your plan.

ChatController is a QObject so it can own signals; it lives on the main
thread. The ONLY thing that happens off-thread is the model call — see
ui/workers.py for why that split is law.
"""

import re

from PySide6 import QtCore

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# How many past messages we keep sending. A small model with 8k context
# drowns in long histories; recent turns + fresh scene beat a full log.
# Inside a step run this caps the self-correction chatter, nothing else —
# the step's context starts empty every time.
_HISTORY_LIMIT = 12


def extract_code_blocks(text: str):
    return [m.strip() for m in _CODE_BLOCK_RE.findall(text) if m.strip()]


def _prose_summary(text: str, limit: int = 200) -> str:
    """The model's reply minus its code, squashed to one line — this is what
    a finished step contributes to every later step's context. Keep it SHORT:
    it is paid for on every subsequent step."""
    prose = _CODE_BLOCK_RE.sub("", text).strip()
    return " ".join(prose.split())[:limit]


class ChatController(QtCore.QObject):
    # signals the panel listens to (all delivered on main thread)
    assistant_chunk = QtCore.Signal(str)        # streaming answer fragment
    assistant_thinking = QtCore.Signal(str)     # reasoning-model thoughts
    assistant_done = QtCore.Signal(str)         # full reply text
    code_ready = QtCore.Signal(str)             # a block awaiting [Run]
    exec_finished = QtCore.Signal(bool, str)    # ok?, message for the chat
    status_changed = QtCore.Signal(str)         # one-line status bar text
    busy_changed = QtCore.Signal(bool)          # lock/unlock the input box
    stream_aborted = QtCore.Signal(str)         # generation died — reset the UI
    plan_changed = QtCore.Signal(object)        # plan dict, or None to hide
    command_result = QtCore.Signal(str)         # /slash command output

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []            # [{"role":..., "content":...}, ...]
        self.plan_mode = False       # ON = next message writes a plan, not code
        self.cloud_model = None      # a cloud alias from the dropdown, or None
        self.cloud_calls = 0         # what cloud has cost you this session
        self.cloud_cost_usd = 0.0

        self._worker = None
        self._client = None          # the client of the in-flight call
        self._mode = "chat"          # chat | plan | visual
        self._fix_attempts_left = 0
        # Two separate budgets, or the LAST fix would never be looked at:
        # N fixes are allowed, N+1 looks (the extra look confirms the final
        # fix and gives you a verdict on it).
        self._visual_fixes_left = 0
        self._visual_checks_left = 0
        self._queue = []             # code blocks waiting to run, in order
        self._request = ""           # what the user actually asked for
        self._pinn_job = None
        self._step = None            # {"plan": name, "n": int, "history": []}

    # context assembly — the point of the plan pipeline is what we DON'T
    # put in here
    @property
    def _active_history(self):
        """Chat mode writes to the chat history; a step run writes to the
        step's own scratch history, which is thrown away when the step ends.
        That throw-away is what keeps the context from growing."""
        return self.history if self._step is None else self._step["history"]

    def _system_message(self, user_text: str) -> dict:
        from physicsiq.llm import prompts
        from physicsiq.rag import retriever
        from physicsiq.tools import registry, scene_context

        template = prompts.load("master.md")
        # .replace, NOT .format: recipe code in the template may contain
        # literal braces, and .format() would explode on them.
        system = (template
                  .replace("{TOOL_DOCS}", registry.render_docs())
                  .replace("{SCENE}", scene_context.describe_document())
                  .replace("{SNIPPETS}", retriever.retrieve(user_text)))

        if self._step is not None:
            from physicsiq.agent import plan_store
            plan = plan_store.load(self._step["plan"])
            if plan is not None:
                system += "\n\n" + plan_store.render_for_prompt(
                    plan, self._step["n"])

        from physicsiq.memory import store
        notes = store.render_for_prompt()
        if notes:
            system += f"\n\n## Remembered notes\n{notes}"
        return {"role": "system", "content": system}

    def _planner_system_message(self) -> dict:
        from physicsiq.llm import prompts
        from physicsiq.tools import scene_context
        return {"role": "system",
                "content": prompts.load("planner.md").replace(
                    "{SCENE}", scene_context.describe_document())}

    # public API (called by the panel, on the main thread)
    def send_user_message(self, text: str, images=None):
        """images: optional list of either raw PNG bytes (screenshots, PDF
        pages) or ready-made content-part dicts (attachments)."""
        from physicsiq.agent import chat_commands
        if chat_commands.is_command(text):
            out = chat_commands.handle(text, self)   # never reaches the model
            if out:
                self.command_result.emit(out)
            return

        # A message YOU send goes to whichever brain the dropdown says.
        cloud = self.cloud_model is not None

        if self.plan_mode:
            self.make_plan(text, cloud=cloud)
            return
        self._post_turn(text, images, cloud=cloud)

    def _agent_message(self, text: str):
        """A turn the AGENT started (the PINN fix loop). cloud=None routes it
        through _loop_uses_cloud() — i.e. it runs on the local model whenever
        one is loaded. It is NOT a message you sent, so it must not bill your
        plan just because cloud is selected in the dropdown."""
        self._post_turn(text, None, cloud=None)

    def _post_turn(self, text: str, images, cloud):
        if images:
            content = [{"type": "text", "text": text}]
            from physicsiq.llm.client import image_part
            content += [img if isinstance(img, dict) else image_part(img)
                        for img in images]
        else:
            content = text
        self._active_history.append({"role": "user", "content": content})

        from physicsiq.settings import S
        self._request = text
        self._fix_attempts_left = S.max_fix_attempts()
        self._visual_fixes_left = S.max_visual_fixes()
        self._visual_checks_left = self._visual_fixes_left + 1
        self._mode = "chat"
        self._generate(latest_user_text=text, cloud=cloud)

    # planning
    def make_plan(self, request: str, cloud: bool = False):
        """Think hard ONCE and write the plan to disk. No code runs here."""
        self._step = None
        self._mode = "plan"
        self._request = request
        self.history = [{"role": "user", "content": request}]
        self._generate(latest_user_text=request, cloud=cloud)

    def run_step(self, n: int, plan_name: str = None):
        """Execute ONE step of the plan in a fresh, isolated context."""
        from physicsiq.agent import plan_store
        from physicsiq.settings import S

        plan = plan_store.load(plan_name)
        if plan is None:
            self.status_changed.emit("no plan yet — tick plan mode and "
                                     "describe the job first")
            return
        step = next((s for s in plan["steps"] if s["n"] == n), None)
        if step is None:
            self.status_changed.emit(f"plan has no step {n}")
            return

        # The step's instructions ride in the SYSTEM message (assembled from
        # the files on disk by _system_message), so editing step-N.md in your
        # editor changes what this run does.
        self._mode = "chat"
        self._step = {"plan": plan["name"], "n": n, "history": []}
        self._step["history"].append({
            "role": "user",
            "content": f"Do step {n} ({step['title']}) now. "
                       "Reply with one python block."})
        self._request = f"step {n}: {step['title']} — {step['body']}"
        self._fix_attempts_left = S.max_fix_attempts()
        self._visual_fixes_left = S.max_visual_fixes()
        self._visual_checks_left = self._visual_fixes_left + 1
        self.status_changed.emit(f"running step {n}: {step['title']}")
        self._generate(latest_user_text=self._request)   # LOCAL, always

    def run_next_step(self, plan_name: str = None):
        from physicsiq.agent import plan_store
        plan = plan_store.load(plan_name)
        if plan is None:
            self.status_changed.emit("no plan to run")
            return
        n = plan_store.next_todo(plan)
        if not n:
            self.status_changed.emit(f"plan '{plan['title']}' is complete")
            return
        self.run_step(n, plan["name"])

    def abandon_plan(self):
        """/clear-plan: stop working on it. The .md files stay on disk."""
        from physicsiq.agent import plan_store
        from physicsiq.ui.workers import shutdown_worker
        if self._step is not None:
            shutdown_worker(self._worker)   # a step was mid-flight
            self._step = None
            self.busy_changed.emit(False)
        plan_store.clear_active()
        self.plan_changed.emit(None)        # panel removes the card

    def adopt_plan(self, name: str):
        """/plan use <name>: make an old plan active again."""
        from physicsiq.agent import plan_store
        plan_store.set_active(name)
        self.plan_changed.emit(plan_store.load(name))

    def _finish_step(self, ok: bool, result: str):
        """Close the current step: write its ONE-LINE result to state.json and
        drop its scratch history. Nothing from it survives into the next step
        except that line."""
        from physicsiq.agent import plan_store
        if self._step is None:
            return
        plan_name, n = self._step["plan"], self._step["n"]
        self._step = None            # scratch history dies here, on purpose
        plan_store.set_status(
            plan_name, n, plan_store.DONE if ok else plan_store.FAILED, result)
        plan = plan_store.load(plan_name)
        if plan is None:             # cleared underneath us
            return
        self.plan_changed.emit(plan)
        if ok:
            nxt = plan_store.next_todo(plan)
            self.status_changed.emit(
                f"step {n} done — next: step {nxt}" if nxt
                else f"plan '{plan['title']}' complete")
        else:
            self.status_changed.emit(f"step {n} FAILED — fix it and re-run")

    # running code: a QUEUE, so several blocks in one reply run in order and
    # a failure stops the rest (a failure starts a self-correction stream —
    # letting block 2 run now would race it)
    def run_code(self, code: str):
        """Execute one block. Called by [Run] buttons and by auto-run."""
        self._queue = [code]
        self._drain_queue()

    def _drain_queue(self):
        while self._queue:
            if not self._exec_one(self._queue.pop(0)):
                self._queue.clear()   # a retry stream is now in flight
                return
        self._after_code_settled()

    def _exec_one(self, code: str) -> bool:
        from physicsiq.tools.exec_python import run_freecad_python
        result = run_freecad_python(code)

        if result.ok:
            note = "Code executed OK."
            if result.stdout.strip():
                note += f"\noutput:\n{result.stdout.strip()[:800]}"
            self.exec_finished.emit(True, note)
            self._active_history.append({"role": "user",
                                         "content": f"[executed OK] {note}"})
            return True

        self.exec_finished.emit(False, result.error)
        if self._fix_attempts_left > 0:
            # THE self-correction loop: hand the traceback back and let the
            # model repair its own code. Inside a step this stays in the
            # step's scratch history — the chat never sees the mess.
            self._fix_attempts_left -= 1
            self._active_history.append({
                "role": "user",
                "content": ("The code failed. Fix it and reply with one "
                            "corrected python block only.\n"
                            f"Error:\n{result.traceback}")})
            self.status_changed.emit(
                f"error — asking model to fix "
                f"({self._fix_attempts_left} attempts left)")
            self._mode = "chat"
            self._generate(latest_user_text=result.error)   # LOCAL, always
        else:
            self.status_changed.emit("giving up after repeated errors — see chat")
            self._finish_step(False, f"failed: {result.error}")
        return False

    def _after_code_settled(self):
        """Everything the model asked for has run. Now LOOK at it."""
        if self._should_visual_check():
            self._start_visual_check()
        elif self._step is not None:
            self._finish_step(True, self._step.get("summary", "done"))

    # the visual check: the model looks at what it built (LOCAL model only —
    # this fires on every code run, far too often to bill a cloud plan for)
    def _should_visual_check(self) -> bool:
        from physicsiq.llm.server import MANAGER
        from physicsiq.settings import S
        if not S.visual_check() or self._visual_checks_left <= 0:
            return False
        if self._loop_uses_cloud():
            return True                 # Claude can see; it costs a prompt
        # Local: only if llama-server actually loaded an mmproj. has_vision()
        # reports what we really passed to --mmproj, not what a filename hinted.
        return MANAGER.is_running() and MANAGER.has_vision()

    def _start_visual_check(self):
        from physicsiq.llm import prompts
        from physicsiq.llm.client import image_part
        from physicsiq.tools import scene_context, screenshot

        self._visual_checks_left -= 1
        try:
            png = screenshot.grab_3d_view()     # main thread — touches the view
        except Exception as exc:  # noqa: BLE001 — no 3D view is not an error
            self.status_changed.emit(f"visual check skipped: {exc}")
            if self._step is not None:
                self._finish_step(True, self._step.get("summary", "done"))
            return

        system = (prompts.load("visual_check.md")
                  .replace("{REQUEST}", self._request or "(not recorded)")
                  .replace("{SCENE}", scene_context.describe_document()))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Here is the 3D viewport after your code ran."},
                image_part(png)]},
        ]
        self._mode = "visual"
        self.status_changed.emit("checking the result against the request…")
        self._generate(latest_user_text="", messages=messages)   # LOCAL

    def _on_visual_verdict(self, text: str):
        self._mode = "chat"
        blocks = extract_code_blocks(text)
        says_fix = "VERDICT: FIX" in text.upper()

        if says_fix and blocks and self._visual_fixes_left > 0:
            self._visual_fixes_left -= 1
            self.status_changed.emit(
                f"the model spotted a problem — fixing "
                f"({self._visual_fixes_left} visual fixes left)")
            # Run the correction. When it settles we look AGAIN, bounded by
            # _visual_fixes_left — so two models can't argue forever.
            self._queue = [blocks[0]]
            self._drain_queue()
            return

        if says_fix:
            # It complained but gave us nothing to run (or we're out of
            # rounds). Say so plainly rather than pretending it passed.
            self.status_changed.emit(
                "the model still isn't happy — have a look yourself")
            if self._step is not None:
                self._finish_step(True, self._step.get("summary", "done"))
            return

        self.status_changed.emit("✓ looks right")
        if self._step is not None:
            self._finish_step(True, self._step.get("summary", "done"))

    # generation plumbing
    def _loop_uses_cloud(self) -> bool:
        """Which brain does the AGENT use when it acts on its own (retries,
        visual checks, PINN fixes)? A loaded local model, whenever there is
        one — those calls repeat and must stay free. Only when you never
        loaded a gguf at all do they fall back to the cloud brain, and even then
        the caps (max_fix_attempts, max_visual_fixes) bound the damage."""
        from physicsiq.llm.server import MANAGER
        if MANAGER.is_running():
            return False
        return self.cloud_model is not None

    def _generate(self, latest_user_text: str, cloud: bool = None,
                  messages=None):
        """cloud=None means "the agent is acting on its own" -> route through
        _loop_uses_cloud(). Only send_user_message/make_plan pass it
        explicitly, because only YOUR messages are meant to spend a prompt."""
        from physicsiq.llm.client import get_client
        from physicsiq.settings import S
        from physicsiq.ui.workers import LlmStreamWorker, shutdown_worker

        shutdown_worker(self._worker)  # never two streams at once
        if cloud is None:
            cloud = self._loop_uses_cloud()

        if messages is None:
            system = (self._planner_system_message() if self._mode == "plan"
                      else self._system_message(latest_user_text))
            messages = [system] + self._active_history[-_HISTORY_LIMIT:]

        self._client = get_client(cloud=cloud, model=self.cloud_model)
        if cloud:
            self.cloud_calls += 1
            self.status_changed.emit(
                f"cloud {self.cloud_model} — {self.cloud_calls} prompt(s) "
                "this session")

        self._worker = LlmStreamWorker(
            self._client, messages,
            temperature=S.temperature(), max_tokens=S.max_tokens(),
            parent=self)
        self._worker.chunk_received.connect(self.assistant_chunk)
        self._worker.thinking_received.connect(self.assistant_thinking)
        self._worker.finished_ok.connect(self._on_reply_done)
        self._worker.failed.connect(self._on_reply_failed)
        self.busy_changed.emit(True)
        if not cloud:
            self.status_changed.emit(
                "planning (the model is thinking — this is the slow part)…"
                if self._mode == "plan" else "thinking…")
        self._worker.start()

    def _note_cloud_cost(self):
        cost = getattr(self._client, "last_cost_usd", None)
        if cost:
            self.cloud_cost_usd += float(cost)

    def _on_reply_done(self, text: str):
        self._note_cloud_cost()
        self.busy_changed.emit(False)
        self.assistant_done.emit(text)

        if self._mode == "plan":
            self._mode = "chat"
            self._save_plan(text)
            return
        if self._mode == "visual":
            self._on_visual_verdict(text)
            return

        self.status_changed.emit("ready")
        self._active_history.append({"role": "assistant", "content": text})

        blocks = extract_code_blocks(text)
        if not blocks:
            if self._step is not None:
                self._finish_step(True, _prose_summary(text))  # info-only step
            return
        if self._step is not None:
            # Remember the prose NOW: it becomes the step's one-line result
            # once the code has run (the exec path can't see the reply text).
            self._step["summary"] = _prose_summary(text)

        from physicsiq.settings import S
        if S.auto_run():
            self._queue = list(blocks)
            self._drain_queue()
        else:
            for code in blocks:
                self.code_ready.emit(code)  # panel renders [Run] cards

    def _save_plan(self, text: str):
        from physicsiq.agent import plan_store
        try:
            plan = plan_store.save_plan(text)
        except ValueError as exc:
            self.status_changed.emit(
                f"could not read a plan out of that reply ({exc}) — "
                "the raw answer is in the chat")
            return
        self.plan_mode = False       # one plan per tick; back to chat
        self.plan_changed.emit(plan)
        self.status_changed.emit(
            f"plan saved: {len(plan['steps'])} steps in plans/{plan['name']}/")

    def _on_reply_failed(self, message: str):
        was = self._mode
        self._mode = "chat"
        self.busy_changed.emit(False)
        self.status_changed.emit(f"LLM error: {message}")
        # Tell the panel to tear down the half-filled bubble and stop its
        # repaint timer — without this it spins forever on an empty reply.
        self.stream_aborted.emit(message)
        if was == "visual":
            # A failed screenshot check is not a failed step: the geometry is
            # already built and the code ran fine. Don't punish it.
            if self._step is not None:
                self._finish_step(True, self._step.get("summary", "done"))
            return
        if self._step is not None:
            self._finish_step(False, f"model error: {message}")

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def shutdown(self):
        from physicsiq.ui.workers import shutdown_worker
        shutdown_worker(self._worker)
        if self._pinn_job is not None:
            self._pinn_job.shutdown()

    # physics: the PINN closed loop (wiring lives in tools/pinn_runner.py)
    def request_pinn_analysis(self, target_label=None,
                              force_N=(0.0, 0.0, -500.0),
                              fixed_rule="zmin", load_rule="zmax"):
        from physicsiq.tools.pinn_runner import PinnJob
        if self._pinn_job is not None and self._pinn_job.is_running():
            self.status_changed.emit("a PINN job is already running")
            return
        self._pinn_job = PinnJob(parent=self, target_label=target_label,
                                 force_N=force_N, fixed_rule=fixed_rule,
                                 load_rule=load_rule)
        self._pinn_job.progress.connect(self.status_changed)
        self._pinn_job.live_field.connect(self._on_pinn_live_field)
        self._pinn_job.finished_ok.connect(self._on_pinn_done)
        self._pinn_job.failed.connect(
            lambda msg: self.status_changed.emit(f"PINN failed: {msg}"))
        self._pinn_job.start()

    def _on_pinn_live_field(self, npz_path: str):
        """Training snapshot arrived: repaint the heatmap while the PINN
        is still running. First snapshot builds the overlay (so the user
        sees color + the pulsing marker seconds after start), later ones
        just recolor it. Cosmetic — never let it kill the run."""
        from physicsiq.tools import heatmap_overlay
        try:
            heatmap_overlay.update_from_files(
                npz_path,
                target_label=getattr(self._pinn_job,
                                     "target_object_label", None))
        except Exception as exc:  # noqa: BLE001
            self.status_changed.emit(f"live heatmap skipped: {exc}")

    def _on_pinn_done(self, report: dict):
        """PINN finished: overlay the heatmap, then close the loop by telling
        the model about the weak points."""
        from physicsiq.settings import S
        from physicsiq.tools import heatmap_overlay

        try:
            # final field replaces whatever live snapshot was showing
            heatmap_overlay.update_from_files(
                report["field_file"],
                target_label=report.get("target_label"))
        except Exception as exc:  # noqa: BLE001 — overlay is cosmetic
            self.status_changed.emit(f"heatmap overlay failed: {exc}")

        summary = heatmap_overlay.summarize_report(report)
        self.exec_finished.emit(True, summary)

        target = S.safety_factor_target()
        min_sf = report.get("min_safety_factor", 99)
        if min_sf < target:
            # Weak part -> ask the model for a fix. With auto-run ON this is
            # the full autonomous loop from the pitch deck — and _agent_message
            # keeps it on the local model, because a loop must never bill you.
            self._agent_message(
                f"The stress analysis found weak points:\n{summary}\n"
                f"Minimum safety factor {min_sf:.2f} is below the target "
                f"{target:.1f}. Modify the geometry to strengthen it (fillet "
                "the hot corner or add a gusset/rib). Reply with one python "
                "block.")
        else:
            self.status_changed.emit(
                f"part is strong enough (min SF {min_sf:.2f} ≥ {target:.1f})")
            # Analysis loop is DONE: stop the pulsing, let the heatmap linger
            # a few seconds for reading, then clean the view up — a stale
            # marker ball outstays its welcome fast.
            heatmap_overlay.stop_pulse()
            QtCore.QTimer.singleShot(12000, heatmap_overlay.clear_overlay)
