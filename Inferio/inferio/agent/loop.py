"""loop.py — the agent brain. One chat turn flows like this:

    user text ──▶ build system prompt (master.md + tool docs
                  + scene description + RAG recipes)
              ──▶ LlmStreamWorker streams the reply (worker thread)
              ──▶ reply parsed for ```python blocks (main thread)
              ──▶ auto-run ON:  execute now, inside a transaction
                  auto-run OFF: panel shows a [Run] button per block
              ──▶ execution failed? traceback goes BACK to the model
                  ("self-correction"), at most S.max_fix_attempts() times
              ──▶ PINN results, when present, are injected the same way
                  and the model proposes a geometry fix (closed loop)

ChatController is a QObject so it can own signals; it lives on the main
thread. The ONLY thing that happens off-thread is the HTTP streaming —
see ui/workers.py for why that split is law.
"""

import re

from PySide6 import QtCore

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# How many past messages we keep sending. A 2B model with 8k context
# drowns in long histories; recent turns + fresh scene beat a full log.
_HISTORY_LIMIT = 12


def extract_code_blocks(text: str):
    return [m.strip() for m in _CODE_BLOCK_RE.findall(text) if m.strip()]


class ChatController(QtCore.QObject):
    # ---- signals the panel listens to (all delivered on main thread) ----
    assistant_chunk = QtCore.Signal(str)        # streaming text fragment
    assistant_done = QtCore.Signal(str)         # full reply text
    code_ready = QtCore.Signal(str)             # a block awaiting [Run]
    exec_finished = QtCore.Signal(bool, str)    # ok?, message for the chat
    status_changed = QtCore.Signal(str)         # one-line status bar text
    busy_changed = QtCore.Signal(bool)          # lock/unlock the input box

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []            # [{"role":..., "content":...}, ...]
        self._worker = None
        self._fix_attempts_left = 0
        self._pinn_job = None        # created lazily in request_pinn_analysis

    # ------------------------------------------------------------------
    # prompt assembly
    # ------------------------------------------------------------------
    def _system_message(self, user_text: str) -> dict:
        from inferio.llm import prompts
        from inferio.rag import retriever
        from inferio.tools import registry, scene_context

        template = prompts.load("master.md")
        # .replace, NOT .format: recipe code in the template may contain
        # literal braces, and .format() would explode on them.
        system = (template
                  .replace("{TOOL_DOCS}", registry.render_docs())
                  .replace("{SCENE}", scene_context.describe_document())
                  .replace("{SNIPPETS}", retriever.retrieve(user_text)))
        return {"role": "system", "content": system}

    def _client(self):
        from inferio.llm.client import LlamaClient
        from inferio.llm.server import MANAGER
        return LlamaClient(MANAGER.base_url())

    # ------------------------------------------------------------------
    # public API (called by the panel, on the main thread)
    # ------------------------------------------------------------------
    def send_user_message(self, text: str, images=None):
        """images: optional list of PNG bytes (screenshots, PDF pages)."""
        if images:
            content = [{"type": "text", "text": text}]
            from inferio.llm.client import image_part
            content += [image_part(png) for png in images]
        else:
            content = text
        self.history.append({"role": "user", "content": content})

        from inferio.settings import S
        self._fix_attempts_left = S.max_fix_attempts()
        self._generate(latest_user_text=text)

    def run_code(self, code: str):
        """Execute one code block. Called by [Run] buttons and by auto-run."""
        from inferio.tools.exec_python import run_freecad_python
        result = run_freecad_python(code)

        if result.ok:
            note = "Code executed OK."
            if result.stdout.strip():
                note += f"\noutput:\n{result.stdout.strip()[:800]}"
            self.exec_finished.emit(True, note)
            # Tell the model what happened so follow-ups have the context.
            self.history.append({"role": "user",
                                 "content": f"[executed OK] {note}"})
        else:
            self.exec_finished.emit(False, result.error)
            if self._fix_attempts_left > 0:
                # THE self-correction loop: hand the traceback back and
                # let the model repair its own code.
                self._fix_attempts_left -= 1
                self.history.append({
                    "role": "user",
                    "content": ("The code failed. Fix it and reply with one "
                                "corrected python block only.\n"
                                f"Error:\n{result.traceback}")})
                self.status_changed.emit(
                    f"error — asking model to fix "
                    f"({self._fix_attempts_left} attempts left)")
                self._generate(latest_user_text=result.error)
            else:
                self.status_changed.emit(
                    "giving up after repeated errors — see chat")

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def shutdown(self):
        from inferio.ui.workers import shutdown_worker
        shutdown_worker(self._worker)
        if self._pinn_job is not None:
            self._pinn_job.shutdown()

    # ------------------------------------------------------------------
    # generation plumbing
    # ------------------------------------------------------------------
    def _generate(self, latest_user_text: str):
        from inferio.settings import S
        from inferio.ui.workers import LlmStreamWorker, shutdown_worker

        shutdown_worker(self._worker)  # never two streams at once

        messages = ([self._system_message(latest_user_text)]
                    + self.history[-_HISTORY_LIMIT:])
        self._worker = LlmStreamWorker(
            self._client(), messages,
            temperature=S.temperature(), max_tokens=S.max_tokens(),
            parent=self)
        self._worker.chunk_received.connect(self.assistant_chunk)
        self._worker.finished_ok.connect(self._on_reply_done)
        self._worker.failed.connect(self._on_reply_failed)
        self.busy_changed.emit(True)
        self.status_changed.emit("thinking…")
        self._worker.start()

    def _on_reply_done(self, text: str):
        self.busy_changed.emit(False)
        self.status_changed.emit("ready")
        self.history.append({"role": "assistant", "content": text})
        self.assistant_done.emit(text)

        blocks = extract_code_blocks(text)
        if not blocks:
            return
        from inferio.settings import S
        if S.auto_run():
            for code in blocks:
                self.run_code(code)   # may recurse into _generate on error
        else:
            for code in blocks:
                self.code_ready.emit(code)  # panel renders [Run] cards

    def _on_reply_failed(self, message: str):
        self.busy_changed.emit(False)
        self.status_changed.emit(f"LLM error: {message}")

    # ------------------------------------------------------------------
    # physics: the PINN closed loop (phase 5 wiring lives in pinn_runner)
    # ------------------------------------------------------------------
    def request_pinn_analysis(self):
        from inferio.tools.pinn_runner import PinnJob
        if self._pinn_job is not None and self._pinn_job.is_running():
            self.status_changed.emit("a PINN job is already running")
            return
        self._pinn_job = PinnJob(parent=self)
        self._pinn_job.progress.connect(self.status_changed)
        self._pinn_job.finished_ok.connect(self._on_pinn_done)
        self._pinn_job.failed.connect(
            lambda msg: self.status_changed.emit(f"PINN failed: {msg}"))
        self._pinn_job.start()

    def _on_pinn_done(self, report: dict):
        """PINN finished: overlay the heatmap, then close the loop by
        telling the model about the weak points."""
        from inferio.settings import S
        from inferio.tools import heatmap_overlay

        try:
            heatmap_overlay.apply_from_files(report["field_file"])
        except Exception as exc:  # noqa: BLE001 — overlay is cosmetic
            self.status_changed.emit(f"heatmap overlay failed: {exc}")

        summary = heatmap_overlay.summarize_report(report)
        self.exec_finished.emit(True, summary)

        target = S.safety_factor_target()
        min_sf = report.get("min_safety_factor", 99)
        if min_sf < target:
            # Weak part → ask the model for a fix. With auto-run ON this
            # is the full autonomous loop from the pitch deck.
            self.send_user_message(
                f"The stress analysis found weak points:\n{summary}\n"
                f"Minimum safety factor {min_sf:.2f} is below the target "
                f"{target:.1f}. Modify the geometry to strengthen it "
                "(fillet the hot corner or add a gusset/rib). Reply with "
                "one python block.")
        else:
            self.status_changed.emit(
                f"part is strong enough (min SF {min_sf:.2f} ≥ {target:.1f})")
