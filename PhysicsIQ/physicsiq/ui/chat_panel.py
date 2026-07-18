"""chat_panel.py — the dock widget you actually talk to.

Layout, top to bottom:

    [model dropdown] [Load] [● status]
    +--------------------------------+
    |  scrolling chat history        |
    |  (message bubbles + code cards)|
    +--------------------------------+
    [ multi-line input            ]
    [x] auto-run   [Send] [Stop]

All the intelligence lives in agent/loop.ChatController — this file is
deliberately dumb: render text, forward clicks. If you want to change how
the agent BEHAVES, edit agent/loop.py, not here.
"""

from PySide6 import QtCore, QtGui, QtWidgets

_MONO = QtGui.QFont("monospace")
_MONO.setStyleHint(QtGui.QFont.Monospace)


class MessageWidget(QtWidgets.QFrame):
    """One chat bubble. Markdown-rendered; assistant bubbles re-render
    while streaming (throttled by the panel's QTimer, not per-chunk —
    re-laying-out a QLabel 40x per second would stutter the 3D view)."""

    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        who = {"user": "You", "assistant": "PhysicsIQ", "system": "System"}
        header = QtWidgets.QLabel(f"<b>{who.get(role, role)}</b>")
        self.body = QtWidgets.QLabel()
        self.body.setTextFormat(QtCore.Qt.MarkdownText)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)  # let users copy answers
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.addWidget(header)
        lay.addWidget(self.body)
        self.set_text(text)

    def set_text(self, text: str):
        self.body.setText(text)


class CodeCard(QtWidgets.QFrame):
    """A generated python block with [Run] / [Copy] buttons — the
    'confirm before execute' half of our safety story."""

    run_requested = QtCore.Signal(str)

    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self.code = code
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        view = QtWidgets.QPlainTextEdit(code)
        view.setReadOnly(True)
        view.setFont(_MONO)
        # Size the card to the code (within reason) instead of a tiny scrollbox.
        rows = min(code.count("\n") + 2, 18)
        view.setFixedHeight(rows * view.fontMetrics().height() + 12)

        run_btn = QtWidgets.QPushButton("▶ Run")
        run_btn.setToolTip("Execute in FreeCAD (undoable with Ctrl+Z)")
        run_btn.clicked.connect(lambda: self.run_requested.emit(self.code))
        copy_btn = QtWidgets.QPushButton("Copy")
        copy_btn.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(self.code))

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(run_btn)
        btns.addWidget(copy_btn)
        btns.addStretch()

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.addWidget(view)
        lay.addLayout(btns)


class PlanCard(QtWidgets.QFrame):
    """The plan, as a live checklist. One [▶] per step so you can run them
    in order — or out of order, or re-run one you edited in your editor.
    The plan itself lives in plans/<slug>/*.md; this only reflects it."""

    run_step_requested = QtCore.Signal(int)
    run_next_requested = QtCore.Signal()
    clear_requested = QtCore.Signal()

    _ICON = {"done": "✓", "failed": "✗", "todo": "○"}

    def __init__(self, plan: dict, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(8, 6, 8, 6)
        self.set_plan(plan)

    @staticmethod
    def _wipe(layout):
        """Empty a layout, NESTED LAYOUTS AND ALL. takeAt() hands back plain
        layout items: for a row (a QHBoxLayout) item.widget() is None, so the
        naive version leaves its buttons alive and floating."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                PlanCard._wipe(item.layout())
                item.layout().deleteLater()

    def set_plan(self, plan: dict):
        self._wipe(self.lay)          # rebuild in place: the card keeps its
                                      # position in the chat history

        done = sum(1 for s in plan["steps"] if s["status"] == "done")
        head = QtWidgets.QLabel(
            f"<b>Plan: {plan['title']}</b>  "
            f"<i>({done}/{len(plan['steps'])} done · plans/{plan['name']}/)</i>")
        head.setWordWrap(True)
        clear = QtWidgets.QPushButton("✕")
        clear.setFixedWidth(30)
        clear.setToolTip("Stop working on this plan (same as /clear-plan).\n"
                         "The step .md files are KEPT — /plan use brings it back.")
        clear.clicked.connect(self.clear_requested.emit)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(head, stretch=1)
        top.addWidget(clear)
        self.lay.addLayout(top)

        for s in plan["steps"]:
            row = QtWidgets.QHBoxLayout()
            btn = QtWidgets.QPushButton("▶")
            btn.setFixedWidth(30)
            btn.setToolTip(f"Run step {s['n']} (re-reads step-{s['n']}.md, "
                           "so edit it first if you want to change it)")
            btn.clicked.connect(
                lambda _=False, n=s["n"]: self.run_step_requested.emit(n))
            label = QtWidgets.QLabel(
                f"{self._ICON.get(s['status'], '○')} <b>{s['n']}.</b> "
                f"{s['title']}"
                + (f"<br><i style='color:gray'>{s['result']}</i>"
                   if s["result"] else ""))
            label.setWordWrap(True)
            row.addWidget(btn)
            row.addWidget(label, stretch=1)
            self.lay.addLayout(row)

        nxt = QtWidgets.QPushButton("▶ Run next step")
        nxt.clicked.connect(self.run_next_requested.emit)
        nxt.setEnabled(done < len(plan["steps"]))
        self.lay.addWidget(nxt)


class ChatPanel(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("PhysicsIQ AI", parent)
        self.setObjectName("PhysicsIQChatPanel")

        from physicsiq.agent.loop import ChatController
        self.controller = ChatController(parent=self)

        self._stream_buffer = ""      # answer text of the bubble being streamed
        self._think_buffer = ""       # reasoning trace (shown dimly)
        self._stream_widget = None    # the bubble being streamed into
        self._pending_images = []     # attachments for the next message
        self._plan_card = None        # the live checklist, if a plan exists
        # Throttle: repaint the streaming bubble at most every 80 ms.
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(80)
        self._render_timer.timeout.connect(self._flush_stream)

        self._build_ui()
        self._wire_controller()

    def _build_ui(self):
        from physicsiq.ui.model_selector import ModelSelector

        root = QtWidgets.QWidget()
        self.setWidget(root)
        outer = QtWidgets.QVBoxLayout(root)

        # -- top bar: model choice + server control
        top = QtWidgets.QHBoxLayout()
        self.model_selector = ModelSelector()
        self.model_selector.model_chosen.connect(self._on_model_chosen)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setToolTip("Start llama-server with this model")
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.status_label = QtWidgets.QLabel("○ no model")
        top.addWidget(self.model_selector, stretch=1)
        top.addWidget(self.load_btn)
        top.addWidget(self.status_label)
        outer.addLayout(top)
        # Apply whatever the dropdown restored from last session.
        kind, name, _mm = self.model_selector.current_model()
        if kind:
            self._on_model_chosen(kind, name, _mm)

        # -- middle: scrolling history
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        self.msg_layout = QtWidgets.QVBoxLayout(inner)
        self.msg_layout.addStretch()  # keeps bubbles pinned to the top
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll, stretch=1)

        # -- bottom: input row
        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText(
            "Describe a part or ask a question…  (Ctrl+Enter to send)")
        self.input.setFixedHeight(64)
        outer.addWidget(self.input)

        # -- bottom row: ONE toggle you flip often, the rest behind ⚙
        # Which brain answers is the dropdown's job now, not a checkbox here.
        bottom = QtWidgets.QHBoxLayout()
        self.plan_box = QtWidgets.QCheckBox("plan mode")
        self.plan_box.setToolTip(
            "ON = your next message is PLANNED, not built: the model thinks "
            "it through and writes plans/<name>/step-*.md. Then run the steps "
            "one at a time — each in a fresh context, so a long job never "
            "fills the model's window. Use it for anything bigger than a "
            "single part, and pick a cloud model to have Claude write it.")
        self.plan_box.toggled.connect(self._on_plan_mode_toggled)

        self.settings_btn = self._build_settings_menu()

        self.attach_btn = QtWidgets.QPushButton("")
        self.attach_btn.setToolTip(
            "Attach image(s) — sent with your next message. Needs a model "
            "that can see: a local [vision] gguf, or any cloud Claude.")
        self.attach_btn.setFixedWidth(44)
        self.attach_btn.clicked.connect(self._on_attach)
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.controller.cancel)
        self.stop_btn.setEnabled(False)

        bottom.addWidget(self.plan_box)
        bottom.addWidget(self.settings_btn)
        bottom.addStretch()
        bottom.addWidget(self.attach_btn)
        bottom.addWidget(self.send_btn)
        bottom.addWidget(self.stop_btn)
        outer.addLayout(bottom)

        # Ctrl+Enter sends — muscle memory from every chat app.
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"),
                        self.input, activated=self._on_send)

    def _build_settings_menu(self) -> QtWidgets.QToolButton:
        """The knobs you set once and forget, tucked out of the way. A row of
        four checkboxes made the panel look like a cockpit."""
        from physicsiq.settings import S

        btn = QtWidgets.QToolButton()
        btn.setText("⚙")
        btn.setToolTip("Agent options")
        btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(btn)

        auto = menu.addAction("auto-run code")
        auto.setCheckable(True)
        auto.setChecked(S.auto_run())
        auto.setToolTip("ON = agent mode: generated code executes immediately "
                        "(still undoable with Ctrl+Z). OFF = you click ▶ Run "
                        "on each block.")
        auto.toggled.connect(S.set_auto_run)

        visual = menu.addAction("visual check")
        visual.setCheckable(True)
        visual.setChecked(S.visual_check())
        visual.setToolTip(
            "After code runs, screenshot the viewport and let the model look "
            f"at its own work. Up to {S.max_visual_fixes()} corrections, each "
            "one verified, then it stops.")
        visual.toggled.connect(S.set_visual_check)

        menu.addSeparator()
        menu.addAction("clear the plan",
                       lambda: self.send_external("/clear-plan"))
        menu.addAction("command help", lambda: self.send_external("/help"))

        btn.setMenu(menu)
        self._auto_run_action = auto      # keep refs: a GC'd QAction is a crash
        self._visual_action = visual
        self._settings_menu = menu
        return btn

    def _wire_controller(self):
        c = self.controller
        c.assistant_chunk.connect(self._on_chunk)
        c.assistant_thinking.connect(self._on_thinking)
        c.assistant_done.connect(self._on_done)
        c.code_ready.connect(self._add_code_card)
        c.exec_finished.connect(self._on_exec_finished)
        c.status_changed.connect(self._set_status)
        c.busy_changed.connect(self._on_busy)
        c.stream_aborted.connect(self._on_stream_aborted)
        c.plan_changed.connect(self._on_plan_changed)
        c.command_result.connect(lambda text: self._add_message("system", text))
        # A plan from an earlier session is still on disk — show it.
        from physicsiq.agent import plan_store
        existing = plan_store.load()
        if existing is not None:
            self._on_plan_changed(existing)

    # plan mode
    def _on_plan_mode_toggled(self, on: bool):
        self.controller.plan_mode = on
        self.input.setPlaceholderText(
            "Describe the WHOLE job — the model will think and write a plan…"
            if on else
            "Describe a part or ask a question…  (Ctrl+Enter to send)")

    def _on_plan_changed(self, plan):
        """A plan was written, a step finished, or /clear-plan ran. ONE card,
        updated in place — we never stack a new card per step."""
        self.plan_box.setChecked(self.controller.plan_mode)
        if plan is None:                      # /clear-plan — take the card away
            if self._plan_card is not None:
                self.msg_layout.removeWidget(self._plan_card)
                self._plan_card.deleteLater()
                self._plan_card = None
            return
        if self._plan_card is None:
            self._plan_card = PlanCard(plan)
            self._plan_card.run_step_requested.connect(
                self.controller.run_step)
            self._plan_card.run_next_requested.connect(
                self.controller.run_next_step)
            self._plan_card.clear_requested.connect(
                self.controller.abandon_plan)
            self.msg_layout.insertWidget(
                self.msg_layout.count() - 1, self._plan_card)
        else:
            self._plan_card.set_plan(plan)
        self._scroll_to_bottom()

    # server / model handling
    def _on_model_chosen(self, kind: str, name: str, mmproj):
        """The dropdown is the ONE place you choose a brain. A cloud entry needs
        no loading — it's ready at once, and each message you send costs a
        prompt. A local entry needs llama-server, hence [Load].

        Note we do NOT stop a running llama-server when you switch to cloud: the
        agent's own loops (retries, visual checks) keep using it and stay
        free. That's the cheap way to work — good brain where you asked for
        it, local everywhere else."""
        from physicsiq.ui.model_selector import CLOUD

        if kind == CLOUD:
            self.controller.cloud_model = name
            self.load_btn.setEnabled(False)
            self.load_btn.setToolTip("cloud models need no loading")
            from physicsiq.llm.server import MANAGER
            local = (" · retries stay on the loaded gguf"
                     if MANAGER.is_running() else "")
            self._set_status(f"cloud {name} — 1 prompt per message{local}")
        else:
            self.controller.cloud_model = None
            self.load_btn.setEnabled(True)
            self.load_btn.setToolTip("Start llama-server with this model")
            self._set_status("press Load to start this model")

    def _on_load_clicked(self):
        from physicsiq.ui.workers import ServerStartWorker, shutdown_worker
        kind, model, mmproj = self.model_selector.current_model()
        if not model or kind != "local":
            self._set_status("no .gguf found in models dir")
            return
        self._set_status("loading model…")
        self.load_btn.setEnabled(False)
        shutdown_worker(getattr(self, "_server_worker", None))
        self._server_worker = ServerStartWorker(model, mmproj, parent=self)
        self._server_worker.ready.connect(
            lambda: (self._set_status("● model ready"),
                     self.load_btn.setEnabled(True)))
        self._server_worker.failed.connect(
            lambda msg: (self._set_status(msg),
                         self.load_btn.setEnabled(True)))
        self._server_worker.start()

    # chat flow
    def _on_attach(self):
        """Pick image files; they ride along with the NEXT message."""
        from physicsiq.llm.client import load_image_file
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Attach images", "",
            "Images (*.png *.jpg *.jpeg)")
        for path in paths:
            try:
                self._pending_images.append(load_image_file(path))
            except OSError as exc:
                self._set_status(f"could not read {path}: {exc}")
        if self._pending_images:
            self.attach_btn.setText(f"attach ({len(self._pending_images)})")
            self._set_status(
                f"{len(self._pending_images)} image(s) attached — they "
                "will be sent with your next message")

    def _on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        images = self._pending_images or None
        self._pending_images = []
        self.attach_btn.setText("attach")
        self.send_external(text, images=images)

    def send_external(self, text: str, images=None):
        """Send a message into the chat as if typed — also used by the PDF
        viewer ('send page to AI'), piq_tools.ask_user, and the ⚙ menu."""
        from physicsiq.agent import chat_commands
        from physicsiq.llm.server import MANAGER

        # A /command needs no model at all, and a cloud brain talks to Claude, not
        # to llama-server — so only a plain LOCAL message needs a loaded gguf.
        needs_local = (not chat_commands.is_command(text)
                       and self.controller.cloud_model is None)
        if needs_local and not MANAGER.is_running():
            self._set_status("press Load to start this model, or pick a cloud one")
            return
        # Ctrl+Enter bypasses the disabled Send button, and ask_user() can fire
        # at any moment — without this guard a second message kills the stream
        # that is still running and both replies get mangled.
        if not self.send_btn.isEnabled():
            self._set_status("still answering — press Stop first")
            return

        suffix = f"  ({len(images)} image)" if images else ""
        if self.controller.cloud_model is not None:
            suffix += "  cloud"
        self._add_message("user", text + suffix)
        # The assistant bubble is opened by _on_busy — every path that
        # generates (chat, plan, a step from the plan card, a self-correction
        # retry, a visual check) goes through busy_changed, so that is the ONE
        # place that needs to know a reply is coming.
        self.controller.send_user_message(text, images=images)

    def _on_chunk(self, piece: str):
        # Called VERY often. Only append to a string here; the QTimer
        # does the (expensive) widget update at a sane rate.
        self._stream_buffer += piece

    def _on_thinking(self, piece: str):
        # Reasoning-model thoughts: shown dimly while they stream so the
        # panel never looks frozen, dropped once the real answer starts.
        self._think_buffer += piece

    def _flush_stream(self):
        if self._stream_widget is None:
            return
        if self._stream_buffer:
            self._stream_widget.set_text(self._stream_buffer)
        elif self._think_buffer:
            # only thoughts so far — show the trailing bit as a quote
            tail = self._think_buffer[-300:]
            self._stream_widget.set_text(f"> *thinking…* {tail}")
        self._scroll_to_bottom()

    def _on_done(self, text: str):
        self._end_stream(text)

    def _on_stream_aborted(self, message: str):
        """The generation died (error, cancel, or a reasoning model that
        spent its whole budget thinking). Tear the streaming bubble down —
        otherwise it sits on '…' forever and the 80 ms repaint timer keeps
        firing for the rest of the session."""
        self._end_stream(f"*(no answer — {message})*")

    def _end_stream(self, final_text: str):
        self._render_timer.stop()
        if self._stream_widget is not None:
            self._stream_widget.set_text(final_text)
        self._stream_widget = None
        self._stream_buffer = ""
        self._think_buffer = ""
        self._scroll_to_bottom()

    def _add_code_card(self, code: str):
        card = CodeCard(code)
        card.run_requested.connect(self.controller.run_code)
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, card)
        self._scroll_to_bottom()

    def _on_exec_finished(self, ok: bool, message: str):
        icon = "✓" if ok else "✗"
        self._add_message("system", f"{icon} {message}")

    def _begin_stream(self):
        """Open a fresh assistant bubble for a reply that is about to stream."""
        self._stream_buffer = ""
        self._think_buffer = ""
        self._stream_widget = self._add_message("assistant", "…")
        self._render_timer.start()

    def _on_busy(self, busy: bool):
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        # Busy means a reply is on its way — no matter who asked for it
        # (you, a plan step, or the self-correction loop). Give it a bubble.
        if busy and self._stream_widget is None:
            self._begin_stream()

    # small helpers
    def _add_message(self, role: str, text: str) -> MessageWidget:
        w = MessageWidget(role, text)
        # insert before the trailing stretch item
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, w)
        self._scroll_to_bottom()
        return w

    def _scroll_to_bottom(self):
        # singleShot(0, ...) waits one event-loop tick so the layout has
        # updated its sizes BEFORE we scroll — otherwise we always land
        # one message short of the bottom.
        QtCore.QTimer.singleShot(
            0, lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()))

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def closeEvent(self, event):
        # Dock close = hide, but if FreeCAD is really shutting down we must
        # stop worker threads FIRST — a live QThread at interpreter
        # teardown is a guaranteed segfault.
        self.controller.shutdown()
        super().closeEvent(event)
