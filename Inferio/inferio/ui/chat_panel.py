"""chat_panel.py — the dock widget you actually talk to.

Layout, top to bottom:

    [model dropdown] [Load] [● status]
    ┌────────────────────────────────┐
    │  scrolling chat history        │
    │  (message bubbles + code cards)│
    └────────────────────────────────┘
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
        who = {"user": "You", "assistant": "Inferio", "system": "System"}
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


class ChatPanel(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Inferio AI", parent)
        self.setObjectName("InferioChatPanel")

        from inferio.agent.loop import ChatController
        self.controller = ChatController(parent=self)

        self._stream_buffer = ""      # text of the bubble being streamed
        self._stream_widget = None    # the bubble being streamed into
        # Throttle: repaint the streaming bubble at most every 80 ms.
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setInterval(80)
        self._render_timer.timeout.connect(self._flush_stream)

        self._build_ui()
        self._wire_controller()

    # ------------------------------------------------------------------
    def _build_ui(self):
        from inferio.ui.model_selector import ModelSelector

        root = QtWidgets.QWidget()
        self.setWidget(root)
        outer = QtWidgets.QVBoxLayout(root)

        # -- top bar: model choice + server control ----------------------
        top = QtWidgets.QHBoxLayout()
        self.model_selector = ModelSelector()
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setToolTip("Start llama-server with this model")
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.status_label = QtWidgets.QLabel("○ no model")
        top.addWidget(self.model_selector, stretch=1)
        top.addWidget(self.load_btn)
        top.addWidget(self.status_label)
        outer.addLayout(top)

        # -- middle: scrolling history -----------------------------------
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        self.msg_layout = QtWidgets.QVBoxLayout(inner)
        self.msg_layout.addStretch()  # keeps bubbles pinned to the top
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll, stretch=1)

        # -- bottom: input row -------------------------------------------
        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText(
            "Describe a part or ask a question…  (Ctrl+Enter to send)")
        self.input.setFixedHeight(64)
        outer.addWidget(self.input)

        bottom = QtWidgets.QHBoxLayout()
        from inferio.settings import S
        self.auto_run_box = QtWidgets.QCheckBox("auto-run code")
        self.auto_run_box.setChecked(S.auto_run())
        self.auto_run_box.setToolTip(
            "ON = agent mode: generated code executes immediately "
            "(still undoable). OFF = you click Run on each block.")
        self.auto_run_box.toggled.connect(S.set_auto_run)
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.controller.cancel)
        self.stop_btn.setEnabled(False)
        bottom.addWidget(self.auto_run_box)
        bottom.addStretch()
        bottom.addWidget(self.send_btn)
        bottom.addWidget(self.stop_btn)
        outer.addLayout(bottom)

        # Ctrl+Enter sends — muscle memory from every chat app.
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"),
                        self.input, activated=self._on_send)

    def _wire_controller(self):
        c = self.controller
        c.assistant_chunk.connect(self._on_chunk)
        c.assistant_done.connect(self._on_done)
        c.code_ready.connect(self._add_code_card)
        c.exec_finished.connect(self._on_exec_finished)
        c.status_changed.connect(self._set_status)
        c.busy_changed.connect(self._on_busy)

    # ------------------------------------------------------------------
    # server / model handling
    # ------------------------------------------------------------------
    def _on_load_clicked(self):
        from inferio.ui.workers import ServerStartWorker, shutdown_worker
        model, mmproj = self.model_selector.current_model()
        if not model:
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

    # ------------------------------------------------------------------
    # chat flow
    # ------------------------------------------------------------------
    def _on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        from inferio.llm.server import MANAGER
        if not MANAGER.is_running():
            self._set_status("load a model first (Load button)")
            return
        self.input.clear()
        self._add_message("user", text)
        # Fresh assistant bubble that the stream will fill.
        self._stream_buffer = ""
        self._stream_widget = self._add_message("assistant", "…")
        self._render_timer.start()
        self.controller.send_user_message(text)

    def _on_chunk(self, piece: str):
        # Called VERY often. Only append to a string here; the QTimer
        # does the (expensive) widget update at a sane rate.
        self._stream_buffer += piece

    def _flush_stream(self):
        if self._stream_widget is not None and self._stream_buffer:
            self._stream_widget.set_text(self._stream_buffer)
            self._scroll_to_bottom()

    def _on_done(self, text: str):
        self._render_timer.stop()
        if self._stream_widget is not None:
            self._stream_widget.set_text(text)
        self._stream_widget = None
        self._scroll_to_bottom()

    def _add_code_card(self, code: str):
        card = CodeCard(code)
        card.run_requested.connect(self.controller.run_code)
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, card)
        self._scroll_to_bottom()

    def _on_exec_finished(self, ok: bool, message: str):
        icon = "✓" if ok else "✗"
        self._add_message("system", f"{icon} {message}")
        # A new streaming bubble may follow (self-correction retry).
        if not ok:
            self._stream_buffer = ""
            self._stream_widget = self._add_message("assistant", "…")
            self._render_timer.start()

    def _on_busy(self, busy: bool):
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
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
