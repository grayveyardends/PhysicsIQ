"""workers.py — THE threading pattern for the whole workbench. Read this
file once and you understand every background job in PhysicsIQ.

THE ONE RULE (FreeCAD + Qt law, breaking it = crashes or corrupt docs):

    Only the MAIN thread may touch widgets or FreeCAD documents.

So how do we stream LLM tokens without freezing the 3D view?

    main thread                      worker thread (QThread)
    -----------
    worker.start()  ------------->   run() does the slow thing
                                      (HTTP streaming, subprocess, ...)
    slot updates the <-- signal ----  emit chunk_received("Hel")
    chat label                        emit chunk_received("lo")
                                      emit finished_ok(full_text)

Qt signals emitted from a QThread are QUEUED: Qt delivers them later, on
the main thread's event loop. That queue IS the thread boundary. The
worker never touches a widget; the slot never blocks.

Lifetime: every worker is parented/stored on the panel that started it and
is properly stopped in `shutdown()` (cancel flag -> quit -> wait). That is
the "kill them properly" part — orphaned QThreads segfault FreeCAD on exit.
"""

import subprocess

from PySide6 import QtCore


class LlmStreamWorker(QtCore.QThread):
    """Streams one chat completion from llama-server."""

    chunk_received = QtCore.Signal(str)   # a few characters of answer text
    thinking_received = QtCore.Signal(str)  # reasoning-model 'thoughts'
    finished_ok = QtCore.Signal(str)      # the complete answer text
    failed = QtCore.Signal(str)           # human-readable error

    def __init__(self, client, messages, temperature, max_tokens,
                 grammar=None, parent=None):
        super().__init__(parent)
        self._client = client
        self._messages = messages
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._grammar = grammar
        self._cancelled = False

    def cancel(self):
        # Just flips a flag. The stream loop polls it between chunks and
        # bails out. Never use QThread.terminate() — it can kill the thread
        # mid-malloc and take FreeCAD down with it.
        self._cancelled = True

    def run(self):  # executes on the WORKER thread
        from physicsiq.llm.client import GenerationCancelled
        try:
            text = self._client.chat_stream(
                self._messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                grammar=self._grammar,
                on_chunk=self.chunk_received.emit,   # signal = thread-safe
                on_thinking=self.thinking_received.emit,
                should_cancel=lambda: self._cancelled,
            )
            if not text.strip():
                self.failed.emit(
                    "model produced no answer (it may have spent the whole "
                    "token budget thinking — try again or raise MaxTokens)")
            else:
                self.finished_ok.emit(text)
        except GenerationCancelled:
            self.failed.emit("cancelled")
        except Exception as exc:  # noqa: BLE001 — anything -> UI, not a crash
            self.failed.emit(f"LLM request failed: {exc}")


class ServerStartWorker(QtCore.QThread):
    """Launches llama-server and waits for /health without blocking the GUI.
    Model loading takes ~5-20 s — plenty of time to freeze FreeCAD if we
    naively waited on the main thread."""

    ready = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, model_path, mmproj_path=None, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._mmproj_path = mmproj_path

    def run(self):
        from physicsiq.llm.server import MANAGER, LOG_PATH
        try:
            MANAGER.start(self._model_path, self._mmproj_path)
            if MANAGER.wait_healthy():
                self.ready.emit()
            else:
                self.failed.emit(
                    f"llama-server did not become healthy. See {LOG_PATH}")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Could not start llama-server: {exc}")


class SubprocessWorker(QtCore.QThread):
    """Runs any external command (the PINN training job, pdf2cad, ...)
    and streams its stdout lines back to the GUI as they appear."""

    line_read = QtCore.Signal(str)        # one stdout line (progress logs)
    finished_ok = QtCore.Signal(int)      # exit code 0
    failed = QtCore.Signal(str)           # non-zero exit or launch error

    def __init__(self, argv, cwd=None, parent=None):
        super().__init__(parent)
        self._argv = list(argv)
        self._cwd = cwd
        self._proc = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    def run(self):
        try:
            # text=True + bufsize=1 = line-buffered strings, so training
            # progress ("epoch 400 loss 3.2e-4") shows up live in the panel.
            self._proc = subprocess.Popen(
                self._argv, cwd=self._cwd, text=True, bufsize=1,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in self._proc.stdout:
                self.line_read.emit(line.rstrip("\n"))
            code = self._proc.wait()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Could not run {self._argv[0]}: {exc}")
            return
        if self._cancelled:
            self.failed.emit("cancelled")
        elif code == 0:
            self.finished_ok.emit(0)
        else:
            self.failed.emit(f"{self._argv[0]} exited with code {code}")


def shutdown_worker(worker, timeout_ms=3000):
    """Stop a worker THE RIGHT WAY: ask nicely, then wait for the thread
    to actually finish. Call for every live worker before closing a panel."""
    if worker is None:
        return
    if worker.isRunning():
        if hasattr(worker, "cancel"):
            worker.cancel()
        worker.quit()
        worker.wait(timeout_ms)
