"""server.py — owns the llama-server subprocess.

Big picture: we do NOT load the model inside FreeCAD. We launch the
system-installed `llama-server` (the CUDA build from the llama.cpp package)
as a child process and talk to it over HTTP on localhost. Why:

  * If the model crashes or eats VRAM, FreeCAD survives.
  * Vision (the mmproj file) works out of the box via the OpenAI API.
  * Swapping models = kill process, start a new one with a different -m.
  * Later, NeuroEmb can replace this file and nothing else changes.

This module is Qt-free on purpose so it can be tested from a plain
terminal: `python -c "from physicsiq.llm.server import ..."`.
"""

import atexit
import os
import subprocess
import tempfile
import time

import requests

LLAMA_SERVER_BIN = "/usr/bin/llama-server"

# One log file, overwritten per FreeCAD session. When the model misbehaves,
# `tail -f` this file — llama-server is chatty and honest.
LOG_PATH = os.path.join(tempfile.gettempdir(), "physicsiq-llama-server.log")


class LlamaServerManager:
    """Start/stop/health-check exactly one llama-server child process."""

    def __init__(self):
        self._proc = None          # subprocess.Popen or None
        self._model_path = None    # what we launched with, to detect "same model, no restart needed"
        self._mmproj_path = None
        self._log_file = None
        # If FreeCAD exits any way short of a hard crash, kill the child.
        # Without this you get orphan llama-servers squatting on VRAM.
        atexit.register(self.stop)

    def base_url(self) -> str:
        from physicsiq.settings import S
        return f"http://127.0.0.1:{S.server_port()}"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def current_model(self):
        return self._model_path

    def has_vision(self) -> bool:
        """Did we load a vision projector with this model? This is the
        authoritative answer (it's what we actually passed to --mmproj), which
        is why the visual-check loop asks HERE and not the dropdown."""
        return self._mmproj_path is not None

    def start(self, model_path: str, mmproj_path: str = None) -> None:
        """Launch llama-server. No-op if the same model is already up."""
        from physicsiq.settings import S

        if self.is_running() and self._model_path == model_path \
                and self._mmproj_path == mmproj_path:
            return  # already serving exactly this model

        self.stop()  # different model (or dead process) — clean slate

        argv = [
            LLAMA_SERVER_BIN,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(S.server_port()),
            "-c", str(S.ctx_size()),
            "-ngl", str(S.gpu_layers()),
            # Keep the prompt cache warm between requests: our master prompt
            # is identical every turn, so llama-server re-uses its KV cache
            # instead of re-reading the whole thing. This is the MVP version
            # of the "tokenize the master prompt once" idea.
            "--keep", "-1",
            # Use the chat template baked into the gguf, and — the reason we
            # care — let llama-server SPLIT a reasoning model's <think> block
            # into `reasoning_content` instead of dumping it into the answer.
            # Without this, DeepSeek-R1's musings become "the reply" and the
            # agent tries to execute code it found inside them.
            "--jinja",
        ]
        if mmproj_path:
            argv += ["--mmproj", mmproj_path]

        # 'w' truncates: one session, one log. Child inherits the handle,
        # so the log keeps filling even though we never touch it again.
        self._log_file = open(LOG_PATH, "w")
        self._proc = subprocess.Popen(
            argv, stdout=self._log_file, stderr=subprocess.STDOUT)
        self._model_path = model_path
        self._mmproj_path = mmproj_path

    def wait_healthy(self, timeout_s: float = 120.0) -> bool:
        """Block until /health says ready (model weights take seconds to
        load into VRAM). Call this from a WORKER thread, never the GUI
        thread — that's what ui/workers.ServerStartWorker is for."""
        deadline = time.time() + timeout_s
        url = self.base_url() + "/health"
        while time.time() < deadline:
            if not self.is_running():
                return False  # process died — check LOG_PATH
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    return True
            except requests.ConnectionError:
                pass  # not listening yet, normal during startup
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()          # polite SIGTERM first
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()           # it had its chance
                self._proc.wait()
        self._proc = None
        self._model_path = None
        self._mmproj_path = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


# Module-level singleton: exactly one server per FreeCAD process.
MANAGER = LlamaServerManager()
