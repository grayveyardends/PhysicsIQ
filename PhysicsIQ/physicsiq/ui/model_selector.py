"""model_selector.py — ONE dropdown: every brain you can talk to.

    gemma-4-E2B-it-Q8_0  [vision]     <- local gguf, runs on your GPU, free
    DeepSeek-R1-0528-Qwen3-8B         <- local gguf
    --------------
    cloud Claude sonnet                   <- the `claude` CLI, costs a prompt
    cloud Claude opus                        from your subscription per message
    cloud Claude fable

Local models are found by reading each .gguf's own header (llm/gguf_meta.py),
so a model is only paired with an mmproj that llama-server would actually
accept — no filename guessing. Cloud entries come from the CloudModels
setting; add or remove aliases there.

Picking a LOCAL model arms the [Load] button (llama-server has to start).
Picking a cloud model needs no loading — it's ready immediately, and each message
you send costs one prompt. Anything the agent does on its OWN (retries, the
visual check, the PINN loop) prefers a loaded local model and only falls back
to the cloud if you never loaded one — see agent/loop.py::_loop_uses_cloud.
"""

from PySide6 import QtCore, QtWidgets

from physicsiq.llm.gguf_meta import scan_models

LOCAL, CLOUD = "local", "cloud"


def list_brains(models_dir: str, cloud_models):
    """[(kind, label, name, mmproj), …] — everything the dropdown shows.
    `name` is a file path for local models and a model alias for cloud ones."""
    rows = [(LOCAL, display + ("  [vision]" if mm else ""), path, mm)
            for display, path, mm in scan_models(models_dir)]
    rows += [(CLOUD, f"cloud Claude {m}", m, None) for m in cloud_models]
    return rows


class ModelSelector(QtWidgets.QComboBox):
    # (kind, name, mmproj|None) — the panel decides what to do with it.
    model_chosen = QtCore.Signal(str, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("Which brain answers you: a local .gguf on your GPU, "
                        "or cloud Claude via your subscription")
        self.refresh()
        # 'activated' fires only on USER selection, not programmatic ones —
        # exactly what we want to avoid restart loops on refresh().
        self.activated.connect(self._on_activated)

    def refresh(self):
        from physicsiq.settings import S
        self.blockSignals(True)
        self.clear()
        rows = list_brains(S.models_dir(), S.cloud_models())
        first_cloud = True
        for kind, label, name, mmproj in rows:
            if kind == CLOUD and first_cloud and self.count():
                self.insertSeparator(self.count())
                first_cloud = False
            self.addItem(label, (kind, name, mmproj))
        self.blockSignals(False)

        # Restore the last pick so the panel comes back the way you left it.
        last = S.last_model()
        for i in range(self.count()):
            data = self.itemData(i)
            if data and data[1] == last:
                self.setCurrentIndex(i)
                break

    def current_model(self):
        """(kind, name, mmproj) of the current selection, or (None, None, None).
        A separator has no data, hence the guard."""
        data = self.currentData()
        return data if data else (None, None, None)

    def _on_activated(self, index):
        from physicsiq.settings import S
        data = self.itemData(index)
        if not data:              # they clicked the separator — ignore
            return
        kind, name, mmproj = data
        S.set_last_model(name)
        self.model_chosen.emit(kind, name, mmproj)
