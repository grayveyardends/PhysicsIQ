"""model_selector.py — dropdown that lists .gguf models and pairs each one
with its vision projector (mmproj) file.

Pairing rule (matches how HuggingFace ships them):
    model : gemma-4-E2B-it-Q8_0.gguf
    mmproj: mmproj-gemma-4-E2B-it-Q8_0.gguf   ← "mmproj-" + same filename

Fallback: if there is exactly ONE mmproj-*.gguf in the folder, use it for
any model that lacks an exact match. No mmproj = text-only chat, which
still works fine; vision features just politely refuse.

Drop a bigger/better .gguf into models/ and it appears here on refresh —
that is the whole "super model" upgrade path.
"""

import os

from PySide6 import QtCore, QtWidgets


def scan_models(models_dir: str):
    """Return [(display_name, model_path, mmproj_path_or_None), ...]."""
    if not os.path.isdir(models_dir):
        return []
    ggufs = sorted(f for f in os.listdir(models_dir) if f.endswith(".gguf"))
    mmprojs = [f for f in ggufs if f.startswith("mmproj-")]
    models = [f for f in ggufs if not f.startswith("mmproj-")]

    result = []
    for name in models:
        exact = "mmproj-" + name
        if exact in mmprojs:
            mm = exact
        elif len(mmprojs) == 1:
            mm = mmprojs[0]
        else:
            mm = None
        result.append((
            name.removesuffix(".gguf"),
            os.path.join(models_dir, name),
            os.path.join(models_dir, mm) if mm else None,
        ))
    return result


class ModelSelector(QtWidgets.QComboBox):
    # Emitted when the user picks a model: (model_path, mmproj_path|None)
    model_chosen = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip("Pick which local .gguf model answers the chat")
        self.refresh()
        # 'activated' fires only on USER selection, not programmatic ones —
        # exactly what we want to avoid restart loops on refresh().
        self.activated.connect(self._on_activated)

    def refresh(self):
        from inferio.settings import S
        self.blockSignals(True)
        self.clear()
        for display, model_path, mmproj in scan_models(S.models_dir()):
            label = display + ("  [vision]" if mmproj else "")
            self.addItem(label, (model_path, mmproj))
        self.blockSignals(False)
        # Restore last used model so the panel comes back the way you left it.
        last = S.last_model()
        for i in range(self.count()):
            if self.itemData(i)[0] == last:
                self.setCurrentIndex(i)
                break

    def current_model(self):
        """(model_path, mmproj) of the current selection, or (None, None)."""
        data = self.currentData()
        return data if data else (None, None)

    def _on_activated(self, index):
        from inferio.settings import S
        model_path, mmproj = self.itemData(index)
        S.set_last_model(model_path)
        self.model_chosen.emit(model_path, mmproj)
