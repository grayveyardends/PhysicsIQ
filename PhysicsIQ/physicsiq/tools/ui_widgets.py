"""ui_widgets.py — Qt UI the LLM is ALLOWED to create.

The master prompt forbids raw blocking dialogs (`input()`, `dialog.exec()`)
because they freeze the agent mid-script. These helpers are the sanctioned
alternative: everything here is NON-blocking — widgets are shown, the
script continues, and answers come back later as chat messages.

Golden rule of this file: keep a Python reference to every widget we
create (_LIVE list). A PySide widget with no reference gets garbage
collected while Qt still points at it -> crash. Main thread only.
"""

from PySide6 import QtCore, QtWidgets

import FreeCADGui as Gui

# Living widgets. Never trimmed automatically — closing a widget is cheap,
# a dangling C++ pointer is not.
_LIVE = []


def popup(message: str, title: str = "PhysicsIQ"):
    """Non-modal info popup. Fire-and-forget: the script keeps running."""
    box = QtWidgets.QMessageBox(Gui.getMainWindow())
    box.setWindowTitle(title)
    box.setText(str(message))
    box.setIcon(QtWidgets.QMessageBox.Information)
    box.setModal(False)          # modal would freeze the agent turn
    box.show()
    _LIVE.append(box)
    return f"popup shown: {title}"


def ask_user(question: str, placeholder: str = ""):
    """Ask the human something via a small input box. NON-blocking:
    this returns immediately; when the user hits OK, their answer is
    injected into the chat as a new user message, so the model simply
    sees it next turn. That's how you do 'input()' in an event loop."""
    dialog = QtWidgets.QDialog(Gui.getMainWindow())
    dialog.setWindowTitle("PhysicsIQ asks")
    lay = QtWidgets.QVBoxLayout(dialog)
    lay.addWidget(QtWidgets.QLabel(str(question)))
    edit = QtWidgets.QLineEdit()
    edit.setPlaceholderText(placeholder)
    lay.addWidget(edit)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    lay.addWidget(buttons)
    buttons.rejected.connect(dialog.close)

    def deliver():
        answer = edit.text().strip()
        dialog.close()
        if not answer:
            return
        from physicsiq import panel_manager
        panel = panel_manager.chat_panel_or_none()
        if panel is not None:
            # back into the conversation it goes — the model reads it
            # as an ordinary user message
            panel.send_external(f"(answer to \"{question}\"): {answer}")

    buttons.accepted.connect(deliver)
    edit.returnPressed.connect(deliver)
    dialog.setModal(False)
    dialog.show()
    edit.setFocus()
    _LIVE.append(dialog)
    return ("question shown to the user — their answer will arrive as "
            "the next chat message; do not wait for it in this script")


def side_panel(title: str, markdown: str):
    """Open (or refresh) a temporary dock tab showing markdown content —
    notes, a bill of materials, a checklist, calculation results…
    Reusing the same title updates that panel instead of stacking a new
    one, so the model can keep a 'live' notes tab."""
    main = Gui.getMainWindow()
    name = f"PhysicsIQSide_{title}"

    dock = main.findChild(QtWidgets.QDockWidget, name)
    if dock is None:
        dock = QtWidgets.QDockWidget(title, main)
        dock.setObjectName(name)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        dock.setWidget(view)
        main.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        # tabify onto the chat panel if it's docked there, keeps it tidy
        from physicsiq import panel_manager
        chat = panel_manager.chat_panel_or_none()
        if chat is not None and not chat.isFloating():
            main.tabifyDockWidget(chat, dock)
        _LIVE.append(dock)
    dock.widget().setMarkdown(str(markdown))
    dock.show()
    dock.raise_()
    return f"side panel '{title}' shown"


def close_panels():
    """Close every popup/dialog/panel this module created."""
    count = 0
    for w in _LIVE:
        try:
            w.close()
            count += 1
        except RuntimeError:
            pass  # C++ side already gone — fine
    _LIVE.clear()
    return f"closed {count} widget(s)"
