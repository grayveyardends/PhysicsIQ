"""panel_manager.py — creates dock panels once and re-shows them after.

FreeCAD's main window is a plain QMainWindow, so we can dock our own
QDockWidgets into it like any Qt app. We keep module-level references
because a QDockWidget with no Python reference gets garbage collected
while Qt still points at it -> crash. (Classic PySide footgun.)
"""

from PySide6 import QtCore, QtWidgets

import FreeCADGui

_chat_panel = None
_pdf_panel = None


def main_window() -> QtWidgets.QMainWindow:
    return FreeCADGui.getMainWindow()


def show_chat_panel():
    global _chat_panel
    if _chat_panel is None:
        from physicsiq.ui.chat_panel import ChatPanel
        _chat_panel = ChatPanel(main_window())
        main_window().addDockWidget(
            QtCore.Qt.RightDockWidgetArea, _chat_panel)
    _chat_panel.show()
    _chat_panel.raise_()
    return _chat_panel


def show_pdf_panel(path=None):
    global _pdf_panel
    if _pdf_panel is None:
        from physicsiq.ui.pdf_viewer import PdfPanel
        _pdf_panel = PdfPanel(main_window())
        main_window().addDockWidget(
            QtCore.Qt.LeftDockWidgetArea, _pdf_panel)
    if path:
        _pdf_panel.open_pdf(path)
    _pdf_panel.show()
    _pdf_panel.raise_()
    return _pdf_panel


def chat_panel_or_none():
    return _chat_panel
