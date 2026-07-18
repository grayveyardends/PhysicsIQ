"""pdf_viewer.py — a dock that shows PDFs (datasheets, drawings) and can
send any page as an IMAGE to the vision model: the first half of the
PDF -> CAD pipeline, driven by eyeballs instead of parsing.

Rendering: PyMuPDF (import name `fitz`) rasterizes a page to RGB bytes ->
QImage -> QLabel. No QtPdf dependency, works on any Qt build.
Install once into FreeCAD's python:  pip install --user PyMuPDF
"""

from PySide6 import QtCore, QtGui, QtWidgets


def _require_fitz():
    try:
        import fitz
        return fitz
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed in FreeCAD's python.\n"
            "Run:  pip install --user PyMuPDF   and restart FreeCAD.")


class PdfPanel(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("PDF (PhysicsIQ)", parent)
        self.setObjectName("PhysicsIQPdfPanel")
        self._doc = None
        self._page_index = 0
        self._zoom = 1.5          # 1.0 ≈ 72 dpi; 1.5 is comfortable

        root = QtWidgets.QWidget()
        self.setWidget(root)
        lay = QtWidgets.QVBoxLayout(root)

        bar = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        self.page_label = QtWidgets.QLabel("– / –")
        self.send_btn = QtWidgets.QPushButton("Send page to AI")
        self.send_btn.setToolTip(
            "Renders this page and asks the vision model to read it")
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.next_btn)
        bar.addWidget(self.page_label)
        bar.addStretch()
        bar.addWidget(self.send_btn)
        lay.addLayout(bar)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QtWidgets.QLabel("Open a PDF via PhysicsIQ > Open PDF")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        lay.addWidget(self.scroll, stretch=1)

        self.prev_btn.clicked.connect(lambda: self._go(-1))
        self.next_btn.clicked.connect(lambda: self._go(+1))
        self.send_btn.clicked.connect(self._send_page_to_ai)

    def open_pdf(self, path: str):
        fitz = _require_fitz()
        self._doc = fitz.open(path)
        self._page_index = 0
        self._render()

    def _go(self, delta: int):
        if self._doc is None:
            return
        self._page_index = max(0, min(len(self._doc) - 1,
                                      self._page_index + delta))
        self._render()

    def _render(self):
        fitz = _require_fitz()
        page = self._doc[self._page_index]
        # Matrix scales the 72dpi page; higher zoom = sharper = more bytes.
        pix = page.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom))
        img = QtGui.QImage(pix.samples, pix.width, pix.height,
                           pix.stride, QtGui.QImage.Format_RGB888)
        # .copy(): QImage must own its memory — pix.samples dies with pix.
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(img.copy()))
        self.page_label.setText(f"{self._page_index + 1} / {len(self._doc)}")

    def _send_page_to_ai(self):
        if self._doc is None:
            return
        fitz = _require_fitz()
        page = self._doc[self._page_index]
        # 2x zoom for the model: small text in drawings needs the pixels.
        png = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")

        from physicsiq import panel_manager
        chat = panel_manager.show_chat_panel()
        chat.send_external(
            "This is a page from an engineering document. Describe the "
            "part(s) shown, list every dimension you can read (with units), "
            "and if there is enough information, produce FreeCAD python "
            "to model the main part.",
            images=[png])
