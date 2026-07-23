#!/usr/bin/env python3
"""extract.py — pull text and page images out of a PDF (PyMuPDF).

This is the headless half of PDF -> CAD; the interactive half is the
PDF dock inside FreeCAD. Use this one for batch jobs and testing:

    python pdf2cad/extract.py drawing.pdf out_dir/
        -> out_dir/page1.png, page1.txt, page2.png, ...
"""

import os
import sys


def page_count(pdf_path: str) -> int:
    import fitz
    with fitz.open(pdf_path) as doc:
        return len(doc)


def page_png(pdf_path: str, page: int, zoom: float = 2.0) -> bytes:
    """Render one page (0-based) to PNG bytes. zoom=2 -> 144 dpi, enough
    for the vision model to read dimension text."""
    import fitz
    with fitz.open(pdf_path) as doc:
        pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")


def page_text(pdf_path: str, page: int, ocr: bool = True,
              dpi: int = 300) -> str:
    """The PDF's embedded text layer. Scanned drawings have none, so when
    the layer comes back empty we run the page through Tesseract instead."""
    import fitz
    with fitz.open(pdf_path) as doc:
        pg = doc[page]
        txt = pg.get_text()
        if txt.strip() or not ocr:
            return txt
        try:
            tp = pg.get_textpage_ocr(flags=0, dpi=dpi, full=True)
            return pg.get_text(textpage=tp)
        except Exception:
            return txt   # no tesseract / no tessdata: fall back to empty


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    pdf, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    for i in range(page_count(pdf)):
        with open(os.path.join(outdir, f"page{i + 1}.png"), "wb") as fh:
            fh.write(page_png(pdf, i))
        with open(os.path.join(outdir, f"page{i + 1}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(page_text(pdf, i))
        print(f"page {i + 1} done")


if __name__ == "__main__":
    main()
