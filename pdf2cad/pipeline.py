#!/usr/bin/env python3
"""pipeline.py — PDF page -> vision model -> structured part description.

Standalone tester for the PDF->CAD idea (no FreeCAD needed): needs a
running llama-server WITH --mmproj (start one via scripts/run_server.sh).

    python pdf2cad/pipeline.py drawing.pdf 1
        -> prints the model's reading of page 1: dimensions, part
          description, and its FreeCAD-python attempt.

Inside FreeCAD the same flow runs through the PDF dock's "Send page to
AI" button; this file exists so you can iterate on the PROMPT quickly
without clicking through the GUI each time.
"""

import sys

# Reuse the workbench's HTTP client instead of duplicating it. The addon
# folder is a plain python package, so pathing it in is enough.
import os
_REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "PhysicsIQ"))
sys.path.insert(0, _REPO)   # so `import pdf2cad` works when run as a script

from physicsiq.llm.client import LlamaClient, image_part  # noqa: E402
from pdf2cad.extract import page_png, page_text  # noqa: E402

READ_PROMPT = """You are reading one page of an engineering document.
1. Say what part(s) the page shows, in one sentence.
2. List EVERY dimension you can read, as `name: value unit`.
3. If the page gives enough information, write FreeCAD python (Part
   primitives, mm) that models the main part. If not, list what is missing.
"""


def describe_page(pdf_path: str, page_no: int,
                  base_url="http://127.0.0.1:8735") -> str:
    png = page_png(pdf_path, page_no)
    text = page_text(pdf_path, page_no).strip()
    prompt = READ_PROMPT
    if text:
        # Give the model the text layer too — reading pixels is hard,
        # cross-checking against embedded text makes it far more accurate.
        prompt += f"\nThe page's embedded text layer says:\n{text[:2000]}"

    client = LlamaClient(base_url)
    return client.chat(
        [{"role": "user",
          "content": [{"type": "text", "text": prompt}, image_part(png)]}],
        max_tokens=1200)


if __name__ == "__main__":
    pdf = sys.argv[1]
    page = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    print(describe_page(pdf, page))
