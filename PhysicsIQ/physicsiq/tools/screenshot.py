"""screenshot.py — render the 3D view to PNG bytes for the vision model.

This closes the SOTA "visual feedback" loop: after generated code runs,
the agent can show the result to the mmproj vision model and ask "does
this look like what the user wanted?". Main thread only (touches the view).
"""

import os
import tempfile


def grab_3d_view(width=768, height=576) -> bytes:
    """Returns PNG bytes of the active 3D view, or raises if there is none.

    saveImage only writes to files, so we bounce through a temp file.
    768px is plenty for the vision encoder and keeps the request small.
    """
    import FreeCADGui as Gui

    view = Gui.ActiveDocument.ActiveView if Gui.ActiveDocument else None
    if view is None or not hasattr(view, "saveImage"):
        raise RuntimeError("No active 3D view to screenshot")

    fd, path = tempfile.mkstemp(suffix=".png", prefix="physicsiq-shot-")
    os.close(fd)
    try:
        # 'White' background renders best for the vision model — dark UI
        # themes confuse edge detection in small vision encoders.
        view.saveImage(path, width, height, "White")
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)  # temp files don't get to live forever
