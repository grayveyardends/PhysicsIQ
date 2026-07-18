"""pinn_runner.py — bridges FreeCAD and the PINN venv.

The two sides can never share a process (FreeCAD = python 3.14 + OCCT,
PINN = venv python 3.11 + JAX), so the bridge is files + a subprocess:

    FreeCAD side (this file, main thread):
        runs/job_<n>/part.step   <- exported geometry
        runs/job_<n>/bcs.json    <- boundary conditions + material
    then launches:
        ~/.venvs/usage/bin/python -m physicsiq_pinn.cli --step ... --bcs ...
    PINN side writes back:
        runs/job_<n>/result.json + field.npz   (grep stdout for RESULT_JSON)

The subprocess runs inside a SubprocessWorker (QThread), so FreeCAD stays
fully usable while the GPU trains.
"""

import json
import os
import time

from PySide6 import QtCore


def _export_solid(obj, step_path) -> int:
    """Write ONE connected solid to a STEP file. Returns how many solids the
    object turned out to be made of.

    Two traps, both found the hard way on an 'F450 Drone' frame:

    1. `Part.export([obj], path)` only accepts Part::Feature objects. An
       App::Link or App::LinkGroup — what you get the moment the model reuses
       an arm four times, or builds an assembly — HAS a perfectly good .Shape
       full of solids, but is NOT a Part::Feature. So Part.export prints
           'F450 Drone' is not a shape, export will be ignored.
       and writes nothing, and the PINN then fails on a missing file. Export
       the SHAPE and the question of which class owns it never arises.

    2. An assembly is MANY solids, and elasticity needs ONE connected body —
       a load has to have a path from the fixed face to the loaded face.
       Touching solids get fused. Genuinely separate ones are an error worth
       naming out loud, because the old code took Solids[0] and silently
       analyzed a single arm.
    """
    shape = getattr(obj, "Shape", None)
    if shape is None or shape.isNull():
        raise RuntimeError(f"'{obj.Label}' has no shape to analyze")
    solids = shape.Solids
    if not solids:
        raise RuntimeError(
            f"'{obj.Label}' has no solid volume — it looks like a sketch, a "
            "surface or a mesh. The PINN needs a solid body.")

    if len(solids) > 1:
        fused = solids[0].multiFuse(solids[1:])
        fused = fused.removeSplitter()    # merge the coplanar faces the fuse left
        if len(fused.Solids) != 1:
            raise RuntimeError(
                f"'{obj.Label}' is {len(fused.Solids)} solids that do not "
                "touch each other. A stress analysis needs one connected "
                "body: fuse the pieces (Part::MultiFuse), or analyze a single "
                "part by passing its label.")
        solid = fused.Solids[0]
    else:
        solid = solids[0]

    solid.exportStep(step_path)
    return len(solids)


def _pick_target(label=None):
    """Which object gets analyzed? Explicit label > current selection >
    the largest solid in the document (usually 'the part')."""
    import FreeCAD as App
    import FreeCADGui as Gui

    doc = App.ActiveDocument
    if doc is None:
        raise RuntimeError("No document open")

    def has_solid(o):
        s = getattr(o, "Shape", None)
        return s is not None and not s.isNull() and s.Solids

    if label:
        for o in doc.Objects:
            if o.Label == label and has_solid(o):
                return o
        raise RuntimeError(f"No solid object labelled '{label}'")

    sel = [o for o in Gui.Selection.getSelection() if has_solid(o)]
    if sel:
        return sel[0]

    solids = [o for o in doc.Objects if has_solid(o)]
    if not solids:
        raise RuntimeError("Document has no solid to analyze")
    return max(solids, key=lambda o: o.Shape.Volume)


class PinnJob(QtCore.QObject):
    progress = QtCore.Signal(str)      # live training log lines
    live_field = QtCore.Signal(str)    # path to field_live.npz, mid-training
    finished_ok = QtCore.Signal(dict)  # parsed result.json
    failed = QtCore.Signal(str)

    def __init__(self, parent=None, target_label=None,
                 force_N=(0.0, 0.0, -500.0),
                 fixed_rule="zmin", load_rule="zmax", epochs=15000):
        # 15k epochs ≈ 30-40 s on the 4060 — the stress field needs the
        # iterations to develop (at 3k it still hugs the beam-theory
        # baseline and misses concentrations).
        super().__init__(parent)
        self.target_label = target_label
        self.force_N = list(force_N)
        self.fixed_rule = fixed_rule
        self.load_rule = load_rule
        self.epochs = epochs
        self._worker = None
        self._result_path = None
        self.target_object_label = None   # filled in start(), overlay uses it

    def is_running(self):
        return self._worker is not None and self._worker.isRunning()

    def start(self):
        """Export + launch. Runs on the MAIN thread (touches the document);
        only the long subprocess part happens on the worker thread."""
        from physicsiq import REPO_DIR, runs_dir
        from physicsiq.settings import S
        from physicsiq.ui.workers import SubprocessWorker

        try:
            obj = _pick_target(self.target_label)
        except RuntimeError as exc:
            self.failed.emit(str(exc))
            return
        self.target_object_label = obj.Label

        job_dir = os.path.join(runs_dir(), f"job_{int(time.time())}")
        os.makedirs(job_dir, exist_ok=True)
        step_path = os.path.join(job_dir, "part.step")
        try:
            n_solids = _export_solid(obj, step_path)
        except RuntimeError as exc:
            self.failed.emit(str(exc))
            return
        if n_solids > 1:
            self.progress.emit(
                f"'{obj.Label}' was {n_solids} solids — fused into one "
                "connected body for the analysis")

        bcs = {
            "fixed": {"rule": self.fixed_rule},
            "load": {"rule": self.load_rule, "force_N": self.force_N},
            "material": {"E": 70000.0, "nu": 0.33,
                         "yield_mpa": S.yield_strength_mpa()},
        }
        with open(os.path.join(job_dir, "bcs.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(bcs, fh, indent=2)

        argv = [S.pinn_python(), "-m", "physicsiq_pinn.cli",
                "--step", step_path,
                "--bcs", os.path.join(job_dir, "bcs.json"),
                "--out", job_dir, "--epochs", str(self.epochs)]
        self._worker = SubprocessWorker(argv, parent=self)
        os.environ.setdefault("PIQ_FREECADCMD", "freecadcmd")

        # SubprocessWorker doesn't take env; simplest correct fix: pass it
        # through the environment of THIS process, which the child inherits.
        pinn_src = os.path.join(REPO_DIR, "pinn", "src")
        existing = os.environ.get("PYTHONPATH", "")
        if pinn_src not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                pinn_src + (os.pathsep + existing if existing else ""))

        self._worker.line_read.connect(self._on_line)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self.failed)
        self.progress.emit(f"PINN job started on '{obj.Label}' "
                           f"(force {self.force_N} N, fixed={self.fixed_rule}, "
                           f"load={self.load_rule})")
        self._worker.start()

    def shutdown(self):
        from physicsiq.ui.workers import shutdown_worker
        shutdown_worker(self._worker)

    def _on_line(self, line: str):
        if line.startswith("RESULT_JSON "):
            self._result_path = line.split(" ", 1)[1].strip()
        elif line.startswith("LIVE_FIELD "):
            # "LIVE_FIELD <path> epoch=<n>" — the overlay repaints from it
            self.live_field.emit(line.split(" ", 2)[1])
        else:
            self.progress.emit(line)

    def _on_done(self, _code: int):
        if not self._result_path or not os.path.exists(self._result_path):
            self.failed.emit("PINN finished but wrote no result.json")
            return
        with open(self._result_path, encoding="utf-8") as fh:
            report = json.load(fh)
        report["target_label"] = self.target_object_label
        self.finished_ok.emit(report)
