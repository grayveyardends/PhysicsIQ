"""sampler.py — runs the freecadcmd sampling script and loads its output.

Why a subprocess? FreeCAD's OCCT kernel lives in FreeCAD's Python (3.14,
system), JAX lives in the venv (3.11). They can never share a process, so
they talk through files — see freecad_sample_script.py for the format.
"""

import json
import os
import subprocess

import numpy as np

_SCRIPT = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "freecad_sample_script.py")
FREECADCMD = os.environ.get("PIQ_FREECADCMD", "freecadcmd")


def sample_step(step_path, out_npz, n_interior=2000, n_boundary=1500):
    """Returns (npz dict, faces meta dict). Raises on failure with the
    subprocess log attached, so errors are debuggable from FreeCAD's UI."""
    # build the native lib here (this venv has a compiler-friendly env);
    # freecadcmd only ctypes-loads the finished .so
    from physicsiq_pinn.geometry import native
    native.load()
    pinn_src = os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", ".."))
    env = dict(os.environ,
               PIQ_STEP=os.path.abspath(step_path),
               PIQ_OUT=os.path.abspath(out_npz),
               PIQ_N_INTERIOR=str(n_interior),
               PIQ_N_BOUNDARY=str(n_boundary),
               PIQ_PINN_SRC=pinn_src)
    proc = subprocess.run(
        [FREECADCMD, _SCRIPT], env=env,
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not os.path.exists(out_npz):
        raise RuntimeError(
            f"geometry sampling failed (exit {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    print(proc.stdout.strip(), flush=True)

    data = dict(np.load(out_npz))
    with open(out_npz + ".faces.json", encoding="utf-8") as fh:
        faces = json.load(fh)
    return data, faces
