"""native.py — ctypes loader for the geomkit C++ helpers.

The .so is optional. Everything it accelerates has a pure-python
fallback, so load() returning None just means "slower today". Built
lazily on first use (a few hundred ms of g++), never at import time.
"""

import ctypes
import os
import subprocess

import numpy as np

_NATIVE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "native"))
_SO = os.path.join(_NATIVE_DIR, "libgeomkit.so")
_lib = None
_tried = False


def load():
    """Returns the ctypes lib or None. Compiles on first call if needed."""
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True

    src = os.path.join(_NATIVE_DIR, "geomkit.cpp")
    if not os.path.exists(_SO) or (os.path.exists(src) and
                                   os.path.getmtime(src) > os.path.getmtime(_SO)):
        try:
            subprocess.run(["sh", os.path.join(_NATIVE_DIR, "build.sh")],
                           capture_output=True, timeout=60, check=True)
        except Exception:
            return None
    try:
        lib = ctypes.CDLL(_SO)
    except OSError:
        return None

    lib.points_in_mesh.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8)]
    lib.nearest_index.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32)]
    _lib = lib
    return _lib


def _dptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def points_in_mesh(tris, pts):
    """tris (T,3,3) float64, pts (N,3) float64 -> bool (N,). None if no lib."""
    lib = load()
    if lib is None:
        return None
    tris = np.ascontiguousarray(tris, dtype=np.float64)
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    out = np.empty(len(pts), dtype=np.uint8)
    lib.points_in_mesh(_dptr(tris), len(tris), _dptr(pts), len(pts),
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out.astype(bool)


def nearest_index(samples, queries):
    """For each query the index of the nearest sample. None if no lib."""
    lib = load()
    if lib is None:
        return None
    samples = np.ascontiguousarray(samples, dtype=np.float64)
    queries = np.ascontiguousarray(queries, dtype=np.float64)
    out = np.empty(len(queries), dtype=np.int32)
    lib.nearest_index(_dptr(samples), len(samples), _dptr(queries),
                      len(queries),
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    return out
