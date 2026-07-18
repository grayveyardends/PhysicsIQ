"""gguf_meta.py — model detection: reads the metadata HEADER of .gguf
files (no weights are loaded) and answers two questions:

  1. what IS this file — a text model, or a vision projector (mmproj)?
  2. which model + mmproj pairs actually FIT together?

Why not filenames or a model_config.json: every gguf already carries its
own metadata in the first few KB of the file — architecture, embedding
sizes, projector type. These are the exact fields llama-server reads at
load time, so pairing from them can never disagree with what the server
will accept. A side-car json would just be a hand-maintained copy of
data the files already contain (and it would drift the moment you drop
a new gguf in).

The pairing rule is the same check llama-server enforces:

    text model  "{arch}.embedding_length"  ==  mmproj  "clip.vision.projection_dim"

When it's violated you get the exact error this module exists to prevent:
    "mtmd_init_from_file: mismatch between text model (n_embd = 4096)
     and mmproj (n_embd = 1536)"

I/O cost: we parse only key-values at the start of the file and SEEK
over big arrays (tokenizer tables), so inspecting a 5 GB gguf takes
milliseconds. Results are cached per (path, size, mtime).
"""

import os
import struct

# gguf value types -> (struct format, byte size). Strings and arrays are
# handled separately below. Spec: github.com/ggml-org/ggml/docs/gguf.md
_SCALAR = {
    0: ("<B", 1),   # uint8
    1: ("<b", 1),   # int8
    2: ("<H", 2),   # uint16
    3: ("<h", 2),   # int16
    4: ("<I", 4),   # uint32
    5: ("<i", 4),   # int32
    6: ("<f", 4),   # float32
    7: ("<?", 1),   # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_STRING, _ARRAY = 8, 9


def _read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype):
    """Read one metadata value. Arrays return None — the only big ones
    are tokenizer tables and we never need them, so we skip their bytes
    instead of loading megabytes into memory."""
    if vtype in _SCALAR:
        fmt, size = _SCALAR[vtype]
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in _SCALAR:
            f.seek(_SCALAR[etype][1] * count, os.SEEK_CUR)
        elif etype == _STRING:
            for _ in range(count):
                (n,) = struct.unpack("<Q", f.read(8))
                f.seek(n, os.SEEK_CUR)
        else:
            raise ValueError(f"gguf array of unknown type {etype}")
        return None
    raise ValueError(f"gguf value of unknown type {vtype}")


def read_metadata(path):
    """All scalar/string header key-values as a plain dict.
    Raises ValueError if the file is not a gguf."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a gguf file")
        _version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        meta = {}
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            value = _read_value(f, vtype)
            if value is not None:
                meta[key] = value
        return meta


# classification + pairing

_CACHE = {}  # (path, size, mtime) -> info dict, so refresh() stays instant


def inspect(path):
    """One flat summary per file:
        {"kind": "model"|"mmproj", "arch": str, "n_embd": int|None,
         "name": str, "path": str}
    n_embd is the pairing key: embedding_length for text models,
    clip.vision.projection_dim for mmproj files. Unreadable files come
    back with n_embd=None, which pairs with nothing (safe default)."""
    st = os.stat(path)
    cache_key = (path, st.st_size, st.st_mtime)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    base = os.path.basename(path)
    info = {"kind": "model", "arch": "", "n_embd": None,
            "name": base.removesuffix(".gguf"), "path": path}
    try:
        meta = read_metadata(path)
        arch = meta.get("general.architecture", "")
        info["arch"] = arch
        info["name"] = meta.get("general.name", info["name"])
        # mmproj files say so themselves: general.type == "mmproj"
        # (older conversions only have arch == "clip" / the clip.* keys)
        if (meta.get("general.type") == "mmproj" or arch == "clip"
                or "clip.has_vision_encoder" in meta):
            info["kind"] = "mmproj"
            info["n_embd"] = meta.get("clip.vision.projection_dim")
        else:
            info["n_embd"] = meta.get(f"{arch}.embedding_length")
    except (ValueError, OSError, struct.error):
        # Corrupt / truncated / not really a gguf: fall back to the
        # filename convention so the UI still lists it instead of dying.
        if base.startswith("mmproj-"):
            info["kind"] = "mmproj"

    _CACHE[cache_key] = info
    return info


def _name_overlap(model_file, mmproj_file):
    """Tie-break when several mmprojs FIT the same model (same n_embd):
    prefer the one whose filename shares the most words with the model,
    because HF uploads name them after each other
    (gemma-4-E2B-it-Q8_0 <-> mmproj-gemma-4-E2B-it-Q8_0)."""
    def words(s):
        s = os.path.basename(s).lower().removesuffix(".gguf")
        s = s.removeprefix("mmproj-")
        return set(w for w in _split_tokens(s) if w)
    return len(words(model_file) & words(mmproj_file))


def _split_tokens(s):
    out, cur = [], []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            out.append("".join(cur))
            cur = []
    out.append("".join(cur))
    return out


def scan_models(models_dir):
    """Return [(display_name, model_path, mmproj_path_or_None), ...] —
    every text model in the folder, each paired with a COMPATIBLE mmproj
    or None. Text-only models simply get no vision; nothing is guessed."""
    if not os.path.isdir(models_dir):
        return []
    infos = [inspect(os.path.join(models_dir, f))
             for f in sorted(os.listdir(models_dir)) if f.endswith(".gguf")]
    models = [i for i in infos if i["kind"] == "model"]
    mmprojs = [i for i in infos if i["kind"] == "mmproj"]

    result = []
    for m in models:
        fits = [p for p in mmprojs
                if p["n_embd"] is not None and p["n_embd"] == m["n_embd"]]
        best = max(fits, key=lambda p: _name_overlap(m["path"], p["path"]),
                   default=None)
        result.append((
            os.path.basename(m["path"]).removesuffix(".gguf"),
            m["path"],
            best["path"] if best else None,
        ))
    return result
