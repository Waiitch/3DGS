"""Byte-identical, vectorised PLY I/O for 3DGS files (float32 properties).

The official `ply_utils` reads every property through a structured dtype and
writes row-by-row with `struct.pack`, which costs ~11 s of the 117 s CPU
baseline on `truck`.  This module reads/writes exactly the same bytes with a
single contiguous pass; `verify_io.py` checks sha256 equality on both sides.
"""
from __future__ import annotations

import numpy as np

from nanogs.utils.ply_utils import (
    PLY_TYPE_TO_DTYPE, PLY_TYPE_TO_STRUCT, PlyHeader, read_ply,
)
from nanogs.utils.splat_utils import logit, quat_normalize, sigmoid


def read_ply_fast(path):
    """Same return contract as nanogs.utils.ply_utils.read_ply."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("EOF while reading PLY header.")
            header_lines.append(line.decode("ascii", errors="replace").rstrip("\n"))
            if header_lines[-1].strip() == "end_header":
                break

        fmt = None
        vertex_count = None
        props = []
        in_vertex = False
        for ln in header_lines:
            parts = ln.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = (parts[1] == "vertex")
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    continue
                props.append((parts[1], parts[2]))

        if fmt != "binary_little_endian":
            return read_ply(path)          # fall back to the official reader
        if not all(PLY_TYPE_TO_DTYPE[pt] is np.float32 for pt, _ in props):
            return read_ply(path)

        n = int(vertex_count)
        ncol = len(props)
        buf = f.read(n * ncol * 4)
        if len(buf) < n * ncol * 4:
            raise ValueError("Not enough binary vertex bytes.")
        flat = np.frombuffer(buf, dtype="<f4").reshape(n, ncol)

    hdr = PlyHeader("binary_little_endian", n, props)
    name_to_col = {name: k for k, (_, name) in enumerate(props)}

    def block(idxs):
        """Contiguous (n, len(idxs)) copy of the given columns in ONE pass.

        Pulling each column separately (`flat[:, k]`) walks the whole ~486 MB row
        buffer once per column; for the 45 SH columns that costs 2.4 s. A single
        block read is a sequential copy (~0.3 s).
        """
        if len(idxs) == 1:
            return np.ascontiguousarray(flat[:, idxs[0]])[:, None]
        if idxs == list(range(idxs[0], idxs[0] + len(idxs))):
            return np.ascontiguousarray(flat[:, idxs[0]:idxs[-1] + 1])
        return np.ascontiguousarray(flat[:, idxs])

    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"]
    for r in required:
        if r not in name_to_col:
            raise ValueError(f"Missing required property '{r}'.")

    fixed = set(required) | {"x", "y", "z"}
    app_names = [name for pt, name in props
                 if name not in {"nx", "ny", "nz"} and name not in fixed
                 and PLY_TYPE_TO_DTYPE[pt] in (np.float32, np.float64)]

    mu = block([name_to_col[k] for k in ("x", "y", "z")])
    op_raw = block([name_to_col["opacity"]])[:, 0]
    sc_raw = block([name_to_col[k] for k in ("scale_0", "scale_1", "scale_2")])
    q_raw = block([name_to_col[k] for k in ("rot_0", "rot_1", "rot_2", "rot_3")])

    if app_names:
        sh = block([name_to_col[n_] for n_ in app_names])
    else:
        sh = np.zeros((mu.shape[0], 0), dtype=np.float32)

    op = sigmoid(op_raw).astype(np.float32)
    sc = np.exp(np.clip(sc_raw, -30.0, 30.0)).astype(np.float32)
    q = quat_normalize(q_raw).astype(np.float32)
    return hdr, mu, op, sc, q, sh, app_names


def store_ply_fast(out_path, hdr, mu, op, sc, q, sh, app_names):
    """Byte-identical to nanogs.utils.ply_utils.store_ply for float32 headers."""
    props = [(pt, name) for pt, name in hdr.properties]
    if not all(PLY_TYPE_TO_DTYPE[pt] is np.float32 for pt, _ in props):
        from nanogs.utils.ply_utils import store_ply
        return store_ply(out_path, hdr, mu, op, sc, q, sh, app_names)

    op_raw_out = logit(op).astype(np.float32)
    sc_raw_out = np.log(np.maximum(sc, 1e-12)).astype(np.float32)
    app_to_col = {name: idx for idx, name in enumerate(app_names)}

    n = mu.shape[0]
    buf = np.zeros((n, len(props)), dtype=np.float32)
    for k, (pt, name) in enumerate(props):
        if name == "x":
            buf[:, k] = mu[:, 0]
        elif name == "y":
            buf[:, k] = mu[:, 1]
        elif name == "z":
            buf[:, k] = mu[:, 2]
        elif name == "opacity":
            buf[:, k] = op_raw_out
        elif name.startswith("scale_"):
            buf[:, k] = sc_raw_out[:, int(name.split("_")[1])]
        elif name.startswith("rot_"):
            buf[:, k] = q[:, int(name.split("_")[1])]
        elif name in app_to_col:
            buf[:, k] = sh[:, app_to_col[name]]

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for pt, name in props:
        header.append(f"property {pt} {name}")
    header.append("end_header")
    with open(out_path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        buf.tofile(f)
