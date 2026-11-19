"""Fast COLMAP camera loader for the O2 fit views.

`scene.colmap_loader.read_extrinsics_binary` materialises every 2D observation of
every image (251 images x ~10k points here) although O2 only needs the pose and
the intrinsics; that parse costs ~1.2 s of the pipeline's wall clock.

This module parses the same two binaries with `struct`, skips the 2D point
arrays, and then calls the *same* `qvec2rotmat` / `getWorld2View2` helpers, so
the resulting `w2v / fx / fy / W / H / C` are bit-identical. `verify_cams.py`
checks exactly that against `ops2.load_cams`.
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

# the COLMAP reader / camera convention live in the 3DGS repo
_GS = os.environ.get("NANOGS_GS_REPO", "/mnt/home1/wyq/3dgs/gaussian-splatting-original")
if _GS not in sys.path:
    sys.path.insert(0, _GS)

COLMAP_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_images_bin(path):
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", f.read(4))[0]                  # image_id
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00" or not ch:
                    break
                name += ch
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(npts * 24, 1)                                # skip xys + ids
            out.append((name.decode(), qvec, tvec, cam_id))
    return out


def _read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            name, nparams = COLMAP_MODELS[model_id]
            params = struct.unpack("<%dd" % nparams, f.read(8 * nparams))
            cams[cam_id] = dict(model=name, width=width, height=height,
                                params=list(params))
    return cams


def load_cams_fast(scene, which="fit", max_views=8, val_every=4):
    """Same selection + same numbers as ops2.split_names / ops2.load_cams."""
    from scene.colmap_loader import qvec2rotmat
    from utils.graphics_utils import getWorld2View2

    base = os.path.join(scene, "sparse", "0")
    recs = _read_images_bin(os.path.join(base, "images.bin"))
    intr = _read_cameras_bin(os.path.join(base, "cameras.bin"))
    names = sorted(r[0] for r in recs)
    test = [n for i, n in enumerate(names) if i % 8 == 0]
    train = [n for i, n in enumerate(names) if i % 8 != 0]
    val = [n for i, n in enumerate(train) if i % val_every == 0]
    fit = [n for i, n in enumerate(train) if i % val_every != 0]
    sel = {"fit": fit, "val": val, "test": test}[which]
    if max_views and len(sel) > max_views:
        step = len(sel) / max_views
        sel = [sel[int(i * step)] for i in range(max_views)]

    by = {r[0]: r for r in recs}
    cams = []
    for nm in sel:
        _, qvec, tvec, cam_id = by[nm]
        cam = intr[cam_id]
        if cam["model"] == "SIMPLE_PINHOLE":
            fx = fy = float(cam["params"][0])
        else:
            fx, fy = float(cam["params"][0]), float(cam["params"][1])
        R = np.transpose(qvec2rotmat(np.asarray(qvec)))
        T = np.asarray(tvec, dtype=np.float64)
        cams.append(dict(name=nm, w2v=getWorld2View2(R, T).astype(np.float64),
                         fx=fx, fy=fy, W=int(cam["width"]), H=int(cam["height"]),
                         C=(-R @ T).astype(np.float64)))
    return cams

def pick_cover_views(cams, pts, k=8, n_sub=30000, seed=0, min_deg=40.0):
    """Greedy max-coverage view selection (plan 13.4).

    The default `fit` split is an even subsample along the file name order, which
    leaves ~15-19% of merged pairs with zero support.  Here we instead pick the k
    views that cover the most points, which costs nothing at run time.
    """
    import numpy as np
    P = np.asarray(pts, dtype=np.float64)
    if len(P) > n_sub:
        rs = np.random.default_rng(seed)
        P = P[rs.choice(len(P), n_sub, replace=False)]
    inside = []
    for c in cams:
        M = c["w2v"].astype(np.float64)
        pc = P @ M[:3, :3].T + M[:3, 3]
        z = pc[:, 2]
        ok = z > 1e-3
        zz = np.where(ok, z, 1.0)
        px = c["fx"] * pc[:, 0] / zz + c["W"] / 2
        py = c["fy"] * pc[:, 1] / zz + c["H"] / 2
        inside.append(ok & (px >= 0) & (px < c["W"]) & (py >= 0) & (py < c["H"]))
    inside = np.stack(inside)                     # (V, n)
    # camera positions, for an angular-diversity constraint: pure greedy coverage
    # otherwise picks neighbouring cameras (garden: indices 2..20), which fixes
    # support but destroys the conditioning of the opacity fit
    C = np.stack([-c["w2v"][:3, :3].T @ c["w2v"][:3, 3] for c in cams])
    ctr = C.mean(0)
    dirs = C - ctr
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    covered = np.zeros(inside.shape[1], bool)
    chosen = []
    for _ in range(min(k, len(cams))):
        gain = (inside & ~covered[None, :]).sum(1).astype(np.float64)
        gain[chosen] = -1.0
        if chosen:
            cosang = dirs @ dirs[chosen].T            # (V, |chosen|)
            maxcos = cosang.max(1)
            limit = np.cos(np.radians(min_deg))
            allowed = maxcos <= limit
            if allowed.any():                         # hard, with relaxation
                gain[~allowed] = -1.0
        i = int(gain.argmax())
        chosen.append(i)
        covered |= inside[i]
    return chosen