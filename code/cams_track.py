"""COLMAP track overlap between fit cameras -- the photogrammetric redundancy
proxy of plan 11.6 (MutualNeRF / NeRF Director d_photo).

`points3D.bin` stores, for every 3-D point, the list of images that observe it.
The normalised intersection of two cameras' track sets

    M[c,c'] = |T_c n T_c'| / sqrt(|T_c| |T_c'|)

is the cheapest available proxy for "these two views see the same content", and,
unlike a camera-centre distance or an optical-axis angle, it is not fooled by two
far-apart cameras that look at the same wall.

Usage:
    from cams_track import load_overlap
    M, names = load_overlap(scene, pool)      # (V,V) float32, cached under cache/
"""
from __future__ import annotations

import os
import struct

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _read_points3D_bin(path):
    """Return (xyz (n,3) float32, tracks: list of int32 arrays of image_id)."""
    xyz = []
    tracks = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<Q", f.read(8))[0]          # point3D_id
            xyz.append(struct.unpack("<3d", f.read(24)))
            f.seek(3, 1)                                # rgb
            f.seek(8, 1)                                # reprojection error
            tl = struct.unpack("<Q", f.read(8))[0]
            buf = np.frombuffer(f.read(tl * 8), dtype=np.int32).reshape(tl, 2)
            tracks.append(buf[:, 0].copy())
    return np.asarray(xyz, np.float32), tracks


def _read_image_ids(path):
    """name -> COLMAP image_id (the id used inside points3D tracks)."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            f.seek(32 + 24, 1)                          # qvec, tvec
            struct.unpack("<I", f.read(4))[0]           # camera_id
            name = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00" or not ch:
                    break
                name += ch
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(npts * 24, 1)
            out[name.decode()] = image_id
    return out


def load_overlap(scene, pool, force=False):
    """Normalised track overlap between the pool's cameras. Cached per scene."""
    tag = os.path.basename(os.path.normpath(scene))
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{tag}_trackoverlap.npz")
    if os.path.exists(fp) and not force:
        z = np.load(fp)
        if list(z["names"]) == list(pool.names):
            return z["M"].astype(np.float32), list(pool.names)
    sp = os.path.join(scene, "sparse", "0")
    _, tracks = _read_points3D_bin(os.path.join(sp, "points3D.bin"))
    ids = _read_image_ids(os.path.join(sp, "images.bin"))
    img2idx = {ids[n]: k for k, n in enumerate(pool.names) if n in ids}
    lens = np.fromiter((len(t) for t in tracks), count=len(tracks), dtype=np.int64)
    flat = np.concatenate(tracks) if tracks else np.zeros(0, np.int32)
    pt_of = np.repeat(np.arange(len(tracks), dtype=np.int32), lens)
    lut = np.full(int(flat.max()) + 2 if flat.size else 2, -1, np.int32)
    for img, k in img2idx.items():
        if img < lut.size:
            lut[img] = k
    rows = lut[flat]
    keep = rows >= 0
    rows = rows[keep]
    cols = pt_of[keep]
    from scipy.sparse import coo_matrix
    A = coo_matrix((np.ones(rows.size, np.float32), (rows, cols)),
                   shape=(pool.V, len(tracks))).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    M = (A @ A.T).toarray().astype(np.float64)
    denom = np.sqrt(np.outer(np.maximum(deg, 1.0), np.maximum(deg, 1.0)))
    M = (M / denom).astype(np.float32)
    np.savez_compressed(fp, M=M, names=np.asarray(pool.names))
    return M, list(pool.names)
