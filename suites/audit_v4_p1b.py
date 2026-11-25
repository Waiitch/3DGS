#!/usr/bin/env python3
"""P1-B: why does T0-C3 win big on bicycle?  Print the regime that each scene
actually lands in, plus the distance distribution of the 8 chosen cameras.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fastio import read_ply_fast                          # noqa: E402
from o2_auto_virt import extract_auto_virtual_cams        # noqa: E402

M = "/mnt/home1/wyq/3dgs/mpl_las/out"
MODEL = {"truck": "truck_vanilla", "room": "room_vanilla",
         "garden": "garden_vanilla", "bicycle": "bicycle_vanilla",
         "playroom": "playroom_vanilla", "train": "tandt_train_vanilla",
         "drjohnson": "drjohnson_vanilla"}
C3 = dict(core_aabb=0.80, use_scale_gate=False, cover_target="gauss",
          n_gauss_probe=20000)
C1 = dict(core_aabb=0.80, use_scale_gate=False)


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"{'scene':9s} {'cfg':4s} {'mode':18s} {'cav':>6s} {'cav6':>6s} "
          f"{'Lmax':>7s} {'dTgt':>7s}  camera distances to centroid (min/med/max)")
    rows = []
    for sn, md in MODEL.items():
        _, mu, op, sc, q, sh, _ = read_ply_fast(
            f"{M}/{md}/point_cloud/iteration_30000/point_cloud.ply")
        xyz = torch.as_tensor(mu, device=dev)
        sct = torch.as_tensor(sc, device=dev)
        qt = torch.as_tensor(q, device=dev)
        opt = torch.as_tensor(op, device=dev)
        for tag, extra in (("C1", C1), ("C3", C3)):
            k = dict(n_views=8, grid_res=64, n_pos=48, n_dirs=6, fov_deg=60.0,
                     cov_k=3, return_stats=True, noblas=True, d_lo_frac=0.20)
            k.update(extra)
            cams, st = extract_auto_virtual_cams(xyz, sct, qt, opt, **k)
            C = np.stack([np.asarray(c["C"]) for c in cams])
            ctr = C.mean(0)
            d = np.linalg.norm(C - ctr[None, :], axis=1)
            # distance to the point-cloud core centroid instead
            P = mu[:: max(1, mu.shape[0] // 50000)]
            core = P.mean(0)
            dc = np.linalg.norm(C - core[None, :], axis=1)
            print(f"{sn:9s} {tag:4s} {st['mode'][:18]:18s} {st['n_cavity']:6d} "
                  f"{st['n_cavity6']:6d} {st['Lmax']:7.2f} {st['d_target']:7.3f}  "
                  f"{dc.min():6.2f}/{np.median(dc):6.2f}/{dc.max():6.2f}")
            rows.append(dict(scene=sn, cfg=tag, mode=st["mode"],
                             cav6=int(st["n_cavity6"]), Lmax=float(st["Lmax"]),
                             d_target=float(st["d_target"]),
                             d_min=float(dc.min()), d_med=float(np.median(dc)),
                             d_max=float(dc.max())))
        del xyz, sct, qt, opt
        torch.cuda.empty_cache()
    with open("timing/av_v4_p1b_audit.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n-> timing/av_v4_p1b_audit.json")


if __name__ == "__main__":
    main()
