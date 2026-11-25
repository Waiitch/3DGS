#!/usr/bin/env python3
"""Fine-grained scene router: pure-PLY features -> configuration regime.

Features are geometric only (no PSNR, no GT): they come from the same Step 1/2
voxelisation that AutoVirt already performs.
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


def features(sn, dev):
    _, mu, op, sc, q, sh, _ = read_ply_fast(
        f"{M}/{MODEL[sn]}/point_cloud/iteration_30000/point_cloud.ply")
    xyz = torch.as_tensor(mu, device=dev)
    sct = torch.as_tensor(sc, device=dev)
    qt = torch.as_tensor(q, device=dev)
    opt = torch.as_tensor(op, device=dev)
    # pca_traj is only used to surface the elongation feature eta
    _, st = extract_auto_virtual_cams(
        xyz, sct, qt, opt, n_views=8, grid_res=64, n_pos=48, n_dirs=6,
        fov_deg=60.0, cov_k=3, return_stats=True, noblas=True,
        d_lo_frac=0.20, pos_mode="pca_traj")
    n_cav = max(int(st["n_cavity"]), 1)
    n_cav6 = int(st["n_cavity6"])
    n_surf = int(st["n_surface"])
    f = dict(scene=sn, mode=st["mode"], Lmax=float(st["Lmax"]),
             n_cav=n_cav, n_cav6=n_cav6, n_surf=n_surf,
             eta=float(st["pca_eta"]),
             cav6_frac=n_cav6 / n_cav,
             wall_density=n_surf / (n_cav ** (2.0 / 3.0) + 1e-9))
    del xyz, sct, qt, opt
    torch.cuda.empty_cache()
    return f


def route(f):
    """Regime rule (mechanism-driven, not fitted to per-scene PSNR).

    Separation evidence on the six scenes (all pure-PLY features):
      indoor : cav6_frac  room 6.1%  vs playroom 59.8%
      outdoor: eta        train 1.96 vs truck 1.31 / garden 1.43 / bicycle 1.22
    """
    if f["mode"].startswith("indoor"):
        if f["cav6_frac"] >= 0.30:
            return "R2_indoor_clutter", "B3_k1"     # playroom: frontier + 1-coverage
        return "R1_indoor_box", "V2p"               # room: gravity ring, dlo 0.05
    if f["eta"] >= 1.6 or f["wall_density"] > 500.0:
        return "R3_outdoor_corridor", "B3"          # train: k=3 along the long axis
    return "R4_outdoor_compact", "B3_k1"            # truck/garden/bicycle: 1-coverage


def main():
    scenes = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODEL)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    rows = []
    print(f"{'scene':9s} {'mode':8s} {'Lmax':>7s} {'n_cav':>7s} {'cav6':>7s} "
          f"{'cav6%':>6s} {'eta':>5s} {'wall':>6s}  ->  regime / config")
    for sn in scenes:
        f = features(sn, dev)
        reg, cfg = route(f)
        rows.append(dict(f, regime=reg, config=cfg))
        print(f"{sn:9s} {f['mode']:8s} {f['Lmax']:7.2f} {f['n_cav']:7d} "
              f"{f['n_cav6']:7d} {100*f['cav6_frac']:5.1f}% {f['eta']:5.2f} "
              f"{f['wall_density']:6.2f}  ->  {reg} / {cfg}")
    with open("timing/av_v3_router.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n-> timing/av_v3_router.json")


if __name__ == "__main__":
    main()
