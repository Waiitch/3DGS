"""Verify the per-pair camera routing of the PICS kernel.

`bench_o2kern.py` checks the kernel when every pair uses the *same* 8 views, and
V1 in the report checks it end-to-end with a global cover8 set.  Neither would
catch a pid<->pair mapping bug, which is the one structural difference of the
pair-conditioned path.  Here every pair gets its own 4-camera set and the result
is compared against running the reference kernel on that pair alone.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pics                                              # noqa: E402
from cams_fast import load_cams_fast                      # noqa: E402
from o2gpu import O2Cameras, opacity_o2cal_gpu            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--o2dump", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(0)
    np.random.seed(0)

    z = np.load(args.o2dump)
    state = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
             for k in ("mu", "sc", "q", "op")}
    comp = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
            for k in ("mu_m", "sc_m", "q_m")}
    pairs_all = np.ascontiguousarray(z["pairs"]).astype(np.int64)
    allc = load_cams_fast(args.scene, "fit", 0)
    pool = pics.CamPool(allc, dev)

    n = args.n
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(len(pairs_all), n, replace=False))
    pairs = torch.as_tensor(pairs_all[idx], device=dev)
    sub = {k: v[torch.as_tensor(idx, device=dev)] for k, v in comp.items()}
    # a different camera set for every pair
    sel_np = np.stack([rng.choice(pool.V, args.k, replace=False) for _ in range(n)])
    sel = torch.as_tensor(sel_np, device=dev, dtype=torch.int32)

    am_pool = pics.opacity_o2cal_gpu_pool(state, sub, pairs, sel, pool, grid=9,
                                          sigma_cut=2.0, calibrate=False)
    am_ref = torch.zeros(n, dtype=torch.float32, device=dev)
    for p in range(n):
        one_pairs = pairs[p:p + 1]
        one_comp = {k: v[p:p + 1] for k, v in sub.items()}
        cams = [allc[int(c)] for c in sel_np[p]]
        am_ref[p] = opacity_o2cal_gpu(state, one_comp, one_pairs,
                                      O2Cameras(cams, dev), grid=9,
                                      sigma_cut=2.0, calibrate=False, batch=1,
                                      noblas=True)[0]
    dif = (am_pool - am_ref).abs()
    print(f"pairs {n} x {args.k} distinct cameras: max|pool-ref| {float(dif.max()):.3e}  "
          f"mean {float(dif.mean()):.3e}")
    # also confirm that swapping the rows really changes the answer (i.e. the
    # comparison is not vacuous)
    am_swap = pics.opacity_o2cal_gpu_pool(
        state, sub, pairs, torch.roll(sel, 1, 0), pool, grid=9, sigma_cut=2.0,
        calibrate=False)
    print(f"control: row-shifted camera sets differ by "
          f"{float((am_pool - am_swap).abs().mean()):.3e} on average")


if __name__ == "__main__":
    main()
