#!/usr/bin/env python3
"""Step D1 of o2cal.md: single-view vs multi-view error decomposition.

Compares a grid-path dump against an analytic-path dump of the SAME pass-1 pair
batch (same cams, same merge) and reports the five D1 statistics, including the
decisive oracle-weight test:

    alpha_cf_oracle = sum_v den_v_grid * alpha_v_cf / sum_v den_v_grid

If |alpha_cf_oracle - alpha_multi_grid| collapses to ~0 while the equal-weight
analytic version is far off, H2 (implicit cross-view weight mismatch) is proven.
"""
from __future__ import annotations

import sys

import torch


def spearman(a, b):
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-12))


def eig_from_s2(S):
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    tr = a + c
    rad = torch.sqrt(((a - c) ** 2 + 4 * b * b).clamp(min=0.0))
    lmax = (0.5 * (tr + rad)).clamp(min=1e-18)
    det = (a * c - b * b).clamp(min=1e-18)
    lmin = det / lmax
    return lmax, lmin, det


def main(dg_path, dc_path):
    dg = torch.load(dg_path, map_location="cpu")
    dc = torch.load(dc_path, map_location="cpu")
    print(f"grid dump: {dg['solver']}  alpha_v {tuple(dg['num_v'].shape)}")
    print(f"cf   dump: {dc['solver']}  alpha_v {tuple(dc['alpha_v'].shape)}")

    den_v = dg["den_v"]
    num_v = dg["num_v"]
    valid_g = dg["valid"]
    valid_c = dc["valid"]
    av_grid = torch.where(den_v > 0, num_v / den_v.clamp(min=1e-12),
                          torch.zeros_like(den_v))
    av_cf = dc["alpha_v"]
    n = min(av_grid.shape[0], av_cf.shape[0])
    den_v, av_grid, av_cf = den_v[:n], av_grid[:n], av_cf[:n]
    valid_g, valid_c = valid_g[:n], valid_c[:n]
    Sm = dg["Sm"][:n]
    am_g, am_c = dg["alpha_multi"][:n], dc["alpha_multi"][:n]

    # ---- 1. mask agreement ------------------------------------------------
    mism = (valid_g != valid_c).float().mean().item()
    drop = ((valid_g & (den_v <= 1e-12)).float().sum()
            / max(1, int(valid_g.sum()))).item()
    print(f"\n[D1-1] mask mismatch = {100*mism:.3f}%   "
          f"grid-valid-but-den<=eps = {100*drop:.3f}%")

    both = valid_g & valid_c & (den_v > 0)
    print(f"        both-valid (pair,view) = {int(both.sum())}")

    # ---- 2. single-view error ---------------------------------------------
    d = (av_cf - av_grid)[both]
    def stats(t, name):
        print(f"        {name:22s} mean={t.mean():+.5f}  "
              f"|.|mean={t.abs().mean():.5f}  med={t.abs().median():.5f}  "
              f"p95={t.abs().quantile(0.95):.5f}  max={t.abs().max():.5f}")
    stats(d, "alpha_cf - alpha_grid")
    mae_single = d.abs().mean().item()

    # ---- 3. multi-view aggregation error ----------------------------------
    dm = (am_c - am_g)
    print()
    stats(dm, "multi cf - grid")

    # ---- 4. ORACLE weight test (decisive) ---------------------------------
    w = (den_v * both.to(den_v.dtype))
    wsum = w.sum(1)
    oracle = torch.where(wsum > 0, (w * av_cf).sum(1) / wsum.clamp(min=1e-12),
                         torch.zeros_like(wsum))
    ok = wsum > 0
    do = (oracle - am_g)[ok]
    print()
    stats(do, "oracle - grid")
    mae_oracle = do.abs().mean().item()

    # ---- 5. what does den_v correlate with? -------------------------------
    lmax, lmin, det = eig_from_s2(Sm)
    area = torch.sqrt(det.clamp(min=0.0))
    ecc_inv = torch.sqrt((lmin / lmax).clamp(min=0.0))
    sel = both
    dv = den_v[sel]
    print(f"\n[D1-5] den_v vs geometry (Spearman, n={int(sel.sum())}):")
    for name, x in (("const(1)", torch.ones_like(dv)),
                    ("sqrt(det) area", area[sel]),
                    ("1/kappa", ecc_inv[sel]),
                    ("1/lambda_max", 1.0 / lmax[sel])):
        if x.std() < 1e-12:
            print(f"        {name:16s} n/a (constant)")
            continue
        r = spearman(dv, x)
        ratio = (dv / x.clamp(min=1e-30))
        print(f"        {name:16s} rho={r:+.4f}   ratio cv="
              f"{float(ratio.std()/ratio.mean().clamp(min=1e-12)):.4f}")

    # ---- verdict -----------------------------------------------------------
    print(f"\n[D1] single-view MAE = {mae_single:.5f}   "
          f"multi MAE = {dm.abs().mean().item():.5f}   "
          f"oracle MAE = {mae_oracle:.5f}")
    if mae_single < 0.005 and mae_oracle < 0.005 and dm.abs().mean() > 0.01:
        print("VERDICT: H2 CONFIRMED (cross-view weight mismatch)")
    elif mae_single > 0.01:
        print("VERDICT: single-view error -> H1 (wiring) or H3 (undersampling); run D2")
    else:
        print("VERDICT: inconclusive; inspect the tables above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
