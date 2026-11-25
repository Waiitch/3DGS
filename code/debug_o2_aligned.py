#!/usr/bin/env python3
"""Stage 1 of o2cal_1.md: aligned dual-path audit from ONE dump.

Both paths consume the exact same tensors, so every statistic below is paired
and trustworthy (no cross-run batch misalignment).
"""
from __future__ import annotations

import sys

import torch

from o2_analytic import o2_closed_form_batch


def quant(name, x):
    x = x.to(torch.float64).flatten()
    if x.numel() == 0:
        print(f"[{name}] empty")
        return
    q = torch.tensor([0.0, 0.5, 0.95, 0.99, 0.999, 1.0], dtype=torch.float64)
    v = torch.quantile(x, q).tolist()
    print(f"[{name:16s}] min={v[0]:+.4g} p50={v[1]:+.4g} p95={v[2]:+.4g} "
          f"p99={v[3]:+.4g} p999={v[4]:+.4g} max={v[5]:+.4g} | "
          f"<0:{100*float((x < 0).float().mean()):.3f}%  "
          f">1:{100*float((x > 1).float().mean()):.3f}%  "
          f">4:{100*float((x > 4).float().mean()):.4f}%")


def eig(S):
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    tr = a + c
    rad = torch.sqrt(((a - c) ** 2 + 4 * b * b).clamp(min=0.0))
    lmax = (0.5 * (tr + rad)).clamp(min=1e-30)
    lmin = (0.5 * (tr - rad)).clamp(min=1e-30)
    return lmax, lmin


def main(path):
    d = torch.load(path, map_location="cpu")
    valid = d["valid"].bool()
    Si, Sj, Sm = (d[k].to(torch.float64) for k in ("Si", "Sj", "Sm"))
    ui, uj, um = (d[k].to(torch.float64) for k in ("ui", "uj", "um"))
    ai, aj = d["a_i"].to(torch.float64), d["a_j"].to(torch.float64)

    # ---- both paths on the SAME tensors ----
    agg_cf, _, aux = o2_closed_form_batch(ui, uj, um, Si, Sj, Sm, ai, aj, valid,
                                          beta=0.0, agg="eq")
    ai_v, aj_v = ai.view(-1, 1), aj.view(-1, 1)
    cf_v = ai_v * aux["c_im"] + aj_v * aux["c_jm"] - (ai_v * aj_v) * aux["c_ijm"]

    den_v, num_v = d["den_v"].to(torch.float64), d["num_v"].to(torch.float64)
    ok = valid & (den_v > 1e-12)
    grid_v = torch.where(ok, num_v / den_v.clamp(min=1e-12),
                         torch.zeros_like(num_v))

    print(f"valid = {100*float(valid.float().mean()):.2f}%   "
          f"ok(den>0) = {int(ok.sum())}")
    quant("grid_v", grid_v[ok])
    quant("cf_v", cf_v[ok])
    quant("grid_agg", d["alpha_multi"].to(torch.float64))
    quant("cf_agg", agg_cf)

    # ---- degeneracy audit ----
    lmax_m, lmin_m = eig(Sm)
    lmax_i, _ = eig(Si)
    lmax_j, _ = eig(Sj)
    over = (lmax_m > 1e6) | (lmax_i > 1e6) | (lmax_j > 1e6)
    print(f"\noverflow (lmax>1e6) among ok: "
          f"{100*float((ok & over).float().sum()/ok.float().sum()):.4f}%  "
          f"n={int((ok & over).sum())}")
    if int((ok & over).sum()):
        quant("cf_v(overflow)", cf_v[ok & over])
        quant("grid_v(overflow)", grid_v[ok & over])
    clean = ok & (~over)

    # ---- aligned single-view error + kappa buckets ----
    dv = cf_v - grid_v
    print(f"\n[aligned clean] signed={float(dv[clean].mean()):+.6f}  "
          f"MAE={float(dv[clean].abs().mean()):.6f}")
    kappa = torch.sqrt((lmax_m / lmin_m).clamp(min=1e-30))
    for lo, hi in ((0.0, 2.0), (2.0, 5.0), (5.0, 1e9)):
        m = clean & (kappa >= lo) & (kappa < hi)
        if int(m.sum()):
            print(f"  kappa [{lo:>4},{hi:>6}): n={int(m.sum()):>7d}  "
                  f"signed={float(dv[m].mean()):+.6f}  "
                  f"MAE={float(dv[m].abs().mean()):.6f}")

    # ---- oracle den_v aggregation ----
    w = torch.where(clean, den_v, torch.zeros_like(den_v))
    ws = w.sum(1)
    pk = ws > 1e-12
    orc = (cf_v * w).sum(1) / ws.clamp(min=1e-12)
    grid_clean = (num_v * clean.to(num_v.dtype)).sum(1) / ws.clamp(min=1e-12)
    e = (orc - grid_clean)[pk]
    print(f"\n[oracle den_v] n={int(pk.sum())}  signed={float(e.mean()):+.6f}  "
          f"MAE={float(e.abs().mean()):.6f}")

    # ---- does clamping c / alpha fix the outliers? ----
    agg2, _, _ = o2_closed_form_batch(ui, uj, um, Si, Sj, Sm, ai, aj, valid,
                                      beta=0.0, agg="eq", clamp_c=True)
    print(f"\n[cf with clamp_c] am_mean={float(agg2.mean()):.6f} "
          f"(no clamp {float(agg_cf.mean()):.6f})")
    agg3, _, _ = o2_closed_form_batch(ui, uj, um, Si, Sj, Sm, ai, aj, valid,
                                      beta=0.0, agg="eq", clamp_c=True,
                                      safe_mask=True)
    print(f"[cf clamp_c+safe] am_mean={float(agg3.mean()):.6f}")
    print(f"[grid am_mean]    {float(d['alpha_multi'].to(torch.float64).mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
