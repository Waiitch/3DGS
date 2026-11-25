#!/usr/bin/env python3
"""Why is the closed form slower on playroom?  Micro-benchmark the solver on the
two dumps with IDENTICAL tensor shapes to separate size effects from data effects.
"""
from __future__ import annotations

import sys
import time

import torch

from o2_analytic import o2_closed_form_batch


def bench(d, iters=20):
    ui, uj, um = (d[k].double() for k in ("ui", "uj", "um"))
    Si, Sj, Sm = (d[k].double() for k in ("Si", "Sj", "Sm"))
    ai, aj = d["a_i"].double(), d["a_j"].double()
    valid = d["valid"].bool()
    # warm up
    for _ in range(3):
        o2_closed_form_batch(ui, uj, um, Si, Sj, Sm, ai, aj, valid,
                             beta=0.0, agg="eq")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o2_closed_form_batch(ui, uj, um, Si, Sj, Sm, ai, aj, valid,
                             beta=0.0, agg="eq")
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3      # ms per call


def stats(d):
    Sm = d["Sm"].double()
    a, b, c = Sm[..., 0, 0], Sm[..., 0, 1], Sm[..., 1, 1]
    det = a * c - b * b
    tr = a + c
    rad = torch.sqrt(((a - c) ** 2 + 4 * b * b).clamp(min=0.0))
    lmax = 0.5 * (tr + rad)
    v = d["valid"].bool()
    return dict(
        lmax_max=float(lmax[v].max()),
        det_min=float(det[v].min()),
        det_neg_frac=float((det[v] <= 0).float().mean()),
        huge_frac=float((lmax[v] > 1e6).float().mean()),
    )


def main(paths):
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"{'scene':10s} {'shape(B,V)':>12s} {'ms/call':>8s} {'lmax_max':>10s} "
          f"{'det_min':>10s} {'det<=0':>7s} {'lmax>1e6':>9s}")
    out = {}
    for p in paths:
        d = torch.load(p, map_location="cpu")
        name = p.split("_")[1] if "_" in p else p
        for k in list(d.keys()):
            if torch.is_tensor(d[k]) and d[k].dtype != torch.bool:
                d[k] = d[k].to(dev)
            elif torch.is_tensor(d[k]):
                d[k] = d[k].to(dev)
        s = stats(d)
        ms = bench(d, iters=10)
        out[name] = ms
        print(f"{name:10s} {str(tuple(d['ui'].shape)):>12s} {ms:8.2f} "
              f"{s['lmax_max']:10.3e} {s['det_min']:10.3e} "
              f"{100*s['det_neg_frac']:6.2f}% {100*s['huge_frac']:8.3f}%")
        del d
        torch.cuda.empty_cache()
    if len(out) >= 2:
        a, b = list(out.values())[:2]
        print(f"\nratio = {b/a:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
