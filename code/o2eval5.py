"""Plan §15: the last two global camera-set rules (G6, G8) + the H upper bound.

  G6  trajectory-stratified medoids: split the fit list into B equal bins along
      the shooting order and take the camera closest to each bin's centroid.
  G8  global set with track redundancy: greedy over the pilot pairs,
      score(c) = cov(c)/max(cov) - lam * max_{s in S} M[c,s]
      where cov(c) = number of pilot pairs for which c passes the support gate.
  H   all supported views: by construction a_p == a_p^all, so E_alpha == 0 and
      the only meaningful measurement is end-to-end (O2 cost ~ |C+|/8 times),
      which is reported as an estimate instead of run.

Usage:
  TMPDIR=/mnt/home1/wyq/tmp python o2eval5.py --scene ... --o2dump ... --tag truck
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pics                                              # noqa: E402
import cams_track                                        # noqa: E402
from cams_fast import load_cams_fast, pick_cover_views    # noqa: E402

ALPHA_FLOOR = 0.05
EP = 1e-9


def g6_medoids(pool, B):
    """One camera per trajectory bin, closest to the bin centroid."""
    C = pool.C.detach().cpu().numpy().astype(np.float64)
    V = pool.V
    bins = np.minimum((np.arange(V) * B) // V, B - 1)
    out = []
    for b in range(B):
        idx = np.nonzero(bins == b)[0]
        cen = C[idx].mean(0)
        out.append(int(idx[np.argmin(((C[idx] - cen) ** 2).sum(1))]))
    return out


def g7_directional(pool, q, cand, B, lam=1.0):
    """CONVERGE-style directional coverage: spread the selected cameras in
    viewing-direction space while keeping pair coverage (plan 15 G7)."""
    C = pool.C.detach().cpu().numpy().astype(np.float64)
    cen = C.mean(0)
    dirs = C - cen
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    cov = cand.sum(0).astype(np.float64)
    covn = cov / max(cov.max(), 1.0)
    sel = []
    red = np.zeros(pool.V)
    for _ in range(B):
        sc = covn - lam * red
        sc[sel] = -1e9
        c = int(np.argmax(sc))
        sel.append(c)
        red = np.maximum(red, np.abs(dirs @ dirs[c]))
    return sel


def g8_track_global(q, cand, M, B, lam=0.2):
    """Global set: coverage of pilot pairs minus track redundancy (plan 15 G8)."""
    cov = cand.sum(0).astype(np.float64)
    covn = cov / max(cov.max(), 1.0)
    sel = []
    red = np.zeros(q.shape[1])
    for _ in range(B):
        sc = covn - lam * red
        sc[sel] = -1e9
        c = int(np.argmax(sc))
        sel.append(c)
        red = np.maximum(red, M[c])
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--o2dump", required=True)
    ap.add_argument("--tag", default="scene")
    ap.add_argument("--pilot", type=int, default=8192)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--sigma-cut", type=float, default=1.25)
    ap.add_argument("--qmin", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    t0 = time.perf_counter()
    z = np.load(args.o2dump)
    state = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
             for k in ("mu", "sc", "q", "op")}
    comp = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
            for k in ("mu_m", "sc_m", "q_m")}
    pairs = torch.as_tensor(np.ascontiguousarray(z["pairs"]).astype(np.int64),
                            device=dev)
    P = int(pairs.shape[0])
    all_cams = load_cams_fast(args.scene, "fit", 0)
    fast_cams = load_cams_fast(args.scene, "fit", args.B)
    pool = pics.CamPool(all_cams, dev)
    n2i = {n: i for i, n in enumerate(pool.names)}
    fast_ids = [n2i[c["name"]] for c in fast_cams]
    M, _ = cams_track.load_overlap(args.scene, pool)
    n_pil = min(args.pilot, P)
    pil = torch.linspace(0, P - 1, n_pil).round().to(torch.int64).to(dev)
    sub = {k: v[pil] for k, v in comp.items()}
    scr = pool.screen(comp["mu_m"], comp["sc_m"], sigma_cut=2.0, want_area=False)
    q = scr["q"][pil].to(torch.float32).cpu().numpy()
    del scr
    pid = torch.arange(n_pil, device=dev).repeat_interleave(pool.V)
    cid = torch.arange(pool.V, device=dev).repeat(n_pil)
    st = pool.patch_stats(state, sub, pairs[pil], pid, cid, grid=args.grid,
                          sigma_cut=args.sigma_cut)
    d = st["d"].cpu().numpy().reshape(n_pil, pool.V)
    n = st["n"].cpu().numpy().reshape(n_pil, pool.V)
    t2 = st["t2"].cpu().numpy().reshape(n_pil, pool.V)
    OK = st["ok"].cpu().numpy().reshape(n_pil, pool.V)
    a0 = st["a0"].cpu().numpy().reshape(n_pil, pool.V)[:, 0]
    del st, pid, cid
    torch.cuda.empty_cache()
    cand = OK & (q >= args.qmin)
    Dm, Nm, T2m = np.where(OK, d, 0.0), np.where(OK, n, 0.0), np.where(OK, t2, 0.0)
    a_all = np.where((Dm.sum(1) > EP), Nm.sum(1) / np.maximum(Dm.sum(1), EP), a0)

    def ev(ids):
        S = np.zeros((n_pil, pool.V), bool)
        S[:, np.asarray(ids, np.int64)] = True
        S &= OK
        cnt = S.sum(1)
        am = np.where(cnt > 0, (Nm * S).sum(1) / np.maximum((Dm * S).sum(1), EP), a0)
        rel = np.abs(am - a_all) / np.maximum(np.abs(a_all), ALPHA_FLOOR)
        return {"E_alpha": float(rel.mean()), "views": float(cnt.mean()),
                "zero": float((cnt == 0).mean())}

    res = {"tag": args.tag, "B": args.B, "pilot": n_pil, "sigma_cut": args.sigma_cut}
    res["G0_fast"] = ev(fast_ids)
    res["G2_cover"] = ev(pick_cover_views(all_cams, state["mu"].cpu().numpy(), k=args.B))
    res["G6_medoids"] = ev(g6_medoids(pool, args.B))
    for lam in (0.3, 0.6, 1.0):
        res[f"G7_dirlam{lam}"] = ev(g7_directional(pool, q, cand, args.B, lam))
    for lam in (0.1, 0.2, 0.4):
        res[f"G8_track_lam{lam}"] = ev(g8_track_global(q, cand, M.astype(np.float64),
                                                      args.B, lam))
    res["H_allcand"] = ev(list(range(pool.V)))     # == a_all by construction
    res["cand_per_pair"] = float(cand.sum(1).mean())
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"[{args.tag}] {k:18s} E_a={v['E_alpha']:.4f}  "
                  f"views={v['views']:.2f}  zero={v['zero']*100:.1f}%", flush=True)
    print(f"[{args.tag}] cand/pair={res['cand_per_pair']:.1f}  "
          f"(H = all candidates -> E_alpha is 0 by construction)", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[{args.tag}] wrote {args.out} ({res['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
