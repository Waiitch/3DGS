"""Plan sections 17.4 / 18.2 / 21.1 -- what the earlier offline evaluation did not do.

  17.4  pair-split generalisation: rank the camera rules on one half of the pilot
        pairs and check the ranking survives on the other half (plus the
        held-out-camera residual, already reported as E_heldout).
  18.2  adaptive budget: residual-driven greedy with the plan's stop rule
        (relative improvement < tau, B_min=4, B_max=12) instead of a fixed B=8.
  21.1  selection quality: trajectory-distribution error, visual redundancy
        (mean COLMAP track overlap among the selected cameras), selection
        stability (Jaccard between two pairs' selections), pool coverage.

Everything runs on the pilot of one dumped pass-1 state, so all numbers are
directly comparable with `o2eval2.py` (same d/n/t2/OK arrays).

Usage:
  python o2eval4.py --scene <scene> --o2dump <dump> --tag truck --pilot 8192
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


def residual_greedy(d, n, t2, cand, k, bmin=1, tau=0.0):
    """Same as o2eval2.residual_greedy (kept local so the two files stay独立)."""
    P, V = d.shape
    D = np.where(cand, d, 0.0)
    N = np.where(cand, n, 0.0)
    T2 = np.where(cand, t2, 0.0)
    dtot, ntot, t2tot = D.sum(1), N.sum(1), T2.sum(1)
    sel = np.full((P, k), -1, np.int32)
    num = np.zeros(P); den = np.zeros(P)
    used = np.zeros((P, V), bool)
    budget = np.zeros(P, np.int32)
    rows_all = np.arange(P)
    for step in range(k):
        rem_d = dtot - (D * used).sum(1)
        rem_n = ntot - (N * used).sum(1)
        rem_t2 = t2tot - (T2 * used).sum(1)
        am = np.where(den > EP, num / np.maximum(den, EP), 0.0)
        cur = am ** 2 * rem_d - 2 * am * rem_n + rem_t2
        am2 = (num[:, None] + N) / np.maximum(den[:, None] + D, EP)
        new = (am2 ** 2 * (rem_d[:, None] - D) - 2 * am2 * (rem_n[:, None] - N)
               + (rem_t2[:, None] - T2))
        new = np.where(cand & ~used, new, np.inf)
        i = np.argmin(new, 1)
        best = new[rows_all, i]
        ok = np.isfinite(best)
        if tau > 0.0:
            gain = cur - best
            ok &= (gain / np.maximum(np.abs(cur), EP) > tau) | (budget < bmin)
        sel[ok, step] = i[ok]
        rows = np.nonzero(ok)[0]
        used[rows, i[rows]] = True
        num[rows] += N[rows, i[rows]]
        den[rows] += D[rows, i[rows]]
        budget[rows] += 1
    return sel, budget


def selection_metrics(sel, M, V, nbins=10):
    """Plan 21.1: trajectory distribution error / redundancy / stability / usage."""
    P, B = sel.shape
    good = sel >= 0
    S = np.zeros((P, V), bool)
    ii, jj = np.nonzero(good)
    S[ii, sel[ii, jj]] = True
    cnt = S.sum(1)
    # trajectory-distribution error: L1 between the selected camera-index
    # histogram (mean over pairs) and the uniform pool histogram
    bins = np.minimum((np.arange(V) * nbins) // V, nbins - 1)
    h_pool = np.bincount(bins, minlength=nbins).astype(float)
    h_pool /= h_pool.sum()
    idx_b = bins[sel[good]]
    pair_id = np.repeat(np.arange(P), good.sum(1))
    h_sel = np.zeros(nbins)
    for p in range(P):
        v = idx_b[pair_id == p]
        if v.size:
            h_sel += np.bincount(v, minlength=nbins) / v.size
    h_sel /= max(P, 1)
    traj_err = float(np.abs(h_sel - h_pool).sum())
    # visual redundancy: mean track overlap among the selected cameras
    reds = []
    for p in range(0, P, max(1, P // 2048)):
        c = sel[p][sel[p] >= 0]
        if c.size >= 2:
            sub = M[np.ix_(c, c)]
            iu = np.triu_indices(c.size, 1)
            reds.append(sub[iu].mean())
    redundancy = float(np.mean(reds)) if reds else float("nan")
    # stability: Jaccard between the selections of random pair pairs
    rs = np.random.default_rng(0)
    if P >= 2:
        a = rs.integers(0, P, 512)
        b = rs.integers(0, P, 512)
        jac = []
        for x, y in zip(a, b):
            sx = set(sel[x][sel[x] >= 0].tolist())
            sy = set(sel[y][sel[y] >= 0].tolist())
            if sx and sy:
                jac.append(len(sx & sy) / len(sx | sy))
        stability = float(np.mean(jac)) if jac else float("nan")
    else:
        stability = float("nan")
    used_frac = float((S.any(0)).sum() / V)
    return {"traj_err": traj_err, "redundancy": redundancy,
            "stability_jaccard": stability, "pool_used": used_frac,
            "mean_views": float(cnt.mean()), "zero_frac": float((cnt == 0).mean())}


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
    t_all = time.perf_counter()
    res = {"tag": args.tag, "pilot": args.pilot, "B": args.B,
           "grid": args.grid, "sigma_cut": args.sigma_cut}

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
    sub_pairs = pairs[pil]
    sub_comp = {k: v[pil] for k, v in comp.items()}
    scr = pool.screen(comp["mu_m"], comp["sc_m"], sigma_cut=2.0, want_area=False)
    q = scr["q"][pil].to(torch.float32).cpu().numpy()
    del scr
    torch.cuda.empty_cache()

    pid = torch.arange(n_pil, device=dev).repeat_interleave(pool.V)
    cid = torch.arange(pool.V, device=dev).repeat(n_pil)
    st = pool.patch_stats(state, sub_comp, sub_pairs, pid, cid, grid=args.grid,
                          sigma_cut=args.sigma_cut)
    d = st["d"].cpu().numpy().reshape(n_pil, pool.V)
    n = st["n"].cpu().numpy().reshape(n_pil, pool.V)
    t2 = st["t2"].cpu().numpy().reshape(n_pil, pool.V)
    OK = st["ok"].cpu().numpy().reshape(n_pil, pool.V)
    a0 = st["a0"].cpu().numpy().reshape(n_pil, pool.V)[:, 0]
    del st, pid, cid
    torch.cuda.empty_cache()

    cand = OK & (q >= args.qmin)
    dsum = (d * OK).sum(1)
    nsum = (n * OK).sum(1)
    a_all = np.where(dsum > EP, nsum / np.maximum(dsum, EP), a0)
    Dm = np.where(OK, d, 0.0)
    Nm = np.where(OK, n, 0.0)
    T2m = np.where(OK, t2, 0.0)
    S_fast = np.zeros((n_pil, pool.V), bool)
    S_fast[:, fast_ids] = True
    has_fast = (S_fast & OK).sum(1) > 0

    cov_ids = pick_cover_views(all_cams, state["mu"].cpu().numpy(), k=args.B)
    scr2 = {"q": torch.as_tensor(q, device=dev), "area": None}
    sels = {}
    sels["fast"] = np.tile(np.asarray(fast_ids, np.int64), (n_pil, 1))
    sels["cover"] = np.tile(np.asarray(cov_ids, np.int64), (n_pil, 1))
    for strat in ("sup", "strat", "repair", "guard"):
        sels[strat] = pool.select(scr2, strategy=strat, B=args.B, qmin=args.qmin,
                                  fast_ids=fast_ids).cpu().numpy()
    sels["pref"] = pool.select(scr2, strategy="pref", B=args.B, qmin=args.qmin,
                               fast_ids=fast_ids, pref_ids=cov_ids).cpu().numpy()
    pool.M = torch.as_tensor(M, device=dev)
    sels["track"] = pool.select(scr2, strategy="track", B=args.B,
                                qmin=args.qmin).cpu().numpy()
    del scr2
    torch.cuda.empty_cache()

    # residual-driven, fixed B (same candidate screen as o2eval2: top-16)
    top16 = np.zeros_like(cand)
    order = np.argsort(-np.where(cand, q, -1e9), axis=1)[:, :16]
    np.put_along_axis(top16, order, True, axis=1)
    sels["resid16"], _ = residual_greedy(d, n, t2, top16, args.B)
    # §18.2 adaptive budget
    for tau in (0.01, 0.03, 0.05, 0.10):
        sel_a, bud = residual_greedy(d, n, t2, top16, 12, bmin=4, tau=tau)
        sels[f"adapt{int(tau*100)}"] = sel_a
        res.setdefault("adapt_budget", {})[f"{tau:.2f}"] = float(bud.mean())

    def evaluate(sel, mask=None):
        S = np.zeros((n_pil, pool.V), bool)
        ii, jj = np.nonzero(sel >= 0)
        S[ii, sel[ii, jj]] = True
        S &= OK
        rows = (np.ones(n_pil, bool) if mask is None else mask)
        if mask is not None:
            S = S & mask[:, None]
        cnt = S.sum(1)
        am = np.where(cnt > 0, (Nm * S).sum(1) / np.maximum((Dm * S).sum(1), EP), a0)
        rel = np.abs(am - a_all) / np.maximum(np.abs(a_all), ALPHA_FLOOR)
        r_all = am[:, None] ** 2 * Dm - 2 * am[:, None] * Nm + T2m
        hold = OK & ~S
        hf = has_fast & rows
        out = {
            "E_alpha": float(rel[rows].mean()),
            "E_alpha_sup": float(rel[hf].mean()) if hf.any() else float("nan"),
            "E_heldout": float(r_all[hold].sum() / max(T2m[hold].sum(), EP)),
            "mean_views": float(cnt[rows].mean()),
            "zero_frac": float((cnt[rows] == 0).mean()),
        }
        return out

    # §17.4 pair-split: odd/even pair halves
    m_tr = np.zeros(n_pil, bool); m_tr[::2] = True
    m_va = ~m_tr
    table = {}
    import hashlib
    for k, sel in sels.items():
        print(f"  sel[{k}] md5={hashlib.md5(sel.tobytes()).hexdigest()[:10]}", flush=True)
        base = evaluate(sel)
        split = evaluate(sel, mask=m_va)
        met = selection_metrics(sel, M.astype(np.float64), pool.V)
        table[k] = {**base, "E_alpha_odd": evaluate(sel, mask=m_tr)["E_alpha"],
                    "E_alpha_even": split["E_alpha"], **met}
    res["strategies"] = table
    res["wall_s"] = round(time.perf_counter() - t_all, 2)

    print(f"[{args.tag}] V={pool.V} pilot={n_pil}  (grid {args.grid}, sigma {args.sigma_cut})")
    for k, v in table.items():
        print(f"[{args.tag}] {k:9s} E_a={v['E_alpha']:.4f} (odd {v['E_alpha_odd']:.4f} / "
              f"even {v['E_alpha_even']:.4f})  views={v['mean_views']:.2f}  "
              f"traj={v['traj_err']:.3f} red={v['redundancy']:.3f} "
              f"stab={v['stability_jaccard']:.3f} used={v['pool_used']:.2f}", flush=True)
    print(f"[{args.tag}] adaptive budget (tau -> mean views): "
          f"{res.get('adapt_budget')}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[{args.tag}] wrote {args.out} ({res['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
