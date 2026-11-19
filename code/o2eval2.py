"""Stage 1 / Stage 2 remaining criteria for O2cal camera sampling (plan 15, 16, 17).

Works on the *pilot* pair set of a dumped pass-1 state, where the exact
per-(pair, camera) O2cal statistics (d, n, t2, ok) have already been computed for
the whole fit pool.  One exact patch pass plus one cheap 4x4 signature pass then
suffices to evaluate all of:

  global fixed subsets (plan 15)
    g_fast      even-stride subsample (= the historical baseline)
    g_cover     greedy coverage (+40 deg, the historical cover)
    g_fps       camera-centre farthest-point sampling              (G3)
    g_posefps   centre + optical-axis farthest-point sampling      (G4)
    g_fvstrack  FPS on d_spatial + 2 * d_photo                     (G5, plan 11.5)
  pair-conditioned (plan 16)
    p_sup / p_strat / p_repair / p_guard / p_pref   (already reported; re-run
                                                     here so the table is complete)
    p_track     support gate + track-redundancy penalty            (P6, plan 11.6)
    p_dopt      response-space D-optimal greedy                    (P5, plan 11.3)
    p_aopt      response-space A-optimal greedy                    (G11)
    p_topt      response-space T-optimal (top ||psi||)             (G11)
    p_resid     greedy held-out residual minimisation, all cands   (P7, plan 17)
    p_resid16   same but only the 16 best-supported candidates     (deployable)
    p_adapt     resid16 with the plan's adaptive stop rule         (plan 18.2)

Usage:
  python o2eval2.py --scene <scene> --o2dump <dump> --tag truck --pilot 8192 \
      --out results/cam_stage12_truck.json
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


# --------------------------------------------------------------------------- #
# global fixed subsets
# --------------------------------------------------------------------------- #
def _fps_metric(D, k, start=0):
    """Greedy farthest-point sampling given a (V,V) *distance* matrix."""
    V = D.shape[0]
    sel = [int(start)]
    dmin = D[sel[0]].copy()
    while len(sel) < min(k, V):
        dmin[sel] = -1.0
        i = int(np.argmax(dmin))
        sel.append(i)
        dmin = np.minimum(dmin, D[i])
    return sel


def global_sets(pool, M, k, cover_ids):
    C = pool.C.detach().cpu().numpy().astype(np.float64)
    A = pool.axis.detach().cpu().numpy().astype(np.float64)
    scale = float(np.median(np.linalg.norm(C[:, None] - C[None], axis=-1))) + 1e-9
    med = int(np.argmin(((C - C.mean(0)) ** 2).sum(1)))          # medoid as seed
    out = {"cover": list(cover_ids)}
    Dc = np.linalg.norm(C[:, None] - C[None], axis=-1)
    out["fps"] = _fps_metric(Dc, k, start=med)
    # pose FPS: centre (scaled) + unit optical axis, so a 180 deg turn costs about
    # one median camera spacing (plan 11.5, joint-pose FPS)
    Z = np.concatenate([C / scale, A], 1)
    out["posefps"] = _fps_metric(np.linalg.norm(Z[:, None] - Z[None], axis=-1), k,
                                 start=med)
    # FVS with photogrammetric distance (plan 11.5 / NeRF Director)
    Dph = 1.0 - np.clip(M, 0.0, 1.0)
    Dc_n = Dc / (np.quantile(Dc, 0.9) + EP)
    Dph_n = Dph / (np.quantile(Dph, 0.9) + EP)
    out["fvstrack"] = _fps_metric(Dc_n + 2.0 * Dph_n, k, start=med)
    return out


# --------------------------------------------------------------------------- #
# selection on the exact (pilot x V) statistics
# --------------------------------------------------------------------------- #
def pair_track_greedy(q, cand, M, k, lam=1.0):
    """Support gate + diminishing redundancy (plan 11.6 / 16.1 P6).

    score = q_{p,c} - lam * max_{s in S} M[c,s]   (so a view that duplicates an
    already-selected one is only chosen if it is much better supported).
    """
    P, V = q.shape
    sel = np.full((P, k), -1, np.int32)
    used = np.zeros((P, V), bool)
    red = np.zeros((P, V), np.float32)
    base = np.where(cand, q.astype(np.float32), -1e9)
    rows_all = np.arange(P)
    for step in range(k):
        score = np.where(cand & ~used, base - lam * red, -1e9)
        i = np.argmax(score, 1)
        ok = score[rows_all, i] > -1e8
        sel[ok, step] = i[ok]
        rows = np.nonzero(ok)[0]
        used[rows, i[rows]] = True
        red[rows] = np.maximum(red[rows], M[i[rows]])
    return sel


def residual_greedy(d, n, t2, cand, k, bmin=1, tau=0.0):
    """Greedy minimisation of the held-out O2cal residual on C_p^+ \\ S (plan 17.3).

    Returns (sel (P,k), budget (P,)). `tau` is the relative-improvement stop
    threshold of plan 18.2 (0 disables it).
    """
    P, V = d.shape
    D = np.where(cand, d, 0.0)
    N = np.where(cand, n, 0.0)
    T2 = np.where(cand, t2, 0.0)
    dtot, ntot, t2tot = D.sum(1), N.sum(1), T2.sum(1)
    sel = np.full((P, k), -1, np.int32)
    num = np.zeros(P)
    den = np.zeros(P)
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


def dopt_greedy(psi, cand, k, sigma2=1.0, lam_rel=0.01, crit="D"):
    """Greedy (D/A/T)-optimal selection in the PCA response space (plan 11.3)."""
    P, V, r = psi.shape
    tr = np.einsum("pvr,pvr->p", psi, psi) / np.maximum(cand.sum(1), 1)
    Finv = (np.eye(r, dtype=np.float32)[None]
            / np.maximum(lam_rel * tr, 1e-9)[:, None, None]).astype(np.float32)
    sel = np.full((P, k), -1, np.int32)
    used = np.zeros((P, V), bool)
    rows_all = np.arange(P)
    for step in range(k):
        u = np.einsum("pvr,prl->pvl", psi, Finv)               # F^-1 psi
        g = np.einsum("pvl,pvl->pv", u, psi) / sigma2          # psi^T F^-1 psi
        if crit == "D":
            score = np.log1p(g)
        elif crit == "A":
            score = g / (1.0 + g)                              # trace reduction
        else:
            score = np.einsum("pvr,pvr->pv", psi, psi)
        score = np.where(cand & ~used, score, -1e9)
        i = np.argmax(score, 1)
        ok = score[rows_all, i] > -1e8
        sel[rows_all[ok], step] = i[ok]
        rows = np.nonzero(ok)[0]
        uu = u[rows, i[rows]]
        gg = np.einsum("nr,nr->n", uu, psi[rows, i[rows]])
        Finv[rows] -= (uu[:, :, None] * uu[:, None, :]) / (sigma2 + gg)[:, None, None]
        used[rows, i[rows]] = True
    return sel


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--o2dump", required=True)
    ap.add_argument("--tag", default="scene")
    ap.add_argument("--pilot", type=int, default=8192)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--sigma-cut", type=float, default=2.0)
    ap.add_argument("--qmin", type=float, default=0.05)
    ap.add_argument("--pca", type=int, default=8)
    ap.add_argument("--adapt-tau", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    res = {"tag": args.tag, "pilot": args.pilot, "B": args.B,
           "grid": args.grid, "qmin": args.qmin, "pca": args.pca}
    t_all = time.perf_counter()

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
    print(f"[{args.tag}] P={P} V={pool.V} overlap mean "
          f"{float(M[~np.eye(pool.V, dtype=bool)].mean()):.4f}", flush=True)

    n_pil = min(args.pilot, P)
    pil = torch.linspace(0, P - 1, n_pil).round().to(torch.int64).to(dev)
    sub_pairs = pairs[pil]
    sub_comp = {k: v[pil] for k, v in comp.items()}
    scr = pool.screen(comp["mu_m"], comp["sc_m"], sigma_cut=args.sigma_cut,
                      want_area=False)
    q = scr["q"][pil].to(torch.float32).cpu().numpy()
    del scr
    torch.cuda.empty_cache()

    pid = torch.arange(n_pil, device=dev).repeat_interleave(pool.V)
    cid = torch.arange(pool.V, device=dev).repeat(n_pil)

    t0 = time.perf_counter()
    st = pool.patch_stats(state, sub_comp, sub_pairs, pid, cid, grid=args.grid,
                          sigma_cut=args.sigma_cut)
    torch.cuda.synchronize(dev)
    print(f"[{args.tag}] patch pilot x all ({pid.numel()} inst) "
          f"{time.perf_counter()-t0:.2f}s", flush=True)
    d = st["d"].cpu().numpy().reshape(n_pil, pool.V)
    n = st["n"].cpu().numpy().reshape(n_pil, pool.V)
    t2 = st["t2"].cpu().numpy().reshape(n_pil, pool.V)
    OK = st["ok"].cpu().numpy().reshape(n_pil, pool.V)
    a0 = st["a0"].cpu().numpy().reshape(n_pil, pool.V)[:, 0]
    del st
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    sig, _ = pool.signature(state, sub_comp, sub_pairs, pid, cid, grid=9,
                            sigma_cut=args.sigma_cut)
    torch.cuda.synchronize(dev)
    print(f"[{args.tag}] signature {time.perf_counter()-t0:.2f}s", flush=True)
    psi_raw = sig.cpu().numpy().reshape(n_pil, pool.V, -1)
    del sig, pid, cid
    torch.cuda.empty_cache()

    cand = OK & (q >= args.qmin)
    dsum = (d * OK).sum(1)
    nsum = (n * OK).sum(1)
    a_all = np.where(dsum > EP, nsum / np.maximum(dsum, EP), a0)
    Dm = np.where(OK, d, 0.0)
    Nm = np.where(OK, n, 0.0)
    T2m = np.where(OK, t2, 0.0)
    res["pool_support"] = {"mean": float(OK.sum(1).mean()),
                           "zero_frac": float((OK.sum(1) == 0).mean())}

    # PCA of the raw signature, fitted on valid candidates
    flat = psi_raw[OK]
    if flat.shape[0] > 200000:
        rs = np.random.default_rng(0)
        flat = flat[rs.choice(flat.shape[0], 200000, replace=False)]
    mu_sig = flat.mean(0)
    _, S, Vt = np.linalg.svd(flat - mu_sig, full_matrices=False)
    r = min(args.pca, Vt.shape[0])
    proj = Vt[:r].T.astype(np.float32)
    psi = ((psi_raw - mu_sig) @ proj).astype(np.float32)
    psi /= (psi[OK].std(0).mean() + 1e-9)
    res["pca_var"] = float(S[:r].sum() / max(S.sum(), 1e-12))
    print(f"[{args.tag}] PCA r={r} var {res['pca_var']:.3f}", flush=True)

    S_fast = np.zeros((n_pil, pool.V), bool)
    S_fast[:, fast_ids] = True
    has_fast = (S_fast & OK).sum(1) > 0

    def evaluate(sel):
        S = np.zeros((n_pil, pool.V), bool)
        ii, jj = np.nonzero(sel >= 0)
        S[ii, sel[ii, jj]] = True
        S &= OK
        cnt = S.sum(1)
        am = np.where(cnt > 0, (Nm * S).sum(1) / np.maximum((Dm * S).sum(1), EP), a0)
        rel = np.abs(am - a_all) / np.maximum(np.abs(a_all), ALPHA_FLOOR)
        r_all = am[:, None] ** 2 * Dm - 2 * am[:, None] * Nm + T2m
        hold = OK & ~S
        return {
            "zero_frac": float((cnt == 0).mean()),
            "mean_views": float(cnt.mean()),
            "E_alpha": float(rel.mean()),
            "E_alpha_sup": float(rel[has_fast].mean()),
            "E_alpha_fb": (float(rel[~has_fast].mean()) if (~has_fast).any()
                           else float("nan")),
            "E_abs": float(np.abs(am - a_all).mean()),
            "E_med": float(np.median(rel)),
            "E_all": float(r_all[OK].sum() / max(T2m[OK].sum(), EP)),
            "E_heldout": float(r_all[hold].sum() / max(T2m[hold].sum(), EP)),
            "bias": float((am - a_all).mean()),
            "corr": float(np.corrcoef(am, a_all)[0, 1]) if am.std() > 1e-9 else 0.0,
        }

    table = {}
    cov_ids = pick_cover_views(all_cams, state["mu"].cpu().numpy(), k=args.B)
    gs = global_sets(pool, M, args.B, cov_ids)
    gs["fast"] = list(fast_ids)
    for k, ids in gs.items():
        table[f"g_{k}"] = evaluate(np.tile(np.asarray(ids, np.int64), (n_pil, 1)))
        table[f"g_{k}"]["ids"] = [int(v) for v in ids]

    scr2 = {"q": torch.as_tensor(q, device=dev), "area": None}
    for strat in ("sup", "strat", "repair", "guard", "pref"):
        sid = pool.select(scr2, strategy=strat, B=args.B, qmin=args.qmin,
                          fast_ids=fast_ids, pref_ids=cov_ids).cpu().numpy()
        table[f"p_{strat}"] = evaluate(sid)
    del scr2
    torch.cuda.empty_cache()

    table["p_track"] = evaluate(pair_track_greedy(q, cand, M, args.B))
    for crit in ("D", "A", "T"):
        table[f"p_{crit.lower()}opt"] = evaluate(dopt_greedy(psi, cand, args.B,
                                                             crit=crit))
    sel_r, _ = residual_greedy(d, n, t2, cand, args.B)
    table["p_resid"] = evaluate(sel_r)
    top16 = np.zeros_like(cand)
    order = np.argsort(-np.where(cand, q, -1e9), axis=1)[:, :16]
    np.put_along_axis(top16, order, True, axis=1)
    sel_r16, _ = residual_greedy(d, n, t2, top16, args.B)
    table["p_resid16"] = evaluate(sel_r16)
    sel_ad, bud_ad = residual_greedy(d, n, t2, top16, 16, bmin=4,
                                     tau=args.adapt_tau)
    table["p_adapt"] = evaluate(sel_ad)
    table["p_adapt"]["budget_mean"] = float(bud_ad.mean())

    res["strategies"] = table
    res["a_all_stats"] = {"median": float(np.median(a_all)),
                          "min": float(a_all.min())}
    res["wall_s"] = round(time.perf_counter() - t_all, 2)

    for k, v in table.items():
        print(f"[{args.tag}] {k:12s} E_a={v['E_alpha']:.4f} "
              f"(sup {v['E_alpha_sup']:.4f}/fb {v['E_alpha_fb']:.4f}) "
              f"|a| {v['E_abs']:.4f} E_all={v['E_all']:.4f} "
              f"zero={v['zero_frac']*100:.1f}% views={v['mean_views']:.2f}",
              flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[{args.tag}] wrote {args.out}  ({res['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
