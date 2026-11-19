"""Offline camera-sampling evaluator for O2cal (plan Stages 0 / 1 / 2 / 3).

Works on a dumped pass-1 state (`<tag>_o2pass1.npz`, written by
`run_o2.py --dump-o2-pass 1`), i.e. the pre-merge gaussians, the accepted pairs
and the MPMM merged components -- everything O2cal consumes.

Two blocks of output:

  Stage 0  support diagnosis: four criteria on the *same* pairs
           cin  merged centre inside the image        (the historical report)
           ov   sigma_cut footprint overlaps the image
           patch  the kernel's own ok mask            (O2 patch mask)
           info   patch & d > tau_d                   (information gate)
  Stage 1/2  camera-selection quality against the all-camera reference
           a_p^all  = sum_{C_p+} n / sum_{C_p+} d
           E_alpha  relative opacity error vs a_p^all
           E_all    full-camera proxy residual at a_p(S)
           E_held   held-out proxy residual (views not in S)

Because the merge sequence of the pipeline does not depend on O2cal (the scorer
sees frozen S0 geometry/SH), comparing strategies on one dumped pair set is an
exact "fixed pair sequence" comparison (plan 19.1).

Usage:
  python o2eval.py --scene <colmap scene> --o2dump dump_p2c1/truck_o2pass1.npz \
      --tag truck --pilot 4096 --strategies fast,cover,sup,info,strat \
      --out results/cam_eval_truck.json
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pics                                      # noqa: E402
from cams_fast import load_cams_fast, pick_cover_views   # noqa: E402

ALPHA_FLOOR = 0.05   # denominator floor for the relative opacity error


# --------------------------------------------------------------------------- #
def _instance_rows(P, cam_ids, dev):
    """(P,C) camera ids -> flat (K,) pair ids / camera ids (K = P*C)."""
    C = len(cam_ids)
    pid = torch.arange(P, device=dev).repeat_interleave(C)
    cid = torch.as_tensor(np.asarray(cam_ids, np.int64), device=dev).repeat(P)
    return pid, cid


def _summarise_counts(cnt_per_pair):
    c = np.asarray(cnt_per_pair, np.float64)
    return {
        "zero_frac": float((c == 0).mean()),
        "mean": float(c.mean()),
        "hist": {f"{a}-{b}": float(((c >= a) & (c <= b)).mean())
                 for a, b in [(0, 0), (1, 1), (2, 3), (4, 5), (6, 8), (9, 16),
                              (17, 10 ** 9)]},
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--o2dump", required=True)
    ap.add_argument("--tag", default="scene")
    ap.add_argument("--pilot", type=int, default=4096)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--B-list", default=None,
                    help="comma separated budgets, evaluates every strategy at each")
    ap.add_argument("--nbins", type=int, default=None,
                    help="trajectory bins for the stratified strategies (default = B)")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--sigma-cut", type=float, default=2.0)
    ap.add_argument("--strategies", default="fast,cover,sup,info,strat")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--qmin", type=float, default=0.05)
    ap.add_argument("--tau-q", type=float, default=0.10,
                    help="quantile of d used as tau_d for the info gate")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    res = {"tag": args.tag, "scene": args.scene, "o2dump": args.o2dump,
           "pilot": args.pilot, "B": args.B, "grid": args.grid,
           "sigma_cut": args.sigma_cut}
    t_wall = time.perf_counter()

    # ---- load the dump -------------------------------------------------- #
    z = np.load(args.o2dump)
    state = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
             for k in ("mu", "sc", "q", "op")}
    comp = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
            for k in ("mu_m", "sc_m", "q_m")}
    pairs = torch.as_tensor(np.ascontiguousarray(z["pairs"]).astype(np.int64),
                            device=dev)
    P = int(pairs.shape[0])
    print(f"[{args.tag}] dump: P={P} N={state['mu'].shape[0]} "
          f"Vdump={comp['mu_m'].shape[0]}", flush=True)

    # ---- camera pool ---------------------------------------------------- #
    t0 = time.perf_counter()
    all_cams = load_cams_fast(args.scene, "fit", 0)
    fast_cams = load_cams_fast(args.scene, "fit", args.B)
    pool = pics.CamPool(all_cams, dev)
    t_cams = time.perf_counter() - t0
    name2idx = {n: i for i, n in enumerate(pool.names)}
    fast_ids = [name2idx[c["name"]] for c in fast_cams]
    res["n_fit"] = pool.V
    res["fast_ids"] = fast_ids
    res["cams_s"] = round(t_cams, 4)
    print(f"[{args.tag}] fit cameras: {pool.V}, fast subset {len(fast_ids)} "
          f"({t_cams:.3f}s)", flush=True)

    # ---- Stage 0 on the historical 8 views, all pairs ------------------- #
    gpu = {}
    if P * len(fast_ids) <= 12_000_000:
        t0 = time.perf_counter()
        pid, cid = _instance_rows(P, fast_ids, dev)
        st = pool.patch_stats(state, comp, pairs, pid, cid, grid=args.grid,
                              sigma_cut=args.sigma_cut)
        torch.cuda.synchronize(dev)
        t_fast = time.perf_counter() - t0
        n_view = np.bincount(pid.cpu().numpy(), minlength=P)
        stage0 = {}
        for key in ("cin", "ov", "ok"):
            cnt = torch.zeros(P, dtype=torch.int32, device=dev).index_add_(
                0, pid, st[key].to(torch.int32)).cpu().numpy()
            stage0[key] = _summarise_counts(cnt)
        # information gate d > tau_d
        d_np = st["d"].cpu().numpy()
        ok_np = st["ok"].cpu().numpy()
        tau = float(np.quantile(d_np[ok_np], args.tau_q)) if ok_np.any() else 0.0
        keep = ok_np & (d_np > tau)
        cnt = np.bincount(pid.cpu().numpy()[keep], minlength=P)
        stage0["info"] = _summarise_counts(cnt)
        stage0["tau_d"] = tau
        stage0["o2_s_fast8_allpairs"] = round(t_fast, 3)
        res["stage0_fast8"] = stage0
        print(f"[{args.tag}] Stage0 fast8 all pairs ({t_fast:.2f}s): " + " ".join(
            f"{k} zero={stage0[k]['zero_frac']*100:.1f}% mean={stage0[k]['mean']:.2f}"
            for k in ("cin", "ov", "ok", "info")), flush=True)
        del pid, cid, st, d_np, ok_np, keep, cnt, n_view
        gpu.clear()
    torch.cuda.empty_cache()

    # ---- Stage 0 on the full pool + Strategy evaluation on a pilot ------ #
    n_pil = min(args.pilot, P) if args.pilot > 0 else P
    if n_pil == P:
        pil = torch.arange(P, device=dev)
    else:
        pil = torch.linspace(0, P - 1, n_pil).round().to(torch.int64).to(dev)
    pil_np = pil.cpu().numpy()
    sub_pairs = pairs[pil]
    sub_comp = {k: v[pil] for k, v in comp.items()}
    res["pilot"] = int(n_pil)

    t0 = time.perf_counter()
    scr_all = pool.screen(comp["mu_m"], comp["sc_m"], sigma_cut=args.sigma_cut)
    torch.cuda.synchronize(dev)
    t_screen = time.perf_counter() - t0
    res["screen_s"] = round(t_screen, 4)
    print(f"[{args.tag}] screen all pairs x {pool.V}: {t_screen:.3f}s", flush=True)

    scr = {"q": scr_all["q"][pil], "area": (scr_all["area"][pil]
                                            if scr_all["area"] is not None else None)}
    del scr_all
    torch.cuda.empty_cache()

    # all-camera patch statistics for the pilot pairs
    t0 = time.perf_counter()
    pid_all, cid_all = _instance_rows(n_pil, list(range(pool.V)), dev)
    st_all = pool.patch_stats(state, sub_comp, sub_pairs, pid_all, cid_all,
                              grid=args.grid, sigma_cut=args.sigma_cut)
    torch.cuda.synchronize(dev)
    t_allp = time.perf_counter() - t0
    res["patch_s_allcams_pilot"] = round(t_allp, 3)
    print(f"[{args.tag}] exact patch, pilot x all cams "
          f"({pid_all.numel()} inst): {t_allp:.2f}s", flush=True)

    d_np = st_all["d"].cpu().numpy()
    n_np = st_all["n"].cpu().numpy()
    t2_np = st_all["t2"].cpu().numpy()
    ok_np = st_all["ok"].cpu().numpy()
    cin_np = st_all["cin"].cpu().numpy()
    ov_np = st_all["ov"].cpu().numpy()
    a0_np = st_all["a0"].cpu().numpy()
    d = d_np.reshape(n_pil, pool.V)
    n = n_np.reshape(n_pil, pool.V)
    t2 = t2_np.reshape(n_pil, pool.V)
    OK = ok_np.reshape(n_pil, pool.V)
    del pid_all, cid_all, st_all
    torch.cuda.empty_cache()

    # pilot-level Stage 0 (four criteria) on the full pool
    stage0_pool = {}
    cnt = OK.sum(1)
    stage0_pool["patch"] = _summarise_counts(cnt)
    stage0_pool["cin"] = _summarise_counts(cin_np.reshape(n_pil, pool.V).sum(1))
    stage0_pool["ov"] = _summarise_counts(ov_np.reshape(n_pil, pool.V).sum(1))
    tau = float(np.quantile(d[OK], args.tau_q)) if OK.any() else 0.0
    stage0_pool["info"] = _summarise_counts((OK & (d > tau)).sum(1))
    stage0_pool["tau_d"] = tau
    res["stage0_pool"] = stage0_pool
    print(f"[{args.tag}] Stage0 full pool (pilot) tau_d={tau:.4g}: " + " ".join(
        f"{k} zero={stage0_pool[k]['zero_frac']*100:.1f}% "
        f"mean={stage0_pool[k]['mean']:.2f}"
        for k in ("cin", "ov", "patch", "info")), flush=True)

    # ---- reference a_p^all over C_p^+ ----------------------------------- #
    ep = 1e-9
    Cplus = OK
    d_sum = (d * Cplus).sum(1)
    n_sum = (n * Cplus).sum(1)
    a_all = np.where(d_sum > ep, n_sum / np.maximum(d_sum, ep), a0_np.reshape(n_pil, -1)[:, 0])
    n_eff = Cplus.sum(1).astype(np.float64)
    res["pool_support"] = {"mean": float(n_eff.mean()),
                           "median": float(np.median(n_eff)),
                           "zero_frac": float((n_eff == 0).mean())}
    print(f"[{args.tag}] C_p+ over full pool: mean {n_eff.mean():.1f} views, "
          f"zero {(n_eff == 0).mean()*100:.1f}%", flush=True)
    res["a_all_stats"] = {"min": float(a_all.min()),
                          "p10": float(np.quantile(a_all, 0.1)),
                          "median": float(np.median(a_all)),
                          "frac_below_floor": float((np.abs(a_all) < ALPHA_FLOOR).mean())}
    print(f"[{args.tag}] a_all: min {a_all.min():.4g} p10 {np.quantile(a_all, 0.1):.4g} "
          f"median {np.median(a_all):.4g} "
          f"frac<{ALPHA_FLOOR}: {(np.abs(a_all) < ALPHA_FLOOR).mean()*100:.1f}%",
          flush=True)

    # fast8 reference counts on the pilot: used to split E_alpha into the
    # "already fitted" and "fallback" populations (plan 24, Stage 2)
    S_fast = np.zeros((n_pil, pool.V), dtype=bool)
    S_fast[:, fast_ids] = True
    S_fast &= OK
    fast_ref_cnt = S_fast.sum(1)
    res["fast_pilot"] = {"zero_frac": float((fast_ref_cnt == 0).mean()),
                         "mean_views": float(fast_ref_cnt.mean())}

    # ---- strategies ----------------------------------------------------- #
    strat_names = [s for s in args.strategies.split(",") if s]
    B_list = [int(x) for x in args.B_list.split(",")] if args.B_list else [args.B]
    fast_by_B = {}
    for Bv in B_list:
        fc = load_cams_fast(args.scene, "fit", Bv)
        fast_by_B[Bv] = [name2idx[c["name"]] for c in fc]
    res["fast_ids"] = fast_by_B[B_list[0]]
    cover_by_B = {}
    for Bv in B_list:
        _pts = state["mu"].cpu().numpy()
        cover_by_B[Bv] = pick_cover_views(all_cams, _pts, k=Bv)
    sel_map = {}
    for Bv in B_list:
        for s in strat_names:
            key = f"{s}@{Bv}"
            if s == "fast":
                sid = np.tile(np.asarray(fast_by_B[Bv], np.int64), (n_pil, 1))
            elif s == "cover":
                sid = np.tile(np.asarray(cover_by_B[Bv], np.int64), (n_pil, 1))
                if Bv == B_list[0]:
                    res["cover_ids"] = [int(v) for v in cover_by_B[Bv]]
            elif s in ("sup", "info", "strat", "repair", "fill", "pref"):
                t0 = time.perf_counter()
                sid = pool.select(scr, strategy=s, B=Bv, qmin=args.qmin,
                                  fast_ids=(fast_by_B[Bv] if s == "repair"
                                            else fast_ids),
                                  nbins=args.nbins,
                                  pref_ids=(cover_by_B[Bv] if s == "pref"
                                            else None)).cpu().numpy()
                res[f"select_s_{key}"] = round(time.perf_counter() - t0, 4)
            else:
                print(f"  !! unknown strategy {s}", flush=True)
                continue
            sel_map[key] = sid
    del scr
    torch.cuda.empty_cache()

    def evaluate(sel):
        """sel: (n_pil, B) camera ids (-1 padded). Returns the plan's metrics."""
        S = np.zeros((n_pil, pool.V), dtype=bool)
        ii, jj = np.nonzero(sel >= 0)
        S[ii, sel[ii, jj]] = True
        S &= OK
        cntS = S.sum(1)
        num = (n * S).sum(1)
        den = (d * S).sum(1)
        a0v = a0_np.reshape(n_pil, -1)[:, 0]
        am = np.where(cntS > 0, num / np.maximum(den, ep), a0v)
        # a_p^all can be ~0 for pairs whose merged footprint barely receives
        # source coverage, which makes the plan's |a(S)-a_all|/|a_all| blow up.
        # Floor the denominator at ALPHA_FLOOR (reported alongside) and also
        # report the absolute and median errors, which need no floor.
        rel = np.abs(am - a_all) / np.maximum(np.abs(a_all), ALPHA_FLOOR)
        # E_alpha, split into "pairs fast8 already fitted" and "fast8 fallbacks"
        has_fast = (fast_ref_cnt > 0)
        e_alpha = float(np.mean(rel))
        e_alpha_sup = float(np.mean(rel[has_fast])) if has_fast.any() else float("nan")
        e_alpha_fb = float(np.mean(rel[~has_fast])) if (~has_fast).any() else float("nan")
        # full-camera residual at am
        r_all = (am[:, None] ** 2 * d - 2 * am[:, None] * n + t2)
        e_all = float(r_all[OK].sum() / max(t2[OK].sum(), ep))
        hold = OK & ~S
        e_held = float(r_all[hold].sum() / max(t2[hold].sum(), ep))
        f = float(a0v.mean() / max(am.mean(), ep))
        return {
            "zero_frac": float((cntS == 0).mean()),
            "mean_views": float(cntS.mean()),
            "E_alpha": e_alpha,
            "E_alpha_sup": e_alpha_sup,
            "E_alpha_fb": e_alpha_fb,
            "E_abs": float(np.mean(np.abs(am - a_all))),
            "E_med": float(np.median(np.abs(am - a_all)
                                     / np.maximum(np.abs(a_all), ALPHA_FLOOR))),
            "bias": float(np.mean(am - a_all)),
            "corr": float(np.corrcoef(am, a_all)[0, 1]) if am.std() > 1e-9 else 0.0,
            "E_all": e_all,
            "E_heldout": e_held,
            "f_cal": f,
            "am_mean": float(am.mean()),
            "a_all_mean": float(a_all.mean()),
            "d_sum_ratio": float(((d * S).sum(1).mean()) / max(d_sum.mean(), ep)),
            "sel_overlap_with_ok": float((cntS
                                          / np.maximum((sel >= 0).sum(1), 1)).mean()),
        }

    table = {}
    for key, sel in sel_map.items():
        t0 = time.perf_counter()
        table[key] = evaluate(sel)
        table[key]["eval_s"] = round(time.perf_counter() - t0, 3)
        tb = table[key]
        print(f"[{args.tag}] {key:10s} E_a={tb['E_alpha']:.4f} "
              f"(sup {tb['E_alpha_sup']:.4f} / fb {tb['E_alpha_fb']:.4f}) "
              f"|a| {tb['E_abs']:.4f} med {tb['E_med']:.4f} "
              f"E_all={tb['E_all']:.4f} corr={tb['corr']:.4f} "
              f"zero={tb['zero_frac']*100:.1f}% views={tb['mean_views']:.2f} "
              f"bias={tb['bias']:+.4f} cost~{tb['mean_views']*0.42/3.07:.2f}x",
              flush=True)
    # all-camera reference row (uses the full pool -> its own residual is 0)
    table["all"] = {
        "zero_frac": float((n_eff == 0).mean()), "mean_views": float(n_eff.mean()),
        "E_alpha": 0.0, "bias": 0.0, "E_all": 0.0, "E_heldout": float("nan"),
        "f_cal": float(a0_np.reshape(n_pil, -1)[:, 0].mean() / max(a_all.mean(), ep)),
        "am_mean": float(a_all.mean()),
    }
    res["strategies"] = table
    res["wall_s"] = round(time.perf_counter() - t_wall, 2)
    print(f"[{args.tag}] total {res['wall_s']}s", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[{args.tag}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
