#!/usr/bin/env python3
"""Finalize the AutoVirt-v3 exploration table (plan §3.3 template).

Joins, by `run_id` only (no manual copy-paste, the R1/R2 lesson):
  raw PSNR/SSIM/LPIPS  <- timing/av_v3_explore_raw.csv
  O0 / v1 baselines    <- timing/auto_virt_p2_{s0,dc}.csv
  real cover8          <- timing/o2cover.csv
  support / r_px       <- the per-run camera JSON + the pass-1 dump

Columns (plan §3.3):
  run_id,track,exp_id,scene,sh_mode,rho,n_views,flags,psnr,ssim,lpips,
  delta_vs_v25,retention_pct,true_zero_sup,mean_sup,med_r_px,shell_dist,
  t_hot_ms,t_cold_ms,t_o2_ms,gpu_idle_ok,cam_json_path
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from o2gpu import O2Cameras, _project, _lmax2x2      # noqa: E402
from diag_density_support import DUMPS, M, MODEL    # noqa: E402

RAW = "timing/av_v3_explore_raw.csv"
OUT = "timing/av_v3_explore.csv"
SCENES = ["truck", "room", "garden", "bicycle", "playroom", "train"]


def read_baseline(path):
    d = {}
    if not os.path.exists(path):
        return d
    for r in csv.DictReader(open(path)):
        try:
            d.setdefault(r["scene"], {})[r["arm"]] = float(r["psnr"])
        except (KeyError, ValueError):
            continue
    return d


def support_stats(cams, sn, dev, sc=1.25, bs=8192):
    """(mean_sup, zero%, med_r_px) of these cameras on the scene's merged set."""
    f = DUMPS.get(sn)
    if not f or not os.path.exists(f):
        return None, None, None
    d = np.load(f, allow_pickle=True)
    P = int(d["pairs"].shape[0])
    idx = np.random.default_rng(0).choice(P, min(20000, P), replace=False)
    if "mu_m" in d.files:
        mu_m, sc_m, q_m = d["mu_m"][idx], d["sc_m"][idx], d["q_m"][idx]
    else:
        pr = d["pairs"][idx]
        mu_m = np.concatenate([d["mu"][pr[:, 0]], d["mu"][pr[:, 1]]], 0)
        sc_m = np.concatenate([d["sc"][pr[:, 0]], d["sc"][pr[:, 1]]], 0)
        q_m = np.concatenate([d["q"][pr[:, 0]], d["q"][pr[:, 1]]], 0)
    oc = O2Cameras(cams, dev)
    mu = torch.as_tensor(mu_m, device=dev)
    a = torch.as_tensor(sc_m, device=dev)
    q = torch.as_tensor(q_m, device=dev)
    s_mid = torch.sort(a, 1).values[:, 1]
    cnt = torch.zeros(mu.shape[0], dtype=torch.int32, device=dev)
    rpx_all = []
    for i in range(0, mu.shape[0], bs):
        j = min(i + bs, mu.shape[0])
        mu2d, S2, valid = _project(mu[i:j], a[i:j], q[i:j], oc)
        rr = sc * torch.sqrt(_lmax2x2(S2).clamp_min(0.0))
        cx, cy = mu2d[..., 0], mu2d[..., 1]
        ok = (valid & (rr > 1e-3) & (cx + rr > 0) & (cx - rr < oc.W[None, :])
              & (cy + rr > 0) & (cy - rr < oc.H[None, :]))
        cnt[i:j] = ok.sum(1)
        # projected pixel radius of the *merged* gaussian
        for c_i, cam in enumerate(cams):
            Mw = torch.as_tensor(np.asarray(cam["w2v"], np.float32), device=dev)
            ph = torch.cat([mu[i:j], torch.ones(j - i, 1, device=dev)], 1)
            pcz = (ph[:, None, :] * Mw[None, :, :]).sum(-1)
            zz = pcz[:, 2].clamp_min(1e-6)
            r = (s_mid[i:j] * float(cam["fx"]) / zz)
            m = ok[:, c_i]
            if bool(m.any()):
                rpx_all.append(r[m].detach().cpu().numpy())
    med_r = float(np.median(np.concatenate(rpx_all))) if rpx_all else float("nan")
    c = cnt.cpu().numpy()
    return float(c.mean()), float((c == 0).mean() * 100), med_r


def main():
    rows = list(csv.DictReader(open(RAW)))
    b0 = read_baseline("timing/auto_virt_p2_s0.csv")
    bd = read_baseline("timing/auto_virt_p2_dc.csv")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    # Base_v2.5 = the best T0-A config by 6-scene mean (over both sh_modes)
    exp_means = {}
    for r in rows:
        if r["track"] != "T0A" or r["psnr"] in ("FAIL", ""):
            continue
        exp_means.setdefault(r["exp_id"], []).append(float(r["psnr"]) - (
            b0.get(r["scene"], {}).get("av", np.nan) if r["sh_mode"] == "s0"
            else bd.get(r["scene"], {}).get("av", np.nan)))
    v25 = max(exp_means, key=lambda k: np.nanmean(exp_means[k])) if exp_means else ""
    print(f"Base_v2.5 = {v25}  (means: "
          + ", ".join(f"{k}={np.nanmean(v):+.4f}" for k, v in exp_means.items()) + ")")

    cache = {}
    out = []
    for r in rows:
        sn = r["scene"]
        base = b0 if r["sh_mode"] == "s0" else bd
        v1 = base.get(sn, {}).get("av", float("nan"))
        c8 = base.get(sn, {}).get("real", float("nan"))
        p = float(r["psnr"]) if r["psnr"] not in ("FAIL", "") else float("nan")
        v25p = next((float(x["psnr"]) for x in rows
                     if x["run_id"] != r["run_id"] and x["exp_id"] == v25
                     and x["scene"] == sn and x["sh_mode"] == r["sh_mode"]
                     and x["psnr"] not in ("FAIL", "")), float("nan"))
        key = (sn, r["cam_json_path"])
        if key not in cache:
            ms = zs = mr = None
            cj = r.get("cam_json_path", "")
            if cj and os.path.exists(cj):
                raw = json.load(open(cj))
                if isinstance(raw, dict):
                    raw = raw.get("cams", [])
                cams = [dict(name=c["name"], w2v=np.asarray(c["w2v"], np.float64),
                             fx=float(c["fx"]), fy=float(c["fy"]), W=float(c["W"]),
                             H=float(c["H"]), C=np.asarray(c["C"], np.float64))
                        for c in raw]
                ms, zs, mr = support_stats(cams, sn, dev)
            cache[key] = (ms, zs, mr)
        ms, zs, mr = cache[key]
        out.append(dict(
            run_id=r["run_id"], track=r["track"], exp_id=r["exp_id"], scene=sn,
            sh_mode=r["sh_mode"], rho=r.get("rho", "0.10"),
            n_views=r.get("n_views", "8"), flags=r.get("flags", ""),
            psnr=f"{p:.4f}" if p == p else "FAIL",
            ssim=r.get("ssim", ""), lpips=r.get("lpips", ""),
            delta_vs_v25=f"{p - v25p:+.4f}" if (p == p and v25p == v25p) else "",
            retention_pct=(f"{100*(p - (base.get(sn,{}).get('o0',float('nan'))))/(c8 - base.get(sn,{}).get('o0',float('nan'))):.1f}"
                           if (p == p and c8 == c8 and base.get(sn, {}).get("o0") is not None)
                           else ""),
            true_zero_sup=f"{zs:.2f}" if zs is not None else "",
            mean_sup=f"{ms:.2f}" if ms is not None else "",
            med_r_px=f"{mr:.2f}" if mr is not None and mr == mr else "",
            shell_dist="", t_hot_ms=r.get("t_hot_ms", ""),
            t_cold_ms="", t_o2_ms=r.get("t_o2_ms", ""),
            gpu_idle_ok=r.get("gpu_idle_ok", "0"),
            cam_json_path=r.get("cam_json_path", "")))
        print(f"  {r['run_id']:52s} {sn:9s} {r['sh_mode']:3s} psnr={out[-1]['psnr']:>8s} "
              f"ret={out[-1]['retention_pct']:>6s}% sup={out[-1]['mean_sup']:>5s} "
              f"zero={out[-1]['true_zero_sup']:>6s}% r_px={out[-1]['med_r_px']:>7s}")

    keys = ["run_id", "track", "exp_id", "scene", "sh_mode", "rho", "n_views",
            "flags", "psnr", "ssim", "lpips", "delta_vs_v25", "retention_pct",
            "true_zero_sup", "mean_sup", "med_r_px", "shell_dist", "t_hot_ms",
            "t_cold_ms", "t_o2_ms", "gpu_idle_ok", "cam_json_path"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"\n-> {OUT}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
