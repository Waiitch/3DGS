"""GPU O2cal: batched projected-coverage opacity fit + per-pass calibration.

Semantics are those of `ops2.fit_alpha_coverage` / `replay_merge.opacity_o2`
(the CPU reference used by the project), re-expressed so that the batch axes are
(pair, view, sample) and nothing is copied per pair:

  a0   = clip(op_i + op_j - op_i*op_j)
  rr   = sigma_cut * sqrt(lambda_max(S2_m))
  ok   = valid_i & valid_j & valid_m & (rr > 1e-3) & centre inside image +- rr
  px   = clamp(cx + gx*rr, 0, W-1), py = clamp(cy + gy*rr, 0, H-1)
  Gm   = exp(-0.5 * clamp(q, 0, 60)), q = d^T S2^-1 d
  a_ref= 1 - (1 - a_i G_i)(1 - a_j G_j)
  am   = sum_{view,sample} Gm*a_ref / sum_{view,sample} Gm^2   (cnt>0, else a0)
  cal  : f = mean(a0)/mean(am); am *= f                      (one factor per pass)

`opacity_o2_cpu` is a thin wrapper around the untouched CPU reference so the two
can be compared pair-for-pair.
"""
from __future__ import annotations

import numpy as np
import torch

LOG2E = 1.4426950408889634


class O2Cameras:
    """Camera constants stacked once per run (never re-uploaded per pass)."""

    def __init__(self, cams, dev):
        self.dev = dev
        self.names = [c["name"] for c in cams]
        self.w2v = torch.as_tensor(
            np.stack([np.asarray(c["w2v"], np.float32) for c in cams]), device=dev)
        self.fx = torch.as_tensor(
            np.asarray([c["fx"] for c in cams], np.float32), device=dev)
        self.fy = torch.as_tensor(
            np.asarray([c["fy"] for c in cams], np.float32), device=dev)
        self.W = torch.as_tensor(
            np.asarray([c["W"] for c in cams], np.float32), device=dev)
        self.H = torch.as_tensor(
            np.asarray([c["H"] for c in cams], np.float32), device=dev)
        self.V = len(cams)

    def grid(self, n):
        g = torch.linspace(-1.0, 1.0, n, device=self.dev)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        return gx.reshape(-1), gy.reshape(-1)


def _rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _project(mu, sc, q, cams: O2Cameras, dtype=torch.float64, noblas=False,
             return_tz=False):
    """EWA projection of (B,3)/(B,3)/(B,4) gaussians into all views at once.

    Runs in float64 by default: the 2-D covariance of a strongly anisotropic
    merged splat has a condition number of 1e4-1e6, and in float32 the smallest
    eigenvalue is lost to cancellation.  That made `q` (and therefore the fit's
    denominator) wrong by orders of magnitude for a handful of pairs, which the
    per-pass mean-based calibration then turned into a global collapse.

    `noblas=True` replaces the three einsums and the matmul with broadcast
    multiply+sum: same maths, but the process then never initialises cuBLAS
    (whose first call costs ~0.7 s inside whichever stage triggers it).

    Returns mu2d (B,V,2), S2 (B,V,2,2), valid (B,V).
    """
    mu = mu.to(dtype)
    sc = sc.to(dtype)
    q = q.to(dtype)
    w2v = cams.w2v.to(dtype)
    fx, fy, W, H = cams.fx.to(dtype), cams.fy.to(dtype), cams.W.to(dtype), cams.H.to(dtype)
    B = mu.shape[0]
    V = cams.V
    ph = torch.cat([mu, torch.ones((B, 1), dtype=dtype, device=mu.device)], 1)
    if noblas:
        pc = (ph[:, None, None, :] * w2v[None, :, :, :]).sum(-1)     # (B,V,4)
    else:
        pc = torch.einsum("bj,vkj->bvk", ph, w2v)                    # (B,V,4)
    tz = pc[..., 2]
    valid = tz > 1e-6
    tz = torch.clamp(tz, min=1e-9)
    tx = pc[..., 0] / tz
    ty = pc[..., 1] / tz
    mu2d = torch.stack([fx * tx + 0.5 * W, fy * ty + 0.5 * H], -1)    # (B,V,2)

    J = torch.zeros((B, V, 2, 3), dtype=dtype, device=mu.device)
    J[..., 0, 0] = fx / tz
    J[..., 0, 2] = -fx * tx / tz
    J[..., 1, 1] = fy / tz
    J[..., 1, 2] = -fy * ty / tz
    if noblas:
        T = (J[:, :, :, :, None] * w2v[:, :3, :3][None, :, None, :, :]).sum(3)
    else:
        T = torch.einsum("bvik,vkj->bvij", J, w2v[:, :3, :3])

    R = _rot(q)                                              # (B,3,3)
    if noblas:
        M_ = (T[:, :, :, :, None] * R[:, None, None, :, :]).sum(3)    # (B,V,2,3)
    else:
        M_ = torch.einsum("bvij,bjk->bvik", T, R)                     # (B,V,2,3)
    Mv = M_ * (sc * sc)[:, None, None, :]
    if noblas:
        S2 = (Mv[:, :, :, None, :] * M_[:, :, None, :, :]).sum(-1)   # (B,V,2,2)
    else:
        S2 = torch.matmul(Mv, M_.transpose(-1, -2))                  # (B,V,2,2)
    eye = torch.eye(2, dtype=dtype, device=mu.device)
    S2 = 0.5 * (S2 + S2.transpose(-1, -2)) + 0.3 * eye
    if return_tz:
        return mu2d, S2, valid, pc[..., 2]
    return mu2d, S2, valid


def _lmax2x2(S2):
    a, b = S2[..., 0, 0], S2[..., 0, 1]
    cc, d = S2[..., 1, 0], S2[..., 1, 1]
    tr = a + d
    det = a * d - b * cc
    disc = torch.sqrt(torch.clamp(tr * tr * 0.25 - det, min=0.0))
    return tr * 0.5 + disc


def _gauss(mu2d, S2, px, py, out32=False):
    """G at sample points; px/py (K,P), mu2d/S2 (K,...).

    The quadratic form is evaluated in the input dtype (float64, because the
    smallest eigenvalue of an anisotropic 2-D covariance is lost in float32);
    only the final exp is cast down, which keeps the accuracy where it matters
    and the bulk arithmetic cheap.
    """
    a, b = S2[..., 0, 0], S2[..., 0, 1]
    c, d = S2[..., 1, 0], S2[..., 1, 1]
    det = torch.clamp(a * d - b * c, min=1e-12)
    i00 = d / det
    i11 = a / det
    i01 = -b / det
    dx = px - mu2d[..., 0:1]
    dy = py - mu2d[..., 1:2]
    qq = i00[..., None] * dx * dx + 2.0 * i01[..., None] * dx * dy + i11[..., None] * dy * dy
    qq = torch.clamp(qq, 0.0, 60.0)
    if out32:
        qq = qq.to(torch.float32)
    return torch.exp(-0.5 * qq)


def _det2(S2):
    return S2[..., 0, 0] * S2[..., 1, 1] - S2[..., 0, 1] * S2[..., 1, 0]


def opacity_o2cal_gpu(state, comp, pairs, cams, grid=9, sigma_cut=2.0,
                      calibrate=False, batch=16384, stats=None, timers=None,
                      dtype=torch.float64, noblas=False, cache=None, sh_sums=None,
                      view_tau=0.0, view_rmax=40.0,
                      solver="grid", cf_beta=0.0, cf_agg="eq",
                      cf_fix=False, cf_safe=False, cf_c0=0.15,
                      clean_mask=False, bound_guard=False,
                      sh_dc=None, dc_out=None,
                      eg_mode="E1", eg_gamma=1.0, eg_eps=0.0, eg_wmode="area",
                      eg_psi=35.0, debug_dump=None):
    """state: dict(mu,sc,q,op) of the PRE-merge model (GPU tensors).
    comp: dict(mu_m,sc_m,q_m,op_m) merge results for `pairs`.
    Returns the fitted opacity for the merged splats (P,), float32, on device.

    `cache`, when a dict, receives the 2-D covariance determinants and validity
    masks that this call already computes (keys det_i/det_j/det_m/v_i/v_j/v_m).

    `sh_sums`, when a dict with keys Si/Sj (float64 (P,) zeros), accumulates
    sum_v op_i/sqrt(det_i) and the same for j over the views where all three
    gaussians project.  That is everything the DC-only SH correction needs: the
    SH DC basis is a constant, so the per-view dimension collapses analytically
    and no extra pass over the pairs is required.
    """
    mu, sc, q, op = state["mu"], state["sc"], state["q"], state["op"]
    i, j = pairs[:, 0], pairs[:, 1]
    P = pairs.shape[0]
    dev = mu.device
    _gx, _gy = cams.grid(grid)
    gx, gy = _gx.to(dtype), _gy.to(dtype)
    G = gx.numel()

    num = torch.zeros(P, dtype=dtype, device=dev)
    den = torch.zeros(P, dtype=dtype, device=dev)
    cnt = torch.zeros(P, dtype=torch.int32, device=dev)
    o2_views = 0

    a_i_all = op[i].to(dtype)
    a_j_all = op[j].to(dtype)
    a0 = torch.clamp(a_i_all + a_j_all - a_i_all * a_j_all, 0.0, 1.0)

    Wd = cams.W.to(dtype)
    Hd = cams.H.to(dtype)
    for b0 in range(0, P, batch):
        b1 = min(P, b0 + batch)
        sl = slice(b0, b1)
        if solver == "eigen":
            # Per-pair eigen-axis closed form (pair_o2cal.md).  NOTE: this branch
            # touches NEITHER `cams` NOR the grid, so it is provably independent
            # of the camera rig: the same weights must reproduce under any
            # --o2-cams value.
            from o2_eigen import o2_eigen_batch as _o2eig
            t = _tic()
            a_star, _hs, _ws = _o2eig(
                mu[i[sl]].to(dtype), mu[j[sl]].to(dtype), comp["mu_m"][sl].to(dtype),
                sc[i[sl]].to(dtype), sc[j[sl]].to(dtype), comp["sc_m"][sl].to(dtype),
                q[i[sl]].to(dtype), q[j[sl]].to(dtype), comp["q_m"][sl].to(dtype),
                a_i_all[b0:b1], a_j_all[b0:b1],
                mode=str(eg_mode), gamma=float(eg_gamma), eps_ewa=float(eg_eps),
                beta=float(cf_beta), w_mode=str(eg_wmode), psi_deg=float(eg_psi))
            num[b0:b1] = a_star
            den[b0:b1] = 1.0
            cnt[b0:b1] = 1
            o2_views += (b1 - b0)
            _toc(timers, "o2_eigen", t)
            continue
        t = _tic()
        t = _tic()
        tz_i = tz_j = tz_m = None
        if clean_mask:
            # M2/M3: a sigma_cut-free validity test.  Only geometry that any
            # renderer must agree on: a safe near plane, the merged centre inside
            # the image, and no degenerate tangential blow-up.
            mu2d_i, S2_i, v_i, tz_i = _project(mu[i[sl]], sc[i[sl]], q[i[sl]],
                                               cams, dtype, noblas, True)
            mu2d_j, S2_j, v_j, tz_j = _project(mu[j[sl]], sc[j[sl]], q[j[sl]],
                                               cams, dtype, noblas, True)
            mu2d_m, S2_m, v_m, tz_m = _project(comp["mu_m"][sl], comp["sc_m"][sl],
                                               comp["q_m"][sl], cams, dtype,
                                               noblas, True)
            _toc(timers, "o2_project", t)
            t = _tic()
            cx, cy = mu2d_m[..., 0], mu2d_m[..., 1]
            ok = (tz_i > 0.2) & (tz_j > 0.2) & (tz_m > 0.2)
            ok &= (cx >= 0.0) & (cx < Wd) & (cy >= 0.0) & (cy < Hd)
            for S_ in (S2_i, S2_j, S2_m):
                s00_, s01_, s11_ = S_[..., 0, 0], S_[..., 0, 1], S_[..., 1, 1]
                ok &= torch.isfinite(s00_) & torch.isfinite(s11_)
                ok &= (s00_ < 1e8) & (s11_ < 1e8)
            rr = sigma_cut * torch.sqrt(torch.clamp(_lmax2x2(S2_m), min=1e-9))
            _toc(timers, "o2_mask", t)
        else:
            mu2d_i, S2_i, v_i = _project(mu[i[sl]], sc[i[sl]], q[i[sl]], cams, dtype, noblas)
            mu2d_j, S2_j, v_j = _project(mu[j[sl]], sc[j[sl]], q[j[sl]], cams, dtype, noblas)
            mu2d_m, S2_m, v_m = _project(comp["mu_m"][sl], comp["sc_m"][sl],
                                         comp["q_m"][sl], cams, dtype, noblas)
            _toc(timers, "o2_project", t)

            t = _tic()
            rr = sigma_cut * torch.sqrt(torch.clamp(_lmax2x2(S2_m), min=1e-9))
            cx, cy = mu2d_m[..., 0], mu2d_m[..., 1]
            ok = v_i & v_j & v_m & (rr > 1e-3)
            ok &= torch.isfinite(cx) & torch.isfinite(cy) & torch.isfinite(rr)
            ok &= (cx > -rr) & (cx < Wd + rr)
            ok &= (cy > -rr) & (cy < Hd + rr)
            _toc(timers, "o2_mask", t)

        if cache is not None or sh_sums is not None:
            det_i = _det2(S2_i)
            det_j = _det2(S2_j)
            det_m = _det2(S2_m)
        if cache is not None:
            cache["det_i"][sl] = det_i
            cache["det_j"][sl] = det_j
            cache["det_m"][sl] = det_m
            cache["v_i"][sl] = v_i
            cache["v_j"][sl] = v_j
            cache["v_m"][sl] = v_m
        if sh_sums is not None:
            # sum_v op/sqrt(det) over views where all three gaussians project
            okv = (v_i & v_j & v_m).to(dtype)
            sh_sums["Si"][sl] += ((a_i_all[sl][:, None]
                                   / det_i.clamp(min=1e-12).sqrt()) * okv).sum(1)
            sh_sums["Sj"][sl] += ((a_j_all[sl][:, None]
                                   / det_j.clamp(min=1e-12).sqrt()) * okv).sum(1)

        bv = torch.nonzero(ok.reshape(-1), as_tuple=False).flatten()   # (K,) local
        if bv.numel() == 0:
            continue

        t = _tic()
        b_idx = (bv // cams.V) + b0
        v_idx = bv % cams.V

        if solver == "analytic":
            # Analytical closed form (o2cal_analytical_closed_form.md): the exact
            # continuous L2 solution, no 9x9 grid, elementwise float64 only.
            #
            # gather: the closed form is evaluated ONLY on the valid (pair, view)
            # entries (K of them), then scattered back per pair.  The grid path
            # already works this way; without it the solver walked the full
            # (B, V) tensor and was 6-7x the grid cost on small scenes.
            from o2_analytic import o2_closed_form_batch as _o2cf
            bn = b1 - b0
            b_loc = bv // cams.V
            flat = lambda x: x.reshape(-1, *x.shape[2:])[bv]
            ones_k = torch.ones(bv.numel(), 1, dtype=torch.bool, device=dev)
            ai_k = a_i_all[b0:b1][b_loc]
            aj_k = a_j_all[b0:b1][b_loc]
            _, _, _trm = _o2cf(
                flat(mu2d_i), flat(mu2d_j), flat(mu2d_m),
                flat(S2_i), flat(S2_j), flat(S2_m), ai_k, aj_k, ones_k,
                beta=float(cf_beta), agg="eq",
                clamp_c=bool(cf_fix), safe_mask=bool(cf_safe),
                c0_floor=float(cf_c0), bound_guard=bool(bound_guard))
            av = _trm["alpha_v"][:, 0]            # per-valid-entry alpha, (K,)
            m_el = _trm["w_v"][:, 0]              # 0 where the entry is degenerate
            if str(cf_agg) == "eq":
                w_el = m_el
            else:
                Sg = flat(S2_m)
                s00_, s01_, s11_ = Sg[..., 0, 0], Sg[..., 0, 1], Sg[..., 1, 1]
                det_ = (s00_ * s11_ - s01_ * s01_).clamp(min=0.0)
                tr_ = s00_ + s11_
                rad_ = torch.sqrt(((s00_ - s11_) ** 2 + 4 * s01_ * s01_).clamp(min=0.0))
                lmax_ = (0.5 * (tr_ + rad_)).clamp(min=1e-18)
                lmin_ = (0.5 * (tr_ - rad_)).clamp(min=1e-18)
                if str(cf_agg) == "area":
                    w_el = m_el * torch.sqrt(det_)
                elif str(cf_agg) == "ecc":
                    w_el = m_el * (torch.sqrt(det_.clamp(min=1e-18)) / lmax_)
                else:                              # ecc_floor
                    c0 = float(cf_c0)
                    w_el = m_el * (c0 + (1.0 - c0)
                                   * torch.sqrt((lmin_ / lmax_).clamp(0.0, 1.0)))
            num_p = torch.zeros(bn, dtype=dtype, device=dev)
            den_p = torch.zeros(bn, dtype=dtype, device=dev)
            num_p.index_add_(0, b_loc, (av * w_el).to(dtype))
            den_p.index_add_(0, b_loc, w_el.to(dtype))
            a_star = torch.where(den_p > 0, num_p / den_p.clamp(min=1e-12),
                                 a0[b0:b1])
            num[b0:b1] = a_star
            den[b0:b1] = (den_p > 0).to(num.dtype)
            cnt[b0:b1] = ok.sum(1).to(torch.int32)
            o2_views += int(bv.numel())
            if sh_dc is not None and dc_out is not None:
                # DC (RGB) closed-form refit on exactly the same (pair, view) set
                from o2_analytic import dc_color_fit as _dcf
                ci_k = sh_dc[i[sl]].to(dtype)[b_loc]
                cj_k = sh_dc[j[sl]].to(dtype)[b_loc]
                col, cw = _dcf(flat(mu2d_i) - flat(mu2d_m),
                               flat(mu2d_j) - flat(mu2d_m),
                               flat(S2_i), flat(S2_j),
                               flat(S2_m), ai_k, aj_k, ci_k, cj_k,
                               torch.ones(bv.numel(), dtype=torch.bool, device=dev))
                num_c = torch.zeros(bn, 3, dtype=dtype, device=dev)
                den_c = torch.zeros(bn, dtype=dtype, device=dev)
                wc = (cw * w_el).to(dtype)
                num_c.index_add_(0, b_loc, col * wc[:, None])
                den_c.index_add_(0, b_loc, wc)
                dc_out[b0:b1] = torch.where(
                    den_c[:, None] > 0, num_c / den_c.clamp(min=1e-12)[:, None],
                    torch.zeros_like(num_c))
            _toc(timers, "o2_cf", t)
            if debug_dump is not None and "alpha_v" not in debug_dump:
                av_full = torch.full((bn, cams.V), float("nan"), dtype=dtype, device=dev)
                av_full.reshape(-1)[bv] = av
                debug_dump["alpha_v"] = av_full.detach().cpu()
                debug_dump["valid"] = ok.detach().cpu()
                debug_dump["Sm"] = S2_m.detach().cpu()
                debug_dump["Si"] = S2_i.detach().cpu()
                debug_dump["Sj"] = S2_j.detach().cpu()
                debug_dump["a_i"] = a_i_all[b0:b1].detach().cpu()
                debug_dump["a_j"] = a_j_all[b0:b1].detach().cpu()
                debug_dump["alpha_multi"] = a_star.detach().cpu()
                debug_dump["solver"] = "analytic"
            continue

        rrk = rr.reshape(-1)[bv]
        px = torch.minimum(torch.maximum(cx.reshape(-1)[bv, None] + gx[None, :] * rrk[:, None],
                                         torch.zeros((), dtype=dtype, device=dev)),
                           (Wd - 1.0)[v_idx, None])
        py = torch.minimum(torch.maximum(cy.reshape(-1)[bv, None] + gy[None, :] * rrk[:, None],
                                         torch.zeros((), dtype=dtype, device=dev)),
                           (Hd - 1.0)[v_idx, None])
        flat = lambda x: x.reshape(-1, *x.shape[2:])[bv]
        Gi = _gauss(flat(mu2d_i), flat(S2_i), px, py)
        Gj = _gauss(flat(mu2d_j), flat(S2_j), px, py)
        Gm = _gauss(flat(mu2d_m), flat(S2_m), px, py)
        _toc(timers, "o2_grid", t)

        t = _tic()
        a_i = a_i_all[b_idx][:, None].to(Gi.dtype)
        a_j = a_j_all[b_idx][:, None].to(Gi.dtype)
        a_ref = 1.0 - (1.0 - a_i * Gi) * (1.0 - a_j * Gj)
        if view_tau > 0.0:
            # S4: per-view projection-conditioning weight.  r^2/(r^2+tau^2) is a
            # smooth 0..1 clarity score: sub-pixel views (already flattened by the
            # +0.3 I low-pass) get down-weighted, sharp views dominate.  A hard
            # cap at view_rmax only guards against degenerate close-ups.
            r2 = rrk * rrk
            wv = r2 / (r2 + float(view_tau) ** 2)
            wv = torch.where(rrk <= float(view_rmax), wv, torch.zeros_like(wv))
            num.index_add_(0, b_idx, (Gm * a_ref * wv[:, None]).sum(1).to(num.dtype))
            den.index_add_(0, b_idx, (Gm * Gm * wv[:, None]).sum(1).to(den.dtype))
        else:
            num.index_add_(0, b_idx, (Gm * a_ref).sum(1).to(num.dtype))
            den.index_add_(0, b_idx, (Gm * Gm).sum(1).to(den.dtype))
        cnt.index_add_(0, b_idx, torch.ones_like(b_idx, dtype=torch.int32))
        o2_views += int(b_idx.numel())
        _toc(timers, "o2_fit", t)
        if debug_dump is not None and "num_v" not in debug_dump:
            # per-(pair, view) intermediates of the GRID path, first batch only
            bn = b1 - b0
            num_v = torch.zeros(bn, cams.V, dtype=dtype, device=dev)
            den_v = torch.zeros(bn, cams.V, dtype=dtype, device=dev)
            loc = torch.nonzero(ok.reshape(-1), as_tuple=False).flatten()
            bl = loc // cams.V
            vl = loc % cams.V
            num_v.index_put_((bl, vl), (Gm * a_ref).sum(1).to(dtype))
            den_v.index_put_((bl, vl), (Gm * Gm).sum(1).to(dtype))
            debug_dump["num_v"] = num_v.detach().cpu()
            debug_dump["den_v"] = den_v.detach().cpu()
            debug_dump["ui"] = mu2d_i.detach().cpu()
            debug_dump["uj"] = mu2d_j.detach().cpu()
            debug_dump["um"] = mu2d_m.detach().cpu()
            debug_dump["valid"] = ok.detach().cpu()
            debug_dump["Sm"] = S2_m.detach().cpu()
            debug_dump["Si"] = S2_i.detach().cpu()
            debug_dump["Sj"] = S2_j.detach().cpu()
            debug_dump["a_i"] = a_i_all[b0:b1].detach().cpu()
            debug_dump["a_j"] = a_j_all[b0:b1].detach().cpu()
            debug_dump["alpha_multi"] = (num[b0:b1]
                                         / torch.clamp(den[b0:b1], min=1e-9)).detach().cpu()
            debug_dump["solver"] = "grid"

    t = _tic()
    am = torch.where(cnt > 0, num / torch.clamp(den, min=1e-9), a0)
    am = torch.where(torch.isfinite(am), am, a0)
    if stats is not None:
        stats["clip_hi"] = stats.get("clip_hi", 0) + int((am > 1).sum())
        stats["clip_lo"] = stats.get("clip_lo", 0) + int((am < 0).sum())
        stats["o2_fallback"] = stats.get("o2_fallback", 0) + int((cnt == 0).sum())
        stats["o2_views"] = stats.get("o2_views", 0) + o2_views
        stats["o2_pairs"] = stats.get("o2_pairs", 0) + P
        stats["am_mean_pre_cal"] = float(am.mean())
        stats["am_max_pre_cal"] = float(am.max())
    if calibrate:
        f = a0.mean() / torch.clamp(am.mean(), min=1e-9)
        if stats is not None:
            stats["o2_calib"] = stats.get("o2_calib", 0.0) + float(f)
            stats["o2_calib_n"] = stats.get("o2_calib_n", 0) + 1
            stats["last_f"] = float(f)
        am = am * f
    out = torch.clamp(am, 0.0, 1.0).to(torch.float32)
    _toc(timers, "o2_cal", t)
    return out


# --------------------------------------------------------------------------- #
def opacity_o2cal_cpu(state, comp, pairs, cams, grid=9, sigma_cut=2.0,
                      calibrate=False, stats=None, timers=None):
    """The untouched CPU reference (replay_merge.opacity_o2)."""
    from replay_merge import opacity_o2
    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)
    st = stats if stats is not None else {}
    t = _tic()
    out = opacity_o2(state["mu"], state["op"], state["sc"], state["q"], i, j,
                     comp["mu_m"], comp["sc_m"], comp["q_m"], cams, st,
                     grid=grid, sigma_cut=sigma_cut, calibrate=calibrate)
    _toc(timers, "o2_cpu", t)
    mu0 = state["mu"]
    dev = mu0.device if isinstance(mu0, torch.Tensor) else torch.device("cpu")
    return torch.as_tensor(out, device=dev)


def _tic():
    import time
    return time.perf_counter()


def _toc(timers, name, t0):
    if timers is None:
        return
    import time
    timers[name] = timers.get(name, 0.0) + (time.perf_counter() - t0)
