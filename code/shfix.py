#!/usr/bin/env python3
"""Contribution-weighted low-rank SH correction for merged gaussians (plan S2/S3).

Settings that are **frozen** (the plan's 閹?.2): merge pairs and order, MPMM
geometry, O2cal opacity, gaussian budget.  Only the merged SH moves.

Model
-----
For a merge (i, j) -> m and a calibration view v, with d_v the view direction of
the merged gaussian (all three are evaluated at the *same* direction, which is
exact for the DC term and a controlled approximation for the linear terms):

    b_v      = SH basis at d_v                     (K,)
    c_x,v    = b_v^T C_x                           colour of x in {i, j, m}
    w_x,v    = alpha_x / sqrt(det S2_x,v)          EWA peak weight in that view
    t_v      = w_i c_i + w_j c_j                   reference contribution
    p_v(閾?   = w_m (b_v^T (C0_m + 閾?)              merged contribution

Minimise sum_v ||p_v(閾? - t_v||^2 + lam ||閾绘澅|^2, i.e. the batched normal equations

    A    = sum_v w_m^2 b_v b_v^T + lam I           (P, K, K)
    rhs  = sum_v w_m b_v (t_v - w_m b_v^T C0_m)^T  (P, K, 3)

K = 1 (DC only, plan S3a), 4 (DC + degree 1, plan S3b), or 16 (full, S2 with r=16).
The update is clipped and falls back to C0 when the system is ill-conditioned or
too few views are valid.
"""
import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


def _basis(d, mode):
    """d: (B,V,3) unit vectors -> (B,V,K)."""
    if mode == "dc":
        return torch.ones(d.shape[:-1] + (1,), dtype=d.dtype, device=d.device) * SH_C0
    if mode == "l1":
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        return torch.stack([torch.full_like(x, SH_C0),
                            -SH_C1 * y, SH_C1 * z, -SH_C1 * x], -1)
    raise ValueError(f"unknown sh mode {mode}")


def _coef_l1(sh):
    """(N,4,3): rows = [dc, l1_x, l1_y, l1_z], cols = RGB."""
    dc = sh[:, 0:3]
    rest = sh[:, 3:48].reshape(-1, 3, 15)             # channel-major: R[15], G[15], B[15]
    l1 = rest[:, :, 0:3]                              # (N,3ch,3coef)
    out = torch.empty((sh.shape[0], 4, 3), dtype=sh.dtype, device=sh.device)
    out[:, 0, :] = dc
    out[:, 1:4, :] = l1.transpose(1, 2)
    return out


def _det2(S2):
    return S2[..., 0, 0] * S2[..., 1, 1] - S2[..., 0, 1] * S2[..., 1, 0]


def make_cache(P, V, dev):
    """Buffers for values O2cal already computes; filled inside its own loop."""
    return {"det_i": torch.empty((P, V), dtype=torch.float64, device=dev),
            "det_j": torch.empty((P, V), dtype=torch.float64, device=dev),
            "det_m": torch.empty((P, V), dtype=torch.float64, device=dev),
            "v_i": torch.empty((P, V), dtype=torch.bool, device=dev),
            "v_j": torch.empty((P, V), dtype=torch.bool, device=dev),
            "v_m": torch.empty((P, V), dtype=torch.bool, device=dev)}


def make_sums(P, dev):
    """Running sums the SH correction needs; filled inside O2cal's own loop."""
    return {"Si": torch.zeros(P, dtype=torch.float64, device=dev),
            "Sj": torch.zeros(P, dtype=torch.float64, device=dev)}


def correct_dc_from_sums(state, comp, pairs, sums, lam=1e-2, clip=0.30,
                         min_diff=0.0, stats=None):
    """DC-only correction from O2cal's accumulated view weights.

    Because the SH DC basis is the constant b0, the per-view least squares
    collapses exactly:

        num = sum_v s_v (t_v - c0) = c_i*Si + c_j*Sj - c0*(Si+Sj)
        den = sum_v s_v            = Si + Sj
        dC  = num / (b0 * den + lam)

    with Si = sum_v w_i, Sj = sum_v w_j.  Only (P,) and (P,3) work remains, so
    this costs a handful of kernels for the whole pass instead of per chunk.
    """
    sh, = (state[4],)
    i, j = pairs[:, 0], pairs[:, 1]
    out = comp["sh_m"]
    sh_dc = sh[:, 0:3].contiguous()
    sm_dc = out[:, 0:3].contiguous()
    Si, Sj = sums["Si"], sums["Sj"]
    Ss = Si + Sj

    ci = sh_dc[i].double()
    cj = sh_dc[j].double()
    c0 = sm_dc.double()

    if min_diff > 0.0:
        sel = (ci - cj).abs().amax(dim=1) > min_diff
        # leave the unselected merges at their S0 DC value
        Si = torch.where(sel, Si, torch.zeros_like(Si))
        Sj = torch.where(sel, Sj, torch.zeros_like(Sj))
        Ss = Si + Sj

    num = ci * Si[:, None] + cj * Sj[:, None] - c0 * Ss[:, None]
    den = (SH_C0 * Ss + lam)[:, None]
    delta = num / den
    bad = ~torch.isfinite(delta).all(-1)
    n_fb = int(bad.sum())
    if n_fb:
        delta[bad] = 0.0
    n_clip = int((delta.abs() > clip).any(-1).sum())
    out[:, 0:3] += delta.clamp(-clip, clip).to(out.dtype)

    if stats is not None:
        stats["sh_correction"] = {"mode": "dc_sums", "lam": lam, "clip": clip,
                                  "fallback": n_fb, "clipped": n_clip,
                                  "pairs": int(i.numel())}
    return out


def correct_sh(state, comp, pairs, cams, mode="dc", lam=1e-2, clip=0.30,
               chunk=65536, stats=None, cache=None, min_diff=0.0):
    """Return a corrected copy of comp["sh_m"] (P, 48). Nothing else changes.

    With `cache` supplied the three projections are *not* recomputed: the EWA
    weights come from the 2-D covariance determinants O2cal already produced.

    `min_diff` skips merges whose two source DC vectors differ by less than that
    on every channel.  That test is (P,3) only, so the (P,V,3) work runs just for
    the merges whose colour can actually change.
    """
    from o2gpu import _project

    mu, sc, q, op, sh = state
    i, j = pairs[:, 0], pairs[:, 1]
    P = i.numel()
    K = 1 if mode == "dc" else 4
    out = comp["sh_m"]          # in place: the S0 lineage is a separate tensor

    # Gathering rows of the (N,48) SH block costs 16x more bandwidth than needed:
    # only the 3 DC columns are used.  Materialise them once per pass (N*3 floats,
    # 12 MB for 1M gaussians) and gather from that instead.
    sh_dc = sh[:, 0:3].contiguous() if sh.shape[1] >= 3 else None
    sm_dc = out[:, 0:3].contiguous()

    # camera centre in world space: w2v = [R|t] maps world -> view, C = -R^T t
    R = cams.w2v[:, :3, :3].double()
    t = cams.w2v[:, :3, 3].double()
    C = -torch.matmul(R.transpose(1, 2), t[:, :, None])[:, :, 0]      # (V,3)

    idx_all = None
    if min_diff > 0.0:
        d_dc = (sh_dc[i] - sh_dc[j]).abs().amax(dim=1)
        sel = d_dc > min_diff
        idx_all = torch.nonzero(sel, as_tuple=False).flatten()
        if stats is not None:
            stats["sh_selected_frac"] = round(float(sel.float().mean()), 4)
        if idx_all.numel() == 0:
            if stats is not None:
                stats["sh_correction"] = {"mode": mode, "lam": lam, "clip": clip,
                                          "fallback": 0, "clipped": 0,
                                          "pairs": int(P), "selected": 0}
            return out

    n_work = int(idx_all.numel()) if idx_all is not None else P
    n_fallback = 0
    n_clip = 0
    for s0 in range(0, n_work, chunk):
        s1 = min(n_work, s0 + chunk)
        if idx_all is not None:
            ii = i[idx_all[s0:s1]]
            jj = j[idx_all[s0:s1]]
            sl = idx_all[s0:s1]           # rows of comp["sh_m"] to update
            csl = sl                       # same rows in the cache
        else:
            ii, jj = i[s0:s1], j[s0:s1]
            sl = slice(s0, s1)
            csl = sl

        if cache is None:
            mu2d_i, S2_i, v_i = _project(mu[ii], sc[ii], q[ii], cams, torch.float64, True)
            mu2d_j, S2_j, v_j = _project(mu[jj], sc[jj], q[jj], cams, torch.float64, True)
            mu2d_m, S2_m, v_m = _project(comp["mu_m"][sl], comp["sc_m"][sl],
                                         comp["q_m"][sl], cams, torch.float64, True)
            det_i, det_j, det_m = _det2(S2_i), _det2(S2_j), _det2(S2_m)
        else:
            det_i = cache["det_i"][csl]
            det_j = cache["det_j"][csl]
            det_m = cache["det_m"][csl]
            v_i, v_j, v_m = cache["v_i"][csl], cache["v_j"][csl], cache["v_m"][csl]

        def weight(opv, det):
            return opv[:, None] / det.clamp(min=1e-12).sqrt()

        w_i = weight(op[ii], det_i)
        w_j = weight(op[jj], det_j)
        # the merged weight is only used for "is it valid at all"; op > 0 always
        ok = v_i & v_j & v_m

        if mode == "dc":
            # the DC basis is a constant: no direction, no basis tensor, no
            # (B,V,3) intermediates at all
            ci = sh_dc[ii]
            cj = sh_dc[jj]
            c0 = sm_dc[sl]
            s_w = (w_i + w_j).clamp(min=1e-12)                     # (B,V)
            tgt = (w_i[..., None] * ci[:, None, :]
                   + w_j[..., None] * cj[:, None, :]) / s_w[..., None]
            r = (tgt - c0[:, None, :]) * ok[..., None]
            sv = s_w * ok
            num = (sv[..., None] * r).sum(1)                       # (B,3)
            den = sv.sum(1)                                        # (B,)
            delta = num / (den * SH_C0 + lam)[:, None]             # (B,3)
            bad = ~torch.isfinite(delta).all(-1)
            if bad.any():
                delta[bad] = 0.0
                n_fallback += int(bad.sum())
            n_clip += int((delta.abs() > clip).any(-1).sum())
            out[sl, 0:3] += delta.clamp(-clip, clip).to(out.dtype)
            continue

        # --- degree-1 path (kept for ablation; not used by default) ---------
        d = (comp["mu_m"][sl][:, None, :].double() - C[None, :, :])   # (B,V,3)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        b = _basis(d, mode)                                            # (B,V,K)
        Ci = _coef_l1(sh[ii])
        Cj = _coef_l1(sh[jj])
        C0 = _coef_l1(comp["sh_m"][sl])

        c_i = torch.einsum("bvk,bkc->bvc", b, Ci.double())
        c_j = torch.einsum("bvk,bkc->bvc", b, Cj.double())
        c_0 = torch.einsum("bvk,bkc->bvc", b, C0.double())

        # O2cal has already calibrated the merged opacity to reproduce the pair's
        # total coverage, so the merged colour should match the *contribution
        # weighted mean* of the two source colours -- not their sum.
        s_w = (w_i + w_j).clamp(min=1e-12)
        tgt = (w_i[..., None] * c_i + w_j[..., None] * c_j) / s_w[..., None]
        r = (tgt - c_0) * ok[..., None]

        # views where the pair is prominent should dominate the fit
        sv = s_w * ok
        A = torch.einsum("bv,bvk,bvl->bkl", sv, b, b)
        A = A + lam * torch.eye(K, dtype=A.dtype, device=A.device)[None]
        rhs = torch.einsum("bv,bvk,bvc->bkc", sv, b, r)

        try:
            delta = torch.linalg.solve(A, rhs)
        except Exception:
            delta = torch.zeros_like(rhs)
        bad = ~torch.isfinite(delta).all(-1).all(-1)
        if bad.any():
            delta[bad] = 0.0
            n_fallback += int(bad.sum())

        big = delta.abs() > clip
        n_clip += int(big.any(-1).any(-1).sum())
        delta = delta.clamp(-clip, clip)

        if mode == "dc":
            out[sl, 0:3] += delta[:, 0, :].to(out.dtype)
        else:
            out[sl, 0:3] += delta[:, 0, :].to(out.dtype)
            d3 = delta[:, 1:4, :]                                      # (B,3,3) rows=coef, cols=RGB
            # write back channel-major: R gets coef 0..2 of channel 0
            out[sl, 3:6] += d3[:, :, 0].to(out.dtype)
            out[sl, 18:21] += d3[:, :, 1].to(out.dtype)
            out[sl, 33:36] += d3[:, :, 2].to(out.dtype)

    if stats is not None:
        stats["sh_correction"] = {"mode": mode, "lam": lam, "clip": clip,
                                  "fallback": n_fallback, "clipped": n_clip,
                                  "pairs": int(P)}
    return out
