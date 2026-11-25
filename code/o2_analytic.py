#!/usr/bin/env python3
"""Analytical closed-form O2cal (o2cal_analytical_closed_form.md).

Replaces the 9x9 local grid sampling with the exact continuous L2 solution

    alpha*_v = argmin_a ∫ G_m(x)^beta (a G_m(x) - a_ref(x))^2 dx
             = a_i c_im + a_j c_jm - a_i a_j c_ijm

All arithmetic is elementwise float64: zero cuBLAS / cuSOLVER.
"""
from __future__ import annotations

import math

import torch


def _inv_det_2x2(a, b, c, eps=1e-18):
    det = (a * c - b * b).clamp(min=eps)
    inv = 1.0 / det
    return c * inv, -b * inv, a * inv, det


def _quad(dx, dy, l00, l01, l11):
    return l00 * dx * dx + 2.0 * l01 * dx * dy + l11 * dy * dy


def o2_closed_form_batch(ui, uj, um, Si, Sj, Sm, alpha_i, alpha_j, valid_mask,
                         beta=0.0, agg="eq", eps=1e-18, clamp_c=False,
                         safe_mask=False, c0_floor=None, bound_guard=False):
    """ui/uj/um: (B,V,2); Si/Sj/Sm: (B,V,2,2); alpha: (B,); valid: (B,V) bool.

    Returns (alpha_star (B,), has_support (B,) bool, terms dict).
    """
    # accept (B,2) as a degenerate single-view (B,1,2)
    if ui.dim() == 2:
        ui = ui[:, None, :]
    if uj.dim() == 2:
        uj = uj[:, None, :]
    if um.dim() == 2:
        um = um[:, None, :]
    if Si.dim() == 3:
        Si = Si[:, None]
    if Sj.dim() == 3:
        Sj = Sj[:, None]
    if Sm.dim() == 3:
        Sm = Sm[:, None]
    if valid_mask.dim() == 1:
        valid_mask = valid_mask[:, None]
    ui = ui.to(torch.float64)
    uj = uj.to(torch.float64)
    um = um.to(torch.float64)
    Si = Si.to(torch.float64)
    Sj = Sj.to(torch.float64)
    Sm = Sm.to(torch.float64)
    ai = alpha_i.to(torch.float64).view(-1, 1)
    aj = alpha_j.to(torch.float64).view(-1, 1)

    dix = ui[..., 0] - um[..., 0]
    diy = ui[..., 1] - um[..., 1]
    djx = uj[..., 0] - um[..., 0]
    djy = uj[..., 1] - um[..., 1]

    si00, si01, si11 = Si[..., 0, 0], Si[..., 0, 1], Si[..., 1, 1]
    sj00, sj01, sj11 = Sj[..., 0, 0], Sj[..., 0, 1], Sj[..., 1, 1]
    sm00, sm01, sm11 = Sm[..., 0, 0], Sm[..., 0, 1], Sm[..., 1, 1]

    li00, li01, li11, det_i = _inv_det_2x2(si00, si01, si11, eps)
    lj00, lj01, lj11, det_j = _inv_det_2x2(sj00, sj01, sj11, eps)
    lm00, lm01, lm11, det_m = _inv_det_2x2(sm00, sm01, sm11, eps)

    one_pb = 1.0 + float(beta)
    inv_1pb = 1.0 / one_pb
    scale_2pb = 2.0 + float(beta)
    ratio_21 = scale_2pb * inv_1pb

    tsm00, tsm01, tsm11 = sm00 * inv_1pb, sm01 * inv_1pb, sm11 * inv_1pb

    pim00, pim01, pim11, det_im = _inv_det_2x2(si00 + tsm00, si01 + tsm01,
                                               si11 + tsm11, eps)
    q_im = _quad(dix, diy, pim00, pim01, pim11).clamp(0.0, 160.0)
    c_im = ratio_21 * torch.sqrt((det_i / det_im).clamp(min=0.0)) * torch.exp(-0.5 * q_im)

    pjm00, pjm01, pjm11, det_jm = _inv_det_2x2(sj00 + tsm00, sj01 + tsm01,
                                               sj11 + tsm11, eps)
    q_jm = _quad(djx, djy, pjm00, pjm01, pjm11).clamp(0.0, 160.0)
    c_jm = ratio_21 * torch.sqrt((det_j / det_jm).clamp(min=0.0)) * torch.exp(-0.5 * q_jm)

    A00 = li00 + lj00 + one_pb * lm00
    A01 = li01 + lj01 + one_pb * lm01
    A11 = li11 + lj11 + one_pb * lm11
    invA00, invA01, invA11, det_ijm = _inv_det_2x2(A00, A01, A11, eps)

    rx = li00 * dix + li01 * diy + lj00 * djx + lj01 * djy
    ry = li01 * dix + li11 * diy + lj01 * djx + lj11 * djy
    yx = invA00 * rx + invA01 * ry
    yy = invA01 * rx + invA11 * ry

    q_ijm = (_quad(dix - yx, diy - yy, li00, li01, li11)
             + _quad(djx - yx, djy - yy, lj00, lj01, lj11)
             + one_pb * _quad(yx, yy, lm00, lm01, lm11)).clamp(0.0, 160.0)
    c_ijm = (scale_2pb / torch.sqrt((det_m * det_ijm).clamp(min=eps))
             * torch.exp(-0.5 * q_ijm))

    alpha_v = ai * c_im + aj * c_jm - (ai * aj) * c_ijm
    # Degenerate near-plane projections (Sigma ~ 1e41) produce NaN / 1e42 in the
    # cross ratios.  A single such value poisons the round-level calibration mean
    # because the kernel replaces non-finite alpha with a0 (~1.0), which is far
    # above the typical 0.85.  Zero them here and let the mask handle it.
    bad_v = ~torch.isfinite(alpha_v)
    if bound_guard:
        # For beta >= 0 and any SPD covariances 0 <= c <= (2+beta)/(1+beta), so a
        # finite-but-huge c (7e42 was observed) is a numerical artefact too.
        cu_b = (2.0 + float(beta)) / (1.0 + float(beta)) + 0.01
        bad_v = bad_v | (c_im < 0.0) | (c_im > cu_b)
        bad_v = bad_v | (c_jm < 0.0) | (c_jm > cu_b)
        bad_v = bad_v | (c_ijm < 0.0) | (c_ijm > cu_b)
    if bool(bad_v.any()):
        alpha_v = torch.where(bad_v, torch.zeros_like(alpha_v), alpha_v)
    c_im = torch.nan_to_num(c_im, nan=0.0, posinf=0.0, neginf=0.0)
    c_jm = torch.nan_to_num(c_jm, nan=0.0, posinf=0.0, neginf=0.0)
    c_ijm = torch.nan_to_num(c_ijm, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_c:
        # Fix 2: for any beta >= 0 and any SPD covariances the three cross ratios
        # satisfy 0 <= c <= (2+beta)/(1+beta), hence alpha_v <= 4 for a_i,a_j<=1.
        # A larger value can only come from a numerically degenerate (near-plane)
        # covariance, and a handful of those poison the round-level `cal` mean.
        cu = torch.tensor((2.0 + float(beta)) / (1.0 + float(beta)),
                          dtype=alpha_v.dtype, device=alpha_v.device)
        c_im = c_im.clamp(min=0.0, max=float(cu))
        c_jm = c_jm.clamp(min=0.0, max=float(cu))
        c_ijm = c_ijm.clamp(min=0.0, max=float(cu))
        alpha_v = (ai * c_im + aj * c_jm - (ai * aj) * c_ijm).clamp(0.0, 4.0)

    m_f = valid_mask.to(torch.float64)
    if bool(bad_v.any()):
        m_f = m_f * (~bad_v).to(torch.float64)
    if safe_mask:
        # Fix 1: reject numerically degenerate 2-D covariances (near-plane blowup)
        for S_ in (Si.to(torch.float64), Sj.to(torch.float64), Sm.to(torch.float64)):
            s00_, s01_, s11_ = S_[..., 0, 0], S_[..., 0, 1], S_[..., 1, 1]
            det_ = s00_ * s11_ - s01_ * s01_
            tr_ = s00_ + s11_
            rad_ = torch.sqrt(((s00_ - s11_) ** 2 + 4 * s01_ * s01_).clamp(min=0.0))
            lmax_ = 0.5 * (tr_ + rad_)
            m_f = m_f * (torch.isfinite(s00_) & torch.isfinite(s01_)
                         & torch.isfinite(s11_) & (s00_ > 0.0) & (s11_ > 0.0)
                         & (lmax_ < 1e8)
                         & (det_ > 1e-12 * (s00_ * s11_))).to(torch.float64)
    if agg == "eq":
        w_v = m_f
    elif agg == "area":
        w_v = m_f * torch.sqrt(det_m.clamp(min=0.0))
    elif agg == "ecc":
        # ecc = the grid path's IMPLICIT weight.  The 9x9 grid is laid out on an
        # isotropic pixel box [-,+]^2 x sigma_cut*sqrt(lambda_max), so its per-view
        # denominator scales like 1/kappa = sqrt(lambda_min/lambda_max) instead of
        # the area.  Measured Spearman(den_v_grid, 1/kappa) = 0.98 on truck, so
        # this reproduces the grid's cross-view weighting in closed form.
        tr_m = sm00 + sm11
        diff_m = sm00 - sm11
        rad_m = torch.sqrt((diff_m * diff_m + 4.0 * sm01 * sm01).clamp(min=0.0))
        lam_max = (0.5 * (tr_m + rad_m)).clamp(min=eps)
        w_v = m_f * (torch.sqrt(det_m.clamp(min=eps)) / lam_max)
    elif agg == "ecc_floor":
        # Fix 3: 1/kappa with the positive floor that the 9x9 grid always has
        # (its centre sample contributes G_m(0)^2 = 1).
        tr_m = sm00 + sm11
        diff_m = sm00 - sm11
        rad_m = torch.sqrt((diff_m * diff_m + 4.0 * sm01 * sm01).clamp(min=0.0))
        lam_max = (0.5 * (tr_m + rad_m)).clamp(min=eps)
        lam_min = (0.5 * (tr_m - rad_m)).clamp(min=eps)
        c0 = 0.15 if c0_floor is None else float(c0_floor)
        w_v = m_f * (c0 + (1.0 - c0)
                     * torch.sqrt((lam_min / lam_max).clamp(0.0, 1.0)))
    else:
        raise ValueError(f"unknown agg: {agg}")
    w_sum = w_v.sum(dim=1)
    has_support = w_sum > 0.0
    a0 = (ai + aj - ai * aj).squeeze(1)
    alpha_fit = (alpha_v * w_v).sum(dim=1) / w_sum.clamp(min=eps)
    alpha_star = torch.where(has_support, alpha_fit, a0)
    return alpha_star, has_support, {"c_im": c_im, "c_jm": c_jm,
                                     "c_ijm": c_ijm, "alpha_v": alpha_v,
                                     "w_v": w_v}


# --------------------------------------------------------------------------- #
# dense-grid reference integrator (only used by the unit test / phase 1)
# --------------------------------------------------------------------------- #
def dc_color_fit(du_i, du_j, Si, Sj, Sm, ai, aj, ci, cj, valid_mask,
                 beta=0.0, eps=1e-18):
    """Closed-form DC (RGB) refit for the merged splat, per (pair, view).

    The merged splat must reproduce the transmittance-weighted colour of the two
    sources over its own footprint:

        c_m = argmin_c ∫ G_m (c - c_ref)^2,   c_ref = (a_i G_i c_i + a_j G_j c_j)
                                                    / (a_i G_i + a_j G_j)

    => c_m = (a_i J_i c_i + a_j J_j c_j) / (a_i J_i + a_j J_j),
       J_k = ∫ G_m G_k = 2 pi / sqrt(det(L_m + L_k)) * exp(-1/2 du_k' (Sm+Sk)^-1 du_k)

    The 2*pi cancels, so only sqrt(det(Sm+Sk)) and the quadratic form are needed.
    Returns (per-element colour (K,3), per-element weight (K,)).
    """
    du_i = du_i.to(torch.float64)
    du_j = du_j.to(torch.float64)
    Si = Si.to(torch.float64)
    Sj = Sj.to(torch.float64)
    Sm = Sm.to(torch.float64)
    ci = ci.to(torch.float64)
    cj = cj.to(torch.float64)
    ai = ai.to(torch.float64).reshape(-1)
    aj = aj.to(torch.float64).reshape(-1)

    def _w_k(du, Sk):
        a00 = Sm[..., 0, 0] + Sk[..., 0, 0]
        a01 = Sm[..., 0, 1] + Sk[..., 0, 1]
        a11 = Sm[..., 1, 1] + Sk[..., 1, 1]
        i00, i01, i11, det = _inv_det_2x2(a00, a01, a11, eps)
        q = _quad(du[..., 0], du[..., 1], i00, i01, i11).clamp(0.0, 160.0)
        return torch.sqrt(det.clamp(min=0.0)) * torch.exp(-0.5 * q)

    wi = _w_k(du_i, Si)
    wj = _w_k(du_j, Sj)
    ni = (ai * wi)[:, None]
    nj = (aj * wj)[:, None]
    den = (ni + nj).clamp(min=eps)
    col = (ni * ci + nj * cj) / den
    bad = ~torch.isfinite(col).all(-1)
    if bool(bad.any()):
        col = torch.where(bad[:, None], torch.zeros_like(col), col)
    m = valid_mask.to(torch.float64).reshape(-1)
    return col, m * (~bad).to(torch.float64)


def grid_reference(u, S, alpha, beta, n=385, sig_span=4.0):
    """Integrate the continuous objective with a dense, per-element scaled grid.

    u: (M,3,2) centres, S: (M,3,2,2) covs, alpha: (M,2).
    The grid spans +-sig_span * sigma_max of the MERGED gaussian, so the
    resolution adapts to each element (trapezoid error ~ (dx/sigma)^2 / 12).
    """
    u = u.to(torch.float64)
    S = S.to(torch.float64)
    M = u.shape[0]
    out = torch.empty(M, dtype=torch.float64, device=u.device)
    inv = []
    for k in range(3):
        a, b, c = S[:, k, 0, 0], S[:, k, 0, 1], S[:, k, 1, 1]
        det = (a * c - b * b).clamp(min=1e-18)
        inv.append((c / det, -b / det, a / det))
    smax = torch.sqrt(torch.clamp(S[:, 2, 0, 0] + S[:, 2, 1, 1], min=1e-12))
    for m in range(M):
        half = float(sig_span) * float(smax[m])
        g = torch.linspace(-half, half, n, dtype=torch.float64, device=u.device)
        gx, gy = torch.meshgrid(g, g, indexing="ij")
        px, py = gx[None, :, :], gy[None, :, :]

        def G(k):
            dx = px - (u[m, k, 0] - u[m, 2, 0])
            dy = py - (u[m, k, 1] - u[m, 2, 1])
            i00, i01, i11 = inv[k][0][m], inv[k][1][m], inv[k][2][m]
            return torch.exp(-0.5 * (i00 * dx * dx + 2 * i01 * dx * dy + i11 * dy * dy))

        Gi, Gj, Gm = G(0), G(1), G(2)
        a_ref = alpha[m, 0] * Gi + alpha[m, 1] * Gj - alpha[m, 0] * alpha[m, 1] * Gi * Gj
        w = Gm ** float(beta)
        num = (w * Gm * a_ref).sum()
        den = (w * Gm * Gm).sum().clamp(min=1e-300)
        out[m] = num / den
    return out
