#!/usr/bin/env python3
"""Per-pair eigen-axis closed-form O2cal  (pair_o2cal.md).

Zero global camera, zero voxel grid, zero grid sampling: the merged splat's own
principal axes give a local orthographic observation frame, and the resulting 2-D
projections go straight into the continuous L2 closed form.

This module is deliberately SELF-CONTAINED: it imports nothing from
o2_auto_virt / pics / router / o2gpu.  Everything is elementwise float64, so no
cuBLAS / cuSOLVER is ever initialised.

Conventions (must match o2gpu):
  * `sc` is the LINEAR half-axis scale (not log), `q` is (w, x, y, z).
  * The gaussian covariance is  R diag(sc^2) R^T  with R = quat_to_rot(q).
  * A view is a pair of orthonormal tangent vectors (t1, t2) in world space;
    the observation direction is t1 x t2.
  * Pixel scaling gamma = f_px / d_eff is applied to the whole projection, and
    alpha* is invariant to a uniform gamma, so E1 (gamma = 1, eps = 0) is
    scale-free by construction.
"""
from __future__ import annotations

import math

import torch

from o2_analytic import o2_closed_form_batch


# --------------------------------------------------------------------------- #
# local geometry (self-contained, no external imports)
# --------------------------------------------------------------------------- #
def quat_to_rot(q):
    """(B,4) (w,x,y,z) -> (B,3,3); COLUMNS are the gaussian axes."""
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


def axes_desc(sc, q):
    """Half-axis scales sorted DESCENDING plus the matching axis columns.

    Returns (axes (B,3,3) with columns e1,e2,e3 and s1 >= s2 >= s3, s (B,3)).
    """
    R = quat_to_rot(q)
    order = torch.argsort(-sc, dim=1)
    axes = torch.gather(R, 2, order[:, None, :].expand(-1, 3, -1))
    s = torch.gather(sc, 1, order)
    return axes, s


def qform(axes, s2, t1, t2):
    """Projected 2x2 covariance of sum_k s2_k e_k e_k^T onto (t1, t2).

    axes (B,3,3) columns = e_k, s2 (B,3) squared scales, t1/t2 (B,3).
    Returns (s00, s01, s11), all (B,).  Pure elementwise: no matmul.
    """
    a1 = (axes * t1[:, None, :]).sum(-1)          # (B,3) = e_k . t1
    a2 = (axes * t2[:, None, :]).sum(-1)
    s00 = (s2 * a1 * a1).sum(-1)
    s01 = (s2 * a1 * a2).sum(-1)
    s11 = (s2 * a2 * a2).sum(-1)
    return s00, s01, s11


def _du(dmu, t1, t2, gamma):
    return torch.stack([(dmu * t1).sum(-1), (dmu * t2).sum(-1)], -1) * gamma


def _stack2(s00, s01, s11, gamma2, eps):
    B = s00.shape[0]
    S = torch.empty((B, 2, 2), dtype=s00.dtype, device=s00.device)
    S[:, 0, 0] = s00 * gamma2 + eps
    S[:, 0, 1] = s01 * gamma2
    S[:, 1, 0] = s01 * gamma2
    S[:, 1, 1] = s11 * gamma2 + eps
    return S


# --------------------------------------------------------------------------- #
# view sets
# --------------------------------------------------------------------------- #
def _cube_views(B, dtype, dev):
    """The 8 (1,1,1)/sqrt(3) cube directions with deterministic tangent bases."""
    dirs = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                v = torch.tensor([sx, sy, sz], dtype=dtype, device=dev)
                dirs.append(v / v.norm())
    out = []
    for d in dirs:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=dev)
        if abs(float((d * up).sum())) > 0.9:
            up = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=dev)
        t1 = torch.cross(up, d, dim=-1)
        t1 = t1 / t1.norm()
        t2 = torch.cross(d, t1, dim=-1)
        out.append((t1[None, :].expand(B, -1), t2[None, :].expand(B, -1)))
    return out


def _eigen_views(axes, mode, psi_deg=35.0):
    """Return [(t1, t2)] for the requested variant, in the merged splat's frame."""
    e1 = axes[:, :, 0]
    e2 = axes[:, :, 1]
    e3 = axes[:, :, 2]
    if mode in ("E0", "E0cube"):
        return _cube_views(e1.shape[0], e1.dtype, e1.device)
    if mode in ("E1", "E2"):
        return [(e1, e2)]
    if mode == "E3":
        return [(e1, e2), (e1, e3), (e2, e3)]
    if mode == "E4":
        psi = math.radians(float(psi_deg))
        c, s = math.cos(psi), math.sin(psi)
        views = [(e1, e2)]
        for phi_deg in (0.0, 90.0, 180.0, 270.0):
            ph = math.radians(phi_deg)
            cp, sp = math.cos(ph), math.sin(ph)
            # tangent basis of the tilted view, orthogonal by construction
            t1 = -s * e3 + c * (cp * e1 + sp * e2)
            t2 = -sp * e1 + cp * e2
            views.append((t1, t2))
        return views
    raise ValueError(f"unknown eigen mode: {mode}")


def o2_eigen_batch(mu_i, mu_j, mu_m, sc_i, sc_j, sc_m, q_i, q_j, q_m,
                   alpha_i, alpha_j, mode="E1", gamma=1.0, eps_ewa=0.0,
                   beta=0.0, w_mode="area", psi_deg=35.0, eps=1e-18):
    """Per-pair closed-form opacity from local eigen-axis views.

    All inputs are for the SAME pair set, in merge order.
    Returns (alpha_star (B,), has_support (B,) all-True, wsum (B,)).
    """
    dtype = torch.float64
    for t in (mu_i, mu_j, mu_m, sc_i, sc_j, sc_m, q_i, q_j, q_m):
        if t.dtype != dtype:
            raise TypeError("eigen solver requires float64 inputs")
    axes_m, s_m = axes_desc(sc_m, q_m)
    axes_i, _s_i = axes_desc(sc_i, q_i)
    axes_j, _s_j = axes_desc(sc_j, q_j)
    views = _eigen_views(axes_m, mode, psi_deg)

    dmu_i = (mu_i - mu_m).to(dtype)
    dmu_j = (mu_j - mu_m).to(dtype)
    s2i, s2j, s2m = sc_i * sc_i, sc_j * sc_j, s_m * s_m
    g = float(gamma)
    g2 = g * g

    du_i, du_j, Si, Sj, Sm = [], [], [], [], []
    for t1, t2 in views:
        du_i.append(_du(dmu_i, t1, t2, g))
        du_j.append(_du(dmu_j, t1, t2, g))
        Si.append(_stack2(*qform(axes_i, s2i, t1, t2), g2, eps_ewa))
        Sj.append(_stack2(*qform(axes_j, s2j, t1, t2), g2, eps_ewa))
        Sm.append(_stack2(*qform(axes_m, s2m, t1, t2), g2, eps_ewa))

    du_i = torch.stack(du_i, 1)
    du_j = torch.stack(du_j, 1)
    Si = torch.stack(Si, 1)
    Sj = torch.stack(Sj, 1)
    Sm = torch.stack(Sm, 1)
    V = du_i.shape[1]
    valid = torch.ones(du_i.shape[0], V, dtype=torch.bool, device=du_i.device)

    _, _, terms = o2_closed_form_batch(
        du_i, du_j, torch.zeros_like(du_i), Si, Sj, Sm,
        alpha_i, alpha_j, valid, beta=beta, agg="eq")
    alpha_v = terms["alpha_v"]                       # (B,V) per-view solution

    # view weights
    if w_mode == "area":
        w = torch.sqrt(torch.clamp(Sm[:, :, 0, 0] * Sm[:, :, 1, 1]
                                   - Sm[:, :, 0, 1] * Sm[:, :, 1, 0], min=0.0))
    elif w_mode == "eq":
        w = torch.ones_like(Sm[:, :, 0, 0])
    elif w_mode == "det":
        w = torch.clamp(Sm[:, :, 0, 0] * Sm[:, :, 1, 1]
                        - Sm[:, :, 0, 1] * Sm[:, :, 1, 0], min=0.0)
    else:
        raise ValueError(f"unknown w_mode: {w_mode}")
    wsum = w.sum(1)
    a0 = (alpha_i + alpha_j - alpha_i * alpha_j)
    a_star = torch.where(wsum > 0, (alpha_v * w).sum(1) / wsum.clamp(min=eps), a0)
    return a_star, torch.ones_like(wsum, dtype=torch.bool), wsum
