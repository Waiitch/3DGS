"""Torch/CUDA re-implementations of the NanoGS public-code operators.

Every function here mirrors one function of the official NumPy code
(`nanogs.utils.cost.full_cost_pairs`, `nanogs.utils.merge.moment_matching`,
`nanogs.simplification.knn_undirected_edges`, `greedy_pairs_from_edges`)
operation-for-operation, so that the CPU/GPU comparison isolates the backend.

Conventions (identical to the public code):
  * quaternions are [w, x, y, z], normalised, R = R(q) is *world -> local*,
    i.e. Sigma = R diag(v) R^T with v = s^2 + eps.
  * costs are symmetric, computed once per undirected edge (u < v).
  * the merge cost is the Monte-Carlo estimate of KL(p_mix || q_merge) with
    `n_mc` deterministic standard-normal samples plus lam_sh * ||sh_u - sh_v||^2.
"""
from __future__ import annotations

import numpy as np
import torch

LOG2PI = float(np.log(2.0 * np.pi))
TWO_PI_15 = float((2.0 * np.pi) ** 1.5)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q: (N,4) [w,x,y,z] -> R: (N,3,3).  Same algebra as splat_utils."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(M,3,3) -> (M,4) [w,x,y,z], normalised.  Branch-selected like the numpy code."""
    M = R.shape[0]
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    mask_tr = tr > 0
    mask_00 = (m00 > m11) & (m00 > m22) & (~mask_tr)
    mask_11 = (m11 > m22) & (~mask_tr) & (~mask_00)
    mask_22 = (~mask_tr) & (~mask_00) & (~mask_11)

    q = torch.empty((M, 4), dtype=R.dtype, device=R.device)
    for _ in range(4):
        pass  # keep flake quiet

    def assign(mask, S_fn, formulas):
        if mask.any():
            S = S_fn(mask)
            for col, fn in formulas:
                q[mask, col] = fn(mask, S)

    # trace branch
    assign(
        mask_tr,
        lambda m: torch.sqrt(tr[m] + 1.0) * 2.0,
        [
            (0, lambda m, S: 0.25 * S),
            (1, lambda m, S: (R[m, 2, 1] - R[m, 1, 2]) / S),
            (2, lambda m, S: (R[m, 0, 2] - R[m, 2, 0]) / S),
            (3, lambda m, S: (R[m, 1, 0] - R[m, 0, 1]) / S),
        ],
    )
    # m00 branch
    assign(
        mask_00,
        lambda m: torch.sqrt(1.0 + R[m, 0, 0] - R[m, 1, 1] - R[m, 2, 2]) * 2.0,
        [
            (0, lambda m, S: (R[m, 2, 1] - R[m, 1, 2]) / S),
            (1, lambda m, S: 0.25 * S),
            (2, lambda m, S: (R[m, 0, 1] + R[m, 1, 0]) / S),
            (3, lambda m, S: (R[m, 0, 2] + R[m, 2, 0]) / S),
        ],
    )
    # m11 branch
    assign(
        mask_11,
        lambda m: torch.sqrt(1.0 + R[m, 1, 1] - R[m, 0, 0] - R[m, 2, 2]) * 2.0,
        [
            (0, lambda m, S: (R[m, 0, 2] - R[m, 2, 0]) / S),
            (1, lambda m, S: (R[m, 0, 1] + R[m, 1, 0]) / S),
            (2, lambda m, S: 0.25 * S),
            (3, lambda m, S: (R[m, 1, 2] + R[m, 2, 1]) / S),
        ],
    )
    # m22 branch
    assign(
        mask_22,
        lambda m: torch.sqrt(1.0 + R[m, 2, 2] - R[m, 0, 0] - R[m, 1, 1]) * 2.0,
        [
            (0, lambda m, S: (R[m, 1, 0] - R[m, 0, 1]) / S),
            (1, lambda m, S: (R[m, 0, 2] + R[m, 2, 0]) / S),
            (2, lambda m, S: (R[m, 1, 2] + R[m, 2, 1]) / S),
            (3, lambda m, S: 0.25 * S),
        ],
    )
    n = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    return q / n


def sigma_from_scale_quat(R: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Sigma = R diag(s^2) R^T for already-computed R."""
    s2 = s * s
    Rd = R * s2[:, None, :]
    return Rd @ R.transpose(1, 2)


def sigma_from_s2(R: torch.Tensor, s2: torch.Tensor, noblas=False) -> torch.Tensor:
    """Sigma = R diag(s2) R^T; `noblas` avoids the cuBLAS batched GEMM."""
    Rd = R * s2[:, None, :]
    if not noblas:
        return Rd @ R.transpose(1, 2)
    cols = [[(Rd[..., i, :] * R[..., j, :]).sum(-1) for j in range(3)]
            for i in range(3)]
    return torch.stack([torch.stack(row, -1) for row in cols], -2)


def sigma_from_scale_quat_noblas(R: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Same value without cuBLAS: 9 explicit (N,3) dot products.

    The first `matmul` in a process costs ~0.7 s of cuBLAS handle creation, which
    lands inside whichever stage calls it.  The scale_norm+O2cal path can avoid
    cuBLAS entirely, so this variant keeps that 0.7 s off the critical path.
    """
    return sigma_from_s2(R, s * s, noblas=True)


def gauss_logpdf_diagrot(x, mu, R, invdiag, logdet):
    """x: (B,S,3), mu: (B,3), R: (B,3,3), invdiag/logdet: (B,3)/(B,)."""
    d = x - mu[:, None, :]
    y = torch.bmm(d, R)                       # (B,S,3)  d @ R
    quad = ((y * y) * invdiag[:, None, :]).sum(-1)
    return -0.5 * (3.0 * LOG2PI + logdet[:, None] + quad)


# --------------------------------------------------------------------------- #
# per-splat precomputation (identical quantities, computed once per pass
# instead of twice per edge as the numpy code does)
# --------------------------------------------------------------------------- #
def precompute(mu, sc, q, op, eps_cov: float = 1e-8):
    R = quat_to_rotmat(q)
    v = sc * sc + eps_cov
    vc = torch.clamp(v, min=1e-30)
    invdiag = 1.0 / vc
    logdet = torch.log(vc).sum(-1)
    w = TWO_PI_15 * op * sc.prod(-1) + 1e-12
    return {"R": R, "v": v, "invdiag": invdiag, "logdet": logdet, "w": w, "mu": mu}


# --------------------------------------------------------------------------- #
# P1: edge costs
# --------------------------------------------------------------------------- #
def edge_costs_block(u, v, st, sh, Z, lam_geo, lam_sh, eps_cov):
    """u,v: (B,) index tensors.  st = output of precompute()."""
    R, vv, invd, logd, w, mu = st["R"], st["v"], st["invdiag"], st["logdet"], st["w"], st["mu"]

    R_i, R_j = R[u], R[v]
    Rt_i, Rt_j = R_i.transpose(1, 2), R_j.transpose(1, 2)
    v_i, v_j = vv[u], vv[v]
    inv_i, inv_j = invd[u], invd[v]
    ld_i, ld_j = logd[u], logd[v]
    w_i, w_j = w[u], w[v]

    W = w_i + w_j
    W_safe = torch.where(W > 0.0, W, torch.ones_like(W))
    pi = (w_i / W_safe).clamp(1e-12, 1.0 - 1e-12)
    log_pi = torch.log(pi)
    log_pj = torch.log1p(-pi)

    mu_i, mu_j = mu[u], mu[v]
    mu_m = pi[:, None] * mu_i + (1.0 - pi)[:, None] * mu_j

    di = mu_i - mu_m
    dj = mu_j - mu_m
    odi = di[:, :, None] * di[:, None, :]
    odj = dj[:, :, None] * dj[:, None, :]

    Sig_i = torch.matmul(R_i * v_i[:, None, :], Rt_i)
    Sig_j = torch.matmul(R_j * v_j[:, None, :], Rt_j)

    Sig_m = pi[:, None, None] * (Sig_i + odi) + (1.0 - pi)[:, None, None] * (Sig_j + odj)
    eye = torch.eye(3, dtype=Sig_m.dtype, device=Sig_m.device)[None]
    Sig_m = 0.5 * (Sig_m + Sig_m.transpose(1, 2)) + eps_cov * eye

    logdet_m = torch.linalg.slogdet(Sig_m).logabsdet
    E_p_neglogq = 0.5 * (3.0 * LOG2PI + logdet_m + 3.0)

    std_i = torch.sqrt(torch.clamp(v_i, min=0.0))
    std_j = torch.sqrt(torch.clamp(v_j, min=0.0))
    Zi = Z[None, :, :] * std_i[:, None, :]          # (B,S,3)
    Zj = Z[None, :, :] * std_j[:, None, :]
    x_i = mu_i[:, None, :] + torch.bmm(Zi, Rt_i)
    x_j = mu_j[:, None, :] + torch.bmm(Zj, Rt_j)

    logNi_i = gauss_logpdf_diagrot(x_i, mu_i, R_i, inv_i, ld_i)
    logNj_i = gauss_logpdf_diagrot(x_i, mu_j, R_j, inv_j, ld_j)
    logp_i = torch.logaddexp(log_pi[:, None] + logNi_i, log_pj[:, None] + logNj_i)
    Ei = logp_i.mean(1)

    logNi_j = gauss_logpdf_diagrot(x_j, mu_i, R_i, inv_i, ld_i)
    logNj_j = gauss_logpdf_diagrot(x_j, mu_j, R_j, inv_j, ld_j)
    logp_j = torch.logaddexp(log_pi[:, None] + logNi_j, log_pj[:, None] + logNj_j)
    Ej = logp_j.mean(1)

    E_p_logp = pi * Ei + (1.0 - pi) * Ej
    geo = E_p_logp + E_p_neglogq

    if sh is not None and sh.shape[1]:
        diff = sh[u] - sh[v]
        c_sh = (diff * diff).sum(-1)
    else:
        c_sh = torch.zeros_like(geo)
    return lam_geo * geo + lam_sh * c_sh


def edge_costs_gpu(edges, st, sh, Z, lam_geo, lam_sh, block_edges, eps_cov=1e-8,
                   out=None):
    M = edges.shape[0]
    if out is None:
        out = torch.empty((M,), dtype=torch.float32, device=edges.device)
    for e0 in range(0, M, block_edges):
        e1 = min(M, e0 + block_edges)
        out[e0:e1] = edge_costs_block(edges[e0:e1, 0], edges[e0:e1, 1], st, sh,
                                      Z, lam_geo, lam_sh, eps_cov)
    return out


# --------------------------------------------------------------------------- #
# Alternative scorers (P1 of the scorer roadmap).  Same edge set, same stable
# sort, same greedy; only the per-edge score changes.  Definitions follow the
# project's CPU implementations exactly (`nanogs_variants2.geo_cost`):
#
#   scale_norm : ||mu_u-mu_v||^2 / mean_k(s_u,k^2 + s_v,k^2)
#   maha_fast  : d^T ((S_u+S_v)/2)^-1 d ,  S = R diag(max(s^2,1e-12)) R^T
#   kl         : the native NanoGS Monte-Carlo KL (default, unchanged)
# --------------------------------------------------------------------------- #
def _inv3(S):
    """Closed-form batch inverse (adjugate / det); mirrors night_metrics.inv3."""
    a, b, c = S[:, 0, 0], S[:, 0, 1], S[:, 0, 2]
    d, e, f = S[:, 1, 0], S[:, 1, 1], S[:, 1, 2]
    g, h, i = S[:, 2, 0], S[:, 2, 1], S[:, 2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    det = a * A + b * B + c * C
    det = torch.where(det.abs() < 1e-30, torch.full_like(det, 1e-30), det)
    out = torch.empty_like(S)
    out[:, 0, 0] = A
    out[:, 0, 1] = c * h - b * i
    out[:, 0, 2] = b * f - c * e
    out[:, 1, 0] = B
    out[:, 1, 1] = a * i - c * g
    out[:, 1, 2] = c * d - a * f
    out[:, 2, 0] = C
    out[:, 2, 1] = b * g - a * h
    out[:, 2, 2] = a * e - b * d
    return out / det[:, None, None]


def normal_variation(mnr, nbr, chunk=262144):
    """Per-gaussian normal variation: mean_j (1 - |n_i . n_j|).

    `mnr` is the minor axis (proxy for the local surface normal), `nbr` the KNN
    index matrix.  This is the cheapest detail feature that survived screening.
    """
    N, K = nbr.shape
    out = torch.empty(N, dtype=torch.float32, device=mnr.device)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        cos = (mnr[s:e, None, :] * mnr[nbr[s:e]]).sum(-1).abs()      # (B,K)
        out[s:e] = (1.0 - cos).mean(1)
    return out


def _sh_basis(dirs):
    """(K,3) unit directions -> (K,16) real SH basis, 3DGS ordering."""
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396)
    C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    out = torch.empty((dirs.shape[0], 16), dtype=dirs.dtype, device=dirs.device)
    out[:, 0] = C0
    out[:, 1] = -C1 * y
    out[:, 2] = C1 * z
    out[:, 3] = -C1 * x
    out[:, 4] = C2[0] * xy
    out[:, 5] = C2[1] * yz
    out[:, 6] = C2[2] * (2 * zz - xx - yy)
    out[:, 7] = C2[3] * xz
    out[:, 8] = C2[4] * (xx - yy)
    out[:, 9] = C3[0] * y * (3 * xx - yy)
    out[:, 10] = C3[1] * xy * z
    out[:, 11] = C3[2] * y * (4 * zz - xx - yy)
    out[:, 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    out[:, 13] = C3[4] * x * (4 * zz - xx - yy)
    out[:, 14] = C3[5] * z * (xx - yy)
    out[:, 15] = C3[6] * x * (xx - 3 * yy)
    return out


def fixed_dirs(k):
    """k well-spread unit directions (tetrahedron for 4, cube corners for 8)."""
    if k == 4:
        v = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                         dtype=torch.float32)
    elif k == 8:
        v = torch.tensor([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1)
                          for sz in (1, -1)], dtype=torch.float32)
    else:
        raise ValueError("k must be 4 or 8")
    return v / v.norm(dim=-1, keepdim=True)


def view_colour_descriptor(sh, k, chunk=262144):
    """(N,48) SH -> (N, k, 3) colours evaluated at k fixed directions.

    Layout of `sh` in this project: f_dc(3) then f_rest(45) stored channel-major
    (R 0-14, G 15-29, B 30-44).
    """
    N = sh.shape[0]
    basis = _sh_basis(fixed_dirs(k).to(sh.device)).to(torch.float32)   # (k,16)
    out = torch.empty((N, k, 3), dtype=torch.float32, device=sh.device)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d = sh[s:e, :3]
        coef = torch.empty((e - s, 16, 3), dtype=torch.float32, device=sh.device)
        coef[:, 0, :] = d
        coef[:, 1:, 0] = sh[s:e, 3:18]
        coef[:, 1:, 1] = sh[s:e, 18:33]
        coef[:, 1:, 2] = sh[s:e, 33:48]
        out[s:e] = torch.einsum("kc,ncd->nkd", basis, coef) + 0.5
    return out.clamp_(0.0, 1.0)


def precompute_scorer(mu, sc, q, need_sig=False, noblas=False, need_axes=False):
    """Per-splat quantities for the geometry-only scorers.

    Only what the requested scorer needs is built: `scale_norm` wants the mean
    squared scale, `maha_fast` additionally wants the 3x3 covariance (in float64,
    because in float32 the smallest eigenvalue of a flat splat is lost and the
    top-1% edge overlap against the CPU reference drops from 1.00 to 0.84).
    """
    s2 = sc * sc
    out = {"mu": mu, "s2": s2, "s2mean3": s2.sum(-1) / 3.0}
    if need_axes:
        # per-splat principal axes: the major axis is the scale column with the
        # largest s, the minor axis is a cheap proxy for the local surface normal
        R = quat_to_rotmat(q)
        N = mu.shape[0]
        idx = torch.arange(N, device=mu.device)
        out["maj"] = R[idx, :, sc.argmax(-1)]
        out["mnr"] = R[idx, :, sc.argmin(-1)]
        out["smaj"] = sc.max(-1).values
    if need_sig:
        R64 = quat_to_rotmat(q).double()
        s2c64 = torch.clamp(s2, min=1e-12).double()
        out["Sig64"] = sigma_from_s2(R64, s2c64, noblas=noblas)
    return out


def edge_costs_scorer_block(u, v, st, pst, sh, lam_geo, lam_sh, scorer, lam_risk=0.0,
                            lam_axis=0.0, lam_normal=0.0, lam_sratio=0.0,
                            lam_plane=0.0, rho=None, lam_cg=0.0, tau=0.5,
                            cg_scale=1.0, lam_dnv=0.0, dnv_scale=1.0,
                            lam_e4=0.0, e4_scale=1.0, lam_vis=0.0):
    mu = st["mu"]
    d = mu[u] - mu[v]
    if scorer == "scale_norm":
        d2 = (d * d).sum(-1)
        var = pst["s2mean3"][u] + pst["s2mean3"][v]
        geo = d2 / torch.clamp(var, min=1e-20)
    elif scorer == "maha_fast":
        # float64 for the covariance and its inverse: S = (S_u+S_v)/2 can be
        # extremely ill-conditioned for flat splats, and in float32 the adjugate
        # and the determinant disagree with the CPU reference by 1e27 on those
        # edges.  The per-splat covariance is precomputed once per pass, so the
        # inner loop is gather + average + closed-form inverse.
        Su = pst["Sig64"][u]
        Sv = pst["Sig64"][v]
        S = 0.5 * (Su + Sv)
        Sinv = _inv3(S)
        dd = d.double()
        # d^T S^-1 d, written as a sum of 9 elementwise products so the maha path
        # also avoids cuBLAS (einsum with 2 operands lowers to a batched GEMM)
        quad = (dd[:, 0] * (Sinv[:, 0, 0] * dd[:, 0] + Sinv[:, 0, 1] * dd[:, 1]
                            + Sinv[:, 0, 2] * dd[:, 2])
                + dd[:, 1] * (Sinv[:, 1, 0] * dd[:, 0] + Sinv[:, 1, 1] * dd[:, 1]
                              + Sinv[:, 1, 2] * dd[:, 2])
                + dd[:, 2] * (Sinv[:, 2, 0] * dd[:, 0] + Sinv[:, 2, 1] * dd[:, 1]
                              + Sinv[:, 2, 2] * dd[:, 2]))
        geo = torch.clamp(quad, min=0.0).to(torch.float32)
    else:
        raise ValueError(f"unknown scorer {scorer}")
    c_dc = None
    if sh is not None and sh.shape[1]:
        diff = sh[u] - sh[v]
        c_sh = (diff * diff).sum(-1)
        if lam_risk > 0.0:
            c_dc = torch.linalg.vector_norm(diff[:, 0:3], dim=1)
    else:
        c_sh = torch.zeros_like(geo)
    cost = lam_geo * geo + lam_sh * c_sh
    if lam_axis > 0.0 and "maj" in pst:
        # two flat splats merged across their orientation destroy the shape;
        # |cos| between the major axes is a cheap image-free proxy
        c = (pst["maj"][u] * pst["maj"][v]).sum(-1).abs()
        cost = cost + lam_axis * (1.0 - c)
    if lam_normal > 0.0 and "mnr" in pst:
        # the minor axis is a proxy for the local surface normal: if the line
        # joining the two centres runs ALONG the normal, the merge bridges a
        # surface/edge instead of sliding along it
        dn = (d * pst["mnr"][u]).sum(-1) / torch.clamp(d.norm(dim=-1), min=1e-12)
        cost = cost + lam_normal * dn * dn
    if lam_plane > 0.0 and "mnr" in pst and rho is not None:
        # normalized surface-crossing: distance of v from u's local plane in
        # LOCAL-SPACING units, so dense and sparse regions are comparable
        pl = (d * pst["mnr"][u]).sum(-1).abs() / torch.clamp(
            0.5 * (rho[u] + rho[v]), min=1e-9)
        cost = cost + lam_plane * pl
    if lam_cg > 0.0 and "mnr" in pst and rho is not None and sh is not None:
        # CONDITIONAL colour veto: colour only decides among pairs that are
        # already nearly coplanar.  A global colour penalty rewrites the whole
        # ordering (and failed before); this one is gated by geometry.
        pl = (d * pst["mnr"][u]).sum(-1).abs() / torch.clamp(
            0.5 * (rho[u] + rho[v]), min=1e-9)
        gate = torch.exp(-(pl / max(tau, 1e-6)) ** 2)
        if "viewcol" in pst:
            # E1: multi-view SH descriptor.  RMS colour difference over the
            # fixed direction set, same units as the DC version, and cheaper to
            # gather than a 48-column SH slice.
            dv = pst["viewcol"][u] - pst["viewcol"][v]
            dcol = (dv * dv).mean(-1).sum(-1).sqrt()
        else:
            dcol = torch.linalg.vector_norm((sh[u] - sh[v])[:, 0:3], dim=1)
        if cg_scale > 0.0:
            dcol = dcol / cg_scale          # robust unit normalisation
        cost = cost + lam_cg * gate * dcol
    if lam_vis > 0.0 and "vis" in pst:
        # cheap visibility term (no rendering): penalise merging gaussians that
        # the fit views actually see, i.e. prefer peripheral merges
        cost = cost + lam_vis * torch.maximum(pst["vis"][u], pst["vis"][v])
    if lam_e4 > 0.0 and "dnv" in pst and "vol" in pst:
        # E4: "two big gaussians sitting in a low-detail area" is the risky case
        # (an over-smooth patch / under-reconstruction).  The product matters:
        # volume alone is just "big is expensive".
        q = (torch.maximum(pst["vol"][u], pst["vol"][v])
             * (1.0 - torch.maximum(pst["dnv"][u], pst["dnv"][v])))
        if e4_scale > 0.0:
            q = q / e4_scale
        cost = cost + lam_e4 * q
    if lam_dnv > 0.0 and "dnv" in pst and "mnr" in pst:
        # detail-aware merge risk: region importance (normal variation) x the
        # pair-specific normal disagreement.  The product form avoids
        # over-protecting detail-rich but highly redundant regions.
        pn = 1.0 - (pst["mnr"][u] * pst["mnr"][v]).sum(-1).abs()
        dn = torch.maximum(pst["dnv"][u], pst["dnv"][v]) * pn
        if dnv_scale > 0.0:
            dn = dn / dnv_scale
        cost = cost + lam_dnv * dn
    if lam_sratio > 0.0 and "smaj" in pst:
        # a big splat swallowing a much smaller one loses the smaller one's detail
        r = (torch.clamp(pst["smaj"][u], min=1e-12)
             / torch.clamp(pst["smaj"][v], min=1e-12))
        cost = cost + lam_sratio * r.log().abs()
    if c_dc is not None:
        # "front-loaded" merge risk: a pair whose DC colours differ a lot is
        # exactly what a single merged gaussian cannot represent, so bias the
        # greedy towards other candidates.  One extra (M,3) gather, no merge.
        cost = cost + lam_risk * c_dc
    return cost


def edge_costs_scorer_gpu(edges, st, pst, sh, lam_geo, lam_sh, scorer, block_edges,
                          lam_risk=0.0, lam_axis=0.0, lam_normal=0.0, lam_sratio=0.0,
                            lam_plane=0.0, rho=None, lam_cg=0.0, tau=0.5,
                            cg_scale=1.0, lam_dnv=0.0, dnv_scale=1.0,
                            lam_e4=0.0, e4_scale=1.0, lam_vis=0.0):
    M = edges.shape[0]
    out = torch.empty((M,), dtype=torch.float32, device=edges.device)
    for e0 in range(0, M, block_edges):
        e1 = min(M, e0 + block_edges)
        out[e0:e1] = edge_costs_scorer_block(edges[e0:e1, 0], edges[e0:e1, 1],
                                             st, pst, sh, lam_geo, lam_sh, scorer,
                                             lam_risk, lam_axis, lam_normal,
                                             lam_sratio, lam_plane, rho,
                                             lam_cg, tau, cg_scale,
                                             lam_dnv, dnv_scale,
                                             lam_e4, e4_scale, lam_vis)
    return out


# --------------------------------------------------------------------------- #
# P2: MPMM merge
# --------------------------------------------------------------------------- #
def jacobi_eigh3(A, sweeps=5):
    """Batched symmetric 3x3 eigendecomposition with fixed sweep count.

    Replaces `torch.linalg.eigh` so the process never initialises cuSOLVER (whose
    first call costs ~0.86 s, measured).  The cost is therefore uniform in the
    data, unlike cuSOLVER's Jacobi path.  Returns (eigvals asc, eigvecs as columns)
    to match torch's convention; the caller sorts descending anyway.
    """
    B = A.shape[0]
    a = A.clone()
    V = torch.zeros((B, 3, 3), dtype=A.dtype, device=A.device)
    V[:, 0, 0] = V[:, 1, 1] = V[:, 2, 2] = 1.0
    for _ in range(sweeps):
        for p, q in ((0, 1), (0, 2), (1, 2)):
            apq = a[:, p, q]
            app = a[:, p, p]
            aqq = a[:, q, q]
            theta = (aqq - app) / (2.0 * apq)
            t = torch.sign(theta) / (theta.abs() + torch.sqrt(theta * theta + 1.0))
            t = torch.where(apq == 0, torch.zeros_like(t), t)
            c = 1.0 / torch.sqrt(t * t + 1.0)
            s = t * c
            # zero the rotated off-diagonal and update the two diagonals
            a[:, p, p] = app - t * apq
            a[:, q, q] = aqq + t * apq
            a[:, p, q] = torch.zeros_like(apq)
            a[:, q, p] = torch.zeros_like(apq)
            k = 3 - p - q                      # the remaining index
            akp = a[:, k, p].clone()
            akq = a[:, k, q].clone()
            a[:, k, p] = c * akp - s * akq
            a[:, p, k] = a[:, k, p]
            a[:, k, q] = s * akp + c * akq
            a[:, q, k] = a[:, k, q]
            vkp = V[:, :, p].clone()
            vkq = V[:, :, q].clone()
            V[:, :, p] = c[:, None] * vkp - s[:, None] * vkq
            V[:, :, q] = s[:, None] * vkp + c[:, None] * vkq
    evals = torch.stack([a[:, 0, 0], a[:, 1, 1], a[:, 2, 2]], 1)
    return evals, V


def det3(R):
    """3x3 determinant without cuSOLVER."""
    return (R[:, 0, 0] * (R[:, 1, 1] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 1])
            - R[:, 0, 1] * (R[:, 1, 0] * R[:, 2, 2] - R[:, 1, 2] * R[:, 2, 0])
            + R[:, 0, 2] * (R[:, 1, 0] * R[:, 2, 1] - R[:, 1, 1] * R[:, 2, 0]))


def merge_clusters(mu, sc, q, op, sh, groups, jacobi=True):
    """One-shot moment matching of m>=2 gaussians per cluster (plan route B).

    `groups` is (C, m) int64.  Generalises `merge_components` from 2 sources to m:

        W   = sum_i m_i,             m_i = (2 pi)^{3/2} alpha_i prod_k s_ik
        muC = sum_i m_i mu_i / W
        SigC= sum_i m_i [Sig_i + d_i d_i^T] / W,   d_i = mu_i - muC
        alphaC = 1 - prod_i (1 - alpha_i)          (temporary; refined later)
    """
    m = groups.shape[1]
    dt = torch.float64                    # 4-source scales can overflow float32
    mu_g = mu[groups].to(dt)              # (C,m,3)
    sc_g = sc[groups].to(dt)              # (C,m,3)
    q_g = q[groups]                       # (C,m,4)
    a_g = op[groups].to(dt)               # (C,m)

    R = quat_to_rotmat(q_g.reshape(-1, 4))          # (C*m,3,3)
    Sig = sigma_from_s2(R, (sc_g.reshape(-1, 3) ** 2), noblas=True)
    Sig = Sig.reshape(-1, m, 3, 3)                  # (C,m,3,3)

    w = TWO_PI_15 * a_g * sc_g.prod(-1) + 1e-12     # (C,m)
    w = torch.nan_to_num(w, nan=1e-12, posinf=1e12)   # sc.prod can overflow
    W = torch.clamp(w.sum(1), min=1e-12)            # (C,)

    mu_c = (w[..., None] * mu_g).sum(1) / W[:, None]
    d = mu_g - mu_c[:, None, :]                     # (C,m,3)
    od = d[..., :, None] * d[..., None, :]          # (C,m,3,3)
    Sig_c = (w[..., None, None] * (Sig + od)).sum(1) / W[:, None, None]
    eye = torch.eye(3, dtype=dt, device=mu.device)[None]
    Sig_c = 0.5 * (Sig_c + Sig_c.transpose(1, 2)) + eye * 1e-8

    evals, evecs = jacobi_eigh3(Sig_c) if jacobi else torch.linalg.eigh(Sig_c)
    evals = torch.clamp(evals, min=1e-18)
    order = torch.argsort(evals, dim=1, descending=True)
    evals = torch.gather(evals, 1, order)
    evecs = torch.gather(evecs, 2, order[:, None, :].expand(-1, 3, -1))
    det = det3(evecs) if jacobi else torch.linalg.det(evecs)
    flip = det < 0
    if flip.any():
        evecs = evecs.clone()
        evecs[flip, :, 2] *= -1.0
    scales = torch.sqrt(evals)
    quat = rotmat_to_quat(evecs)

    op_c = (1.0 - torch.prod(1.0 - a_g.clamp(0.0, 1.0), dim=1)).clamp(0.0, 1.0)
    if sh is not None and sh.shape[1]:
        sh_c = ((w[..., None] * sh[groups].to(dt)).sum(1) / W[:, None]).to(mu.dtype)
    else:
        sh_c = None
    return {"mu_m": mu_c.to(mu.dtype), "sc_m": scales.to(mu.dtype),
            "q_m": quat, "op_m": op_c.to(torch.float32), "sh_m": sh_c}


def cluster_alpha(mu, sc, q, op, comp, groups, cams, chunk=8192):
    """Peak-weight alpha fit for clusters: alpha_C = sum_v sum_i w_iv / sum_v w_Cv.

    The cheap generalisation of O2cal's coverage fit to m sources (no grid
    integration): w = alpha / sqrt(det Sigma2), summed over the views where all
    m sources and the cluster project.
    """
    from o2gpu import _project

    C = groups.shape[0]
    m = groups.shape[1]
    N = mu.shape[0]
    num = torch.zeros(C, dtype=torch.float64, device=mu.device)
    den = torch.zeros_like(num)
    for s in range(0, C, chunk):
        sl = slice(s, min(C, s + chunk))
        g = groups[sl]
        _, S2c, vc = _project(comp["mu_m"][sl], comp["sc_m"][sl], comp["q_m"][sl],
                              cams, torch.float32, True)
        detc = (S2c[..., 0, 0] * S2c[..., 1, 1] - S2c[..., 0, 1] * S2c[..., 1, 0])
        ok = vc.clone()
        acc = torch.zeros(ok.shape, dtype=torch.float32, device=mu.device)
        for k in range(m):
            idx = g[:, k]
            _, S2k, vk = _project(mu[idx], sc[idx], q[idx], cams, torch.float32, True)
            detk = (S2k[..., 0, 0] * S2k[..., 1, 1] - S2k[..., 0, 1] * S2k[..., 1, 0])
            w = op[idx][:, None] / detk.clamp(min=1e-12).sqrt()
            acc += w * vk
            ok &= vk
        wc = 1.0 / detc.clamp(min=1e-12).sqrt()
        num[sl] = (acc * ok).double().sum(1)
        den[sl] = (wc * ok).double().sum(1)
    return torch.clamp(num / den.clamp(min=1e-12), 0.0, 1.0).to(torch.float32)


def merge_components(mu, sc, q, op, sh, pairs, timers=None, noblas=False, jacobi=False):
    """MPMM moment matching for `pairs`; no compaction (so O2 can use the parts).

    When `timers` is a dict the sub-stages are accumulated into it (P0 profiling).
    """
    import time as _t

    def _mark():
        return _t.perf_counter() if timers is not None else 0.0

    def _add(k, t0):
        if timers is not None:
            # sync so the number is GPU-inclusive, not launch time
            torch.cuda.synchronize()
            timers[k] = timers.get(k, 0.0) + (_t.perf_counter() - t0)

    i, j = pairs[:, 0], pairs[:, 1]
    t0 = _mark()
    mu_i, mu_j = mu[i], mu[j]
    s_i, s_j = sc[i], sc[j]
    a_i, a_j = op[i], op[j]

    _sig = sigma_from_scale_quat_noblas if noblas else sigma_from_scale_quat
    Sig_i = _sig(quat_to_rotmat(q[i]), s_i)
    Sig_j = _sig(quat_to_rotmat(q[j]), s_j)

    w_i = TWO_PI_15 * a_i * s_i.prod(-1) + 1e-12
    w_j = TWO_PI_15 * a_j * s_j.prod(-1) + 1e-12
    W = torch.clamp(w_i + w_j, min=1e-12)

    mu_m = (w_i[:, None] * mu_i + w_j[:, None] * mu_j) / W[:, None]
    di = mu_i - mu_m
    dj = mu_j - mu_m
    odi = di[:, :, None] * di[:, None, :]
    odj = dj[:, :, None] * dj[:, None, :]

    eye = torch.eye(3, dtype=mu.dtype, device=mu.device)[None]
    Sig = (w_i[:, None, None] * (Sig_i + odi) + w_j[:, None, None] * (Sig_j + odj)) / W[:, None, None]
    Sig = 0.5 * (Sig + Sig.transpose(1, 2)) + eye * 1e-8
    _add("m_weights_sig", t0)

    t0 = _mark()
    if jacobi:
        evals, evecs = jacobi_eigh3(Sig)
    else:
        evals, evecs = torch.linalg.eigh(Sig)
    _add("m_eigh", t0)

    t0 = _mark()
    evals = torch.clamp(evals, min=1e-18)
    order = torch.argsort(evals, dim=1, descending=True)
    evals = torch.gather(evals, 1, order)
    evecs = torch.gather(evecs, 2, order[:, None, :].expand(-1, 3, -1))

    det = det3(evecs) if jacobi else torch.linalg.det(evecs)
    flip = det < 0
    if flip.any():
        evecs = evecs.clone()
        evecs[flip, :, 2] *= -1.0

    scales = torch.sqrt(evals)
    _add("m_sort_flip", t0)

    t0 = _mark()
    quat = rotmat_to_quat(evecs)
    _add("m_quat", t0)

    t0 = _mark()
    op_m = a_i + a_j - a_i * a_j

    if sh is not None and sh.shape[1]:
        sh_m = (w_i[:, None] * sh[i] + w_j[:, None] * sh[j]) / W[:, None]
    else:
        sh_m = None
    _add("m_sh", t0)

    return {"mu_m": mu_m, "sc_m": scales, "q_m": quat, "op_m": op_m, "sh_m": sh_m}


def merge_assemble(mu, sc, q, op, sh, pairs, comp):
    """Compact the model with the pairs removed and the merged splats appended."""
    i, j = pairs[:, 0], pairs[:, 1]
    mu_m, scales = comp["mu_m"], comp["sc_m"]
    quat, op_m, sh_m = comp["q_m"], comp["op_m"], comp["sh_m"]

    N = mu.shape[0]
    used = torch.zeros(N, dtype=torch.bool, device=mu.device)
    used[i] = True
    used[j] = True
    keep = torch.nonzero(~used, as_tuple=False).flatten()

    mu2 = torch.cat([mu[keep], mu_m], 0)
    sc2 = torch.cat([sc[keep], scales], 0)
    q2 = torch.cat([q[keep], quat], 0)
    op2 = torch.cat([op[keep], op_m], 0)
    sh2 = torch.cat([sh[keep], sh_m], 0) if sh_m is not None else sh
    return mu2, sc2, q2, op2, sh2


def merge_pairs_gpu(mu, sc, q, op, sh, pairs):
    """pairs: (P,2) int index tensor.  Returns new activated arrays."""
    comp = merge_components(mu, sc, q, op, sh, pairs)
    return merge_assemble(mu, sc, q, op, sh, pairs, comp)


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #
def undirected_edges_gpu(nbr: torch.Tensor) -> torch.Tensor:
    """nbr: (N,k) directed kNN -> unique undirected (u<v) edges, lexicographic."""
    N, k = nbr.shape
    ii = torch.arange(N, device=nbr.device, dtype=torch.int64).repeat_interleave(k)
    jj = nbr.reshape(-1).to(torch.int64)
    u = torch.minimum(ii, jj)
    v = torch.maximum(ii, jj)
    mask = u != v
    u, v = u[mask], v[mask]
    key = u * int(N) + v
    key = torch.unique(key)          # sorted, unique
    u = torch.div(key, int(N), rounding_mode="floor")
    v = key % int(N)
    # int32 is enough for N < 2^31 and halves both the D2H transfer and the
    # numpy copy the greedy needs (10.3M edges: 165 MB int64 -> 82 MB int32)
    return torch.stack([u, v], 1).to(torch.int32)


def _greedy_impl(edges, order, N, P):
    """Maximal matching over the cost-ordered edge list.

    Defined at module level on purpose: numba cannot cache a closure created
    inside a function, so the previous nested version recompiled on every run
    (~0.29 s of the end-to-end time).
    """
    used = np.zeros(N, dtype=np.bool_)
    out = np.empty((min(order.shape[0], N // 2 + 1), 2), dtype=np.int32)
    cnt = 0
    for t in range(order.shape[0]):
        ei = order[t]
        u = edges[ei, 0]
        v = edges[ei, 1]
        if used[u] or used[v]:
            continue
        used[u] = True
        used[v] = True
        out[cnt, 0] = u
        out[cnt, 1] = v
        cnt += 1
        if P >= 0 and cnt >= P:
            break
    return out[:cnt]


try:
    from numba import njit as _njit
    _greedy_cached = _njit(cache=True)(_greedy_impl)
except Exception:                                     # pragma: no cover
    _greedy_cached = None


def greedy_numba_init():
    return _greedy_cached if _greedy_cached is not None else _greedy_impl

def _greedy_blocked_impl(edges, order, N, P, blk_of_node):
    """Blocked matching: each block is processed independently, then the
    cross-block edges are resolved serially.  Blocks are spatial, so almost all
    KNN edges stay inside one block and the result is close to the serial one.
    """
    nblk = int(blk_of_node.max()) + 1
    eblk = blk_of_node[edges[order, 0]]
    cross = eblk != blk_of_node[edges[order, 1]]
    cnt_blk = np.bincount(eblk, minlength=nblk)
    starts = np.zeros(nblk, dtype=np.int64)
    for b in range(1, nblk):
        starts[b] = starts[b - 1] + cnt_blk[b - 1]
    pos = starts.copy()
    idx_by_blk = np.empty(order.shape[0], dtype=np.int64)
    for t in range(order.shape[0]):
        if cross[t]:
            continue
        b = eblk[t]
        idx_by_blk[pos[b]] = order[t]
        pos[b] += 1
    used = np.zeros(N, dtype=np.bool_)
    out = np.empty((min(order.shape[0], N // 2 + 2), 2), dtype=np.int32)
    cnt = 0
    for b in range(nblk):
        s = starts[b]
        e = pos[b]
        for t in range(s, e):
            ei = idx_by_blk[t]
            u = edges[ei, 0]
            v = edges[ei, 1]
            if used[u] or used[v]:
                continue
            used[u] = True
            used[v] = True
            out[cnt, 0] = u
            out[cnt, 1] = v
            cnt += 1
            if P >= 0 and cnt >= P:
                return out[:cnt]
    for t in range(order.shape[0]):
        if not cross[t]:
            continue
        ei = order[t]
        u = edges[ei, 0]
        v = edges[ei, 1]
        if used[u] or used[v]:
            continue
        used[u] = True
        used[v] = True
        out[cnt, 0] = u
        out[cnt, 1] = v
        cnt += 1
        if P >= 0 and cnt >= P:
            break
    return out[:cnt]


try:
    from numba import njit as _njit2
    _greedy_blocked_cached = _njit2(cache=True)(_greedy_blocked_impl)
except Exception:                                     # pragma: no cover
    _greedy_blocked_cached = None


def greedy_blocked_init():
    return (_greedy_blocked_cached if _greedy_blocked_cached is not None
            else _greedy_blocked_impl)


def spatial_blocks(mu, nblk=8):
    """Octant-style partition of the nodes (2x2x2 by the bounding box)."""
    n = len(mu)
    lo = mu.min(0)
    hi = mu.max(0)
    rel = (mu - lo) / np.maximum(hi - lo, 1e-12)
    side = int(round(nblk ** (1.0 / 3.0)))
    q = np.clip((rel * side).astype(np.int64), 0, side - 1)
    return (q[:, 0] * side * side + q[:, 1] * side + q[:, 2]).astype(np.int64)