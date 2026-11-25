#!/usr/bin/env python3
"""Camera-free O2cal virtual rigs (`--o2-cams auto_virt`).

Implements the 5-step pure-GPU pre-extraction pipeline of
`camera_free_o2cal.md`:

  Step 1  core-point AABB + projection-scale self calibration
  Step 2  64^3 voxel occupancy, surface voxels, erosion distance field,
          interior-cavity vs outer-shell candidate positions (topology safe)
  Step 3  multi-direction candidate poses + coarse soft z-buffer visibility
          with the +0.3 I scale gate
  Step 4  submodular k-coverage greedy selection (B views)
  Step 5  returns cams in exactly `cams_fast.load_cams_fast`'s dict format so
          the rest of the O2cal pipeline is untouched.

Conventions (verified by `test_auto_virt_coord.py`):
  * world -> view matrix `w2v` maps world points into the COLMAP/3DGS camera
    frame: +X right, +Y down, +Z forward.
  * pixel x = fx * X/Z + W/2, pixel y = fy * Y/Z + H/2 (principal point at the
    image centre, which is what `o2gpu._project` assumes).
  * `scales` are *linear* (post-exp) scales and `opacities` are post-sigmoid
    values, matching `fastio.read_ply_fast`.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["extract_auto_virtual_cams", "look_at"]


# --------------------------------------------------------------------------
# camera helpers (identical maths to vircams.look_at, batched here)
# --------------------------------------------------------------------------
def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    """w2v for a camera at `eye` looking at `target` (COLMAP axes)."""
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        fwd = np.array([0.0, 0.0, 1.0])
    else:
        fwd = fwd / n
    up = np.asarray(up, np.float64)
    right = np.cross(-up, fwd)
    nn = np.linalg.norm(right)
    if nn < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / nn
    down = np.cross(fwd, right)
    R_c2w = np.stack([right, down, fwd], axis=1)
    w2v = np.eye(4)
    w2v[:3, :3] = R_c2w.T
    w2v[:3, 3] = -R_c2w.T @ eye
    return w2v


def _intr(fov_deg, W):
    fx = 0.5 * W / math.tan(math.radians(fov_deg) * 0.5)
    return fx


def _quantile_axis(x, qs):
    """Per-column quantiles of (n, d) tensor via one sort (no torch.quantile)."""
    n = x.shape[0]
    s, _ = torch.sort(x, dim=0)
    out = []
    for q in qs:
        k = min(max(int(round(q * (n - 1))), 0), n - 1)
        out.append(s[k])
    return out            # list of d-vectors


def _quat_to_mat(q):
    """(K,4) w,x,y,z -> (K,3,3) local->world, same as o2gpu._rot."""
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


def _dir_set(n_dirs):
    if n_dirs == 6:
        d = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif n_dirs == 8:
        s = 1.0 / math.sqrt(3.0)
        d = [(s * a, s * b, s * c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    else:
        raise ValueError("n_dirs must be 6 or 8")
    return np.asarray(d, np.float64)


def _look_at_up(eye, target, up_world):
    """w2v with an explicit world-up: zero-roll basis around the gravity axis."""
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    fwd = np.array([0.0, 0.0, 1.0]) if n < 1e-9 else fwd / n
    down = -np.asarray(up_world, np.float64)
    down = down - fwd * float(down @ fwd)          # perpendicular component
    dn = np.linalg.norm(down)
    if dn < 1e-9:
        down = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        dn = np.linalg.norm(down)
    down = down / max(dn, 1e-12)
    right = np.cross(down, fwd)
    right = right / max(np.linalg.norm(right), 1e-12)
    R_c2w = np.stack([right, down, fwd], axis=1)
    w2v = np.eye(4)
    w2v[:3, :3] = R_c2w.T
    w2v[:3, 3] = -R_c2w.T @ eye
    return w2v


def _gravity_axis(mu, sc, qq, n_sample=8192, ang_deg=12.0, lower_frac=0.0):
    """Dominant plane normal (gravity axis, up to sign) from flat gaussians.

    Pure elementwise GPU ops -- no matmul, so the --no-blas path stays cold-free.
    `lower_frac > 0` restricts the *second* vote to the lower shell along the
    first-pass axis, which locks onto the floor/road instead of tall walls
    (autovirt_v2.md Phase B2).
    """
    N = mu.shape[0]
    K = min(n_sample, N)
    idx = torch.linspace(0, N - 1, K).long().to(mu.device)
    s_sorted = torch.sort(sc[idx], dim=1).values          # ascending
    Rm = _quat_to_mat(qq[idx])
    amin = sc[idx].argmin(1)
    nrm = Rm.gather(2, amin[:, None, None].expand(-1, 3, 1)).squeeze(2)
    flat = (s_sorted[:, 0] / s_sorted[:, 1].clamp_min(1e-9)) < 0.3
    if int(flat.sum()) < 64:
        flat = torch.ones_like(flat)
    n = nrm[flat]
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # area weight: a large flat splat is a better floor/road sample
    wgt = (s_sorted[:, 2] * s_sorted[:, 1])[flat]
    wgt = (wgt / wgt.mean().clamp_min(1e-12)).clamp(0.05, 20.0)
    ctr_all = mu[idx][flat].mean(0)

    def _vote(nn, ww):
        cos_ang = math.cos(math.radians(ang_deg))
        best = None
        bestv = -1.0
        for s0 in range(0, nn.shape[0], 512):
            blk = nn[s0:s0 + 512]
            dots = (blk[:, None, :] * nn[None, :, :]).sum(-1).abs()   # (b,K)
            votes = ((dots > cos_ang).float() * ww[None, :]).sum(1)
            i = int(votes.argmax())
            if float(votes[i]) > bestv:
                bestv = float(votes[i])
                best = blk[i]
        u = best / best.norm().clamp_min(1e-9)
        rel = (mu[idx][flat] - ctr_all)
        if float((rel * u[None, :]).sum().item()) < 0:
            u = -u
        return u, bestv

    u, bestv = _vote(n, wgt)
    if lower_frac > 0.0:
        pts = mu[idx][flat]
        rel = pts - ctr_all
        h = (rel * u[None, :]).sum(1)
        lo = torch.quantile(h, 1.0 - float(lower_frac))
        m2 = h >= lo                                  # lower shell (ground side)
        if int(m2.sum()) >= 64:
            u, bestv = _vote(n[m2], wgt[m2])
    return u, bestv, float(flat.float().mean().item())


def _dirs_gravity(n_dirs, pitches_deg, u):
    """Zero-roll horizontal yaw ring around the gravity axis + optional pitches."""
    u = np.asarray(u, np.float64)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(tmp, u)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(u, e1)
    out = []
    for p in pitches_deg:
        pr = math.radians(p)
        for k in range(n_dirs):
            yaw = 2.0 * math.pi * k / n_dirs
            d = (math.cos(pr) * (math.cos(yaw) * e1 + math.sin(yaw) * e2)
                 + math.sin(pr) * u)
            out.append(d / max(np.linalg.norm(d), 1e-12))
    return np.asarray(out, np.float64)


def _unravel3(idx, R):
    """flat voxel index -> (n,3) ijk, row-major over (R,R,R)."""
    i = torch.div(idx, R * R, rounding_mode="floor")
    j = torch.div(idx - i * (R * R), R, rounding_mode="floor")
    k = idx - i * (R * R) - j * R
    return torch.stack([i, j, k], 1)


def _orient_normals(n_v, s_ijk, occ, R):
    """Flip each normal toward the side with more free voxels (M4/M5)."""
    occ_flat = occ.reshape(-1)
    V = n_v.shape[0]

    def _occ_at(q):
        m = ((q >= 0) & (q < R)).all(-1)
        j = q.clamp(0, R - 1)
        lin = (j[..., 0] * R + j[..., 1]) * R + j[..., 2]
        return torch.where(m, occ_flat[lin], torch.zeros_like(lin, dtype=torch.bool))

    fp = torch.zeros(V, device=n_v.device)
    fm = torch.zeros(V, device=n_v.device)
    for kk in (1, 2, 3):
        off = torch.round(n_v * kk).long()
        fp += (~_occ_at(s_ijk + off)).float()
        fm += (~_occ_at(s_ijk - off)).float()
    return torch.where((fm > fp)[:, None], -n_v, n_v)


def _snap_free(pos, vox_lo, Delta, R, free_flat, dist_flat, d_safe, win=2):
    """Snap world positions to the nearest safe free voxel (window search)."""
    dev = pos.device
    K = pos.shape[0]
    g = torch.floor((pos - vox_lo[None, :]) / Delta).long()
    best = g.clone()
    found = torch.zeros(K, dtype=torch.bool, device=dev)
    bestd = torch.full((K,), -1.0, device=dev)
    base = torch.tensor([win, win, win], device=dev)
    for dx in range(-win, win + 1):
        for dy in range(-win, win + 1):
            for dz in range(-win, win + 1):
                q = g + torch.tensor([dx, dy, dz], device=dev)
                okb = ((q >= 0) & (q < R)).all(1)
                qc = q.clamp(0, R - 1)
                lin = (qc[:, 0] * R + qc[:, 1]) * R + qc[:, 2]
                good = okb & free_flat[lin] & (dist_flat[lin] >= d_safe)
                take = good & (~found)
                if bool(take.any()):
                    best[take] = qc[take]
                    bestd[take] = dist_flat[lin][take]
                    found |= good
    out = vox_lo[None, :] + (best.float() + 0.5) * Delta
    return out, found


def _positions_vote(P, w_v, n_out, occ, dist, vox_lo, Delta, R, d_target,
                    d_safe, n_pos, seed=0):
    """M5: surface normal back-projection voting (FC-Planner-lite).

    Returns (positions, vote direction field, n_landed_votes).
    """
    dev = P.device
    occ_flat = occ.reshape(-1)
    dist_flat = dist.reshape(-1)
    nvox = R * R * R
    Vv = torch.zeros(nvox, device=dev)
    Dv = torch.zeros(nvox, 3, device=dev)
    for a in (0.65, 1.0, 1.35):
        step = a * d_target * n_out
        free_ok = torch.ones(P.shape[0], dtype=torch.bool, device=dev)
        for t in (0.33, 0.66, 1.0):
            m = P + t * step
            gm = torch.floor((m - vox_lo[None, :]) / Delta).long()
            okm = ((gm >= 0) & (gm < R)).all(1)
            gmc = gm.clamp(0, R - 1)
            lin_m = (gmc[:, 0] * R + gmc[:, 1]) * R + gmc[:, 2]
            free_ok &= okm & (~occ_flat[lin_m])
        land = P + step
        gl = torch.floor((land - vox_lo[None, :]) / Delta).long()
        okl = ((gl >= 0) & (gl < R)).all(1)
        glc = gl.clamp(0, R - 1)
        lin_l = (glc[:, 0] * R + glc[:, 1]) * R + glc[:, 2]
        good = okl & free_ok & (~occ_flat[lin_l]) & (dist_flat[lin_l] >= d_safe)
        if bool(good.any()):
            idx = lin_l[good]
            ww = w_v[good]
            Vv.scatter_add_(0, idx, ww)
            Dv.scatter_add_(0, idx[:, None].expand(-1, 3),
                            (-n_out[good]) * ww[:, None])
    V3 = Vv.view(1, 1, R, R, R)
    Vs = F.avg_pool3d(V3, 3, stride=1, padding=1)[0, 0]
    Vn = F.max_pool3d(Vs[None, None], 3, stride=1, padding=1)[0, 0]
    peak = (Vs >= Vn - 1e-9) & (Vs > 0)
    cand = torch.nonzero(peak.reshape(-1), as_tuple=False).reshape(-1)
    if cand.numel() < 8:
        cand = torch.nonzero((Vv > 0).reshape(-1), as_tuple=False).reshape(-1)
    if cand.numel() < 8:
        cand = torch.nonzero(dist_flat.view(R, R, R).reshape(-1) >= d_safe,
                             as_tuple=False).reshape(-1)
    if cand.numel() == 0:
        return None, None, 0
    vv = Vs.reshape(-1)[cand]
    k = max(8, int(0.2 * cand.numel()))
    top = cand[torch.topk(vv, min(k, vv.numel())).indices]
    cpos = vox_lo[None, :] + (_unravel3(top, R).float() + 0.5) * Delta
    sel = _fps(cpos, n_pos, seed)
    dfield = Dv.reshape(R, R, R, 3)
    return cpos[sel], dfield, int(cand.numel())


def _positions_pca(free_b, dist, vox_lo, Delta, R, d_target, d_safe, n_pos,
                   eta0=1.6, n_corr=16, seed=0, surf_P=None):
    """M6: PCA of the safe free space -> compact rings or corridor chain.

    `surf_P` (surface voxel centres) should be passed when available: the *free*
    voxel set fills the padded AABB, so its covariance is isotropic (eta ~ 1.0)
    and the corridor branch would never trigger.  The surface set follows the
    scene's actual shape, which is what the eta discriminator is meant to read.
    """
    dev = free_b.device
    dist_flat = dist.reshape(-1)
    free_flat = free_b.reshape(-1)
    if surf_P is not None and surf_P.shape[0] >= 64:
        sub = surf_P[::max(1, surf_P.shape[0] // 20000)]
    else:
        ok = free_flat & (dist_flat >= d_safe)
        idx = torch.nonzero(ok, as_tuple=False).reshape(-1)
        if idx.numel() < 64:
            return None, float("nan"), "none"
        sub = idx[torch.linspace(0, idx.numel() - 1, min(20000, idx.numel())).long()]
        sub = vox_lo[None, :] + (_unravel3(sub, R).float() + 0.5) * Delta
    X = sub
    c = X.mean(0)
    Y = X - c
    C = (Y[:, :, None] * Y[:, None, :]).sum(0) / max(X.shape[0] - 1, 1)
    w, Vv = np.linalg.eigh(C.detach().cpu().numpy())
    order = np.argsort(w)[::-1]
    w = w[order]
    Vv = Vv[:, order]
    eta = float(math.sqrt(max(float(w[0]), 1e-12) / max(float(w[1]), 1e-12)))
    e1 = torch.as_tensor(Vv[:, 0], dtype=torch.float32, device=dev)
    e2 = torch.as_tensor(Vv[:, 1], dtype=torch.float32, device=dev)
    e3 = torch.as_tensor(Vv[:, 2], dtype=torch.float32, device=dev)
    if eta < eta0:                                    # compact rings
        pts = []
        for r in (0.55 * d_target, 1.0 * d_target):
            for k in range(max(8, n_pos // 2)):
                th = 2.0 * math.pi * k / max(8, n_pos // 2)
                pts.append(c + r * (math.cos(th) * e1 + math.sin(th) * e2))
        P0 = torch.stack(pts, 0)
        kind = "compact"
    else:                                             # corridor along e1
        h = ((X - c) * e1[None, :]).sum(1)            # elementwise, no BLAS
        lo = torch.quantile(h, 0.05)
        hi = torch.quantile(h, 0.95)
        pts = []
        for k in range(n_corr):
            t = lo + (hi - lo) * (k + 0.5) / n_corr
            for sgn in (-1.0, 1.0):
                pts.append(c + t * e1 + sgn * 0.35 * d_target * e2)
        P0 = torch.stack(pts, 0)
        kind = "corridor"
    P1, found = _snap_free(P0, vox_lo, Delta, R, free_flat, dist_flat, d_safe)
    if int(found.sum()) >= 8:
        P1 = P1[found]
    sel = _fps(P1, n_pos, seed)
    P1 = P1[sel]
    return P1, eta, kind


def _gauss_cov(mu, smax, w2v, fx, fy, W, H, Wz, Hz, dev, batch=32, delta=1.0,
               gate=None):
    """(M,K) visibility of probe gaussians: conservative footprint + coarse z-buffer.

    `gate=(rpx_lo, rpx_hi, z_lo, z_hi)` additionally applies the *same* scale /
    depth window that the downstream O2cal kernel uses, so the greedy objective
    matches what the fit will actually accept.
    """
    M = int(w2v.shape[0])
    K = int(mu.shape[0])
    out = torch.zeros((M, K), dtype=torch.bool, device=dev)
    if K == 0:
        return out
    Ph = torch.cat([mu, torch.ones(K, 1, device=dev)], 1)
    fxl = fx * (Wz / W)
    fyl = fy * (Hz / H)
    big = 1e6
    for s0 in range(0, M, batch):
        s1 = min(s0 + batch, M)
        Wm = w2v[s0:s1]
        pc = (Ph[None, :, None, :] * Wm[:, None, :, :]).sum(-1)
        z = pc[..., 2]
        zc = z.clamp_min(1e-6)
        uu = fxl * pc[..., 0] / zc + 0.5 * Wz
        vv = fyl * pc[..., 1] / zc + 0.5 * Hz
        ui = torch.floor(uu).long()
        vi = torch.floor(vv).long()
        ok = (z > 1e-6) & (ui >= 0) & (ui < Wz) & (vi >= 0) & (vi < Hz)
        r = (smax[None, :] * fx / zc) * (Wz / W)      # conservative pixel radius
        ok = ok & (uu + r > 0) & (uu - r < Wz) & (vv + r > 0) & (vv - r < Hz)
        if gate is not None:
            rl, rh, zl, zh = gate
            rpx = smax[None, :] * fx / zc            # full-res pixel radius
            ok = ok & (rpx >= rl) & (rpx <= rh) & (z >= zl) & (z <= zh)
        pix = vi.clamp(0, Hz - 1) * Wz + ui.clamp(0, Wz - 1)
        zf = torch.where(ok, z, torch.full_like(z, big))
        zb = torch.full((s1 - s0, Hz * Wz), big, device=dev)
        zb.scatter_reduce_(1, pix, zf, reduce="amin", include_self=True)
        zb = -F.max_pool2d(-zb.view(s1 - s0, 1, Hz, Wz), 3, 1, 1)[:, 0]
        Zt = zb.reshape(s1 - s0, Hz * Wz).gather(1, pix)
        ok = ok & (z <= Zt + 2.0 * delta)
        out[s0:s1] = ok
    return out


def _gauss_quality(mu, smid, op, w2v, fx, fy, W, H, dev, rstar, rsigma,
                   rmin=0.5, rmax=40.0, batch=32):
    """T1-B: smooth projected-radius quality q_ck (plan Track 1, eq. for q_{c,k}).

    q_ck = alpha_k * exp(-(ln(r_ck/r*))^2 / (2 sigma^2)) * 1[r_ck in [rmin,rmax]]
    with r_ck = fx * s_mid_k / z_ck the full-resolution pixel radius.
    """
    M = int(w2v.shape[0])
    K = int(mu.shape[0])
    out = torch.zeros((M, K), dtype=torch.float32, device=dev)
    if K == 0:
        return out
    Ph = torch.cat([mu, torch.ones(K, 1, device=dev)], 1)
    inv2s2 = 1.0 / (2.0 * float(rsigma) ** 2)
    for s0 in range(0, M, batch):
        s1 = min(s0 + batch, M)
        pc = (Ph[None, :, None, :] * w2v[s0:s1, None, :, :]).sum(-1)
        z = pc[..., 2].clamp_min(1e-6)
        r = (smid[None, :] * fx / z)
        lr = (r.clamp_min(1e-9) / float(rstar)).log()
        q = op[None, :] * torch.exp(-0.5 * lr * lr / inv2s2)
        q = torch.where((r >= rmin) & (r <= rmax) & (pc[..., 2] > 1e-6),
                        q, torch.zeros_like(q))
        out[s0:s1] = q
    return out


def _positions_multishell(P, nocc, free_b, dist, vox_lo, Delta, R, n_pos,
                          ratios=(0.40, 0.35, 0.25), seed=0):
    """T1-A: three concentric shells around the dense foreground core.

    Band radii are multiples of R_core (the median distance of the densest 20%
    of surface voxels from the density focus), so the same rule adapts to a
    truck, a street or a train corridor without any per-scene switch.
    """
    ok = free_b.reshape(-1) & (dist.reshape(-1) >= 1.0)
    idx = torch.nonzero(ok, as_tuple=False).reshape(-1)
    if idx.numel() < 64:
        return None
    X = vox_lo[None, :] + (_unravel3(idx, R).float() + 0.5) * Delta
    nv = nocc.detach().cpu().numpy() if nocc is not None else None
    Pn = P.detach().cpu().numpy()
    if nv is None:
        nv = np.ones(Pn.shape[0])
    k5 = max(8, int(0.05 * Pn.shape[0]))
    top5 = np.argsort(-nv)[:k5]
    c = (Pn[top5] * nv[top5, None]).sum(0) / max(nv[top5].sum(), 1e-9)
    k20 = max(16, int(0.20 * Pn.shape[0]))
    top20 = np.argsort(-nv)[:k20]
    Rc = float(np.median(np.linalg.norm(Pn[top20] - c[None, :], axis=1)))
    Rc = max(Rc, 1e-3)
    r = (X - torch.as_tensor(c, dtype=X.dtype, device=X.device)).norm(dim=1)
    bands = [(0.8, 1.6), (1.6, 3.2), (3.2, 1e9)]
    parts = []
    for (lo, hi), fr in zip(bands, ratios):
        sel = X[(r >= lo * Rc) & (r < hi * Rc)]
        q = max(4, int(round(fr * n_pos)))
        if sel.shape[0] >= q:
            parts.append(sel[_fps(sel, q, seed)])
    if not parts:
        return None
    return torch.cat(parts), Rc


def _positions_rings(P, nocc, dev, n_pos=48, radii=(1.0, 2.0, 3.5),
                     n_az=16, n_h=2, seed=0, clr=None):
    """T5-A: voxel-free analytic multi-ring rig around the dense core.

    No grid is consulted: the rings are laid out around the point-cloud core
    centre and only rejected when they fall inside the point cloud itself.
    """
    Pn = P.detach().cpu().numpy()
    nv = nocc.detach().cpu().numpy() if nocc is not None else np.ones(Pn.shape[0])
    k5 = max(8, int(0.05 * Pn.shape[0]))
    top5 = np.argsort(-nv)[:k5]
    c = (Pn[top5] * nv[top5, None]).sum(0) / max(nv[top5].sum(), 1e-9)
    k20 = max(16, int(0.20 * Pn.shape[0]))
    top20 = np.argsort(-nv)[:k20]
    Rc = float(np.median(np.linalg.norm(Pn[top20] - c[None, :], axis=1)))
    Rc = max(Rc, 1e-3)
    # gravity axis + two in-plane basis vectors
    u = torch.tensor([0.0, 1.0, 0.0], device=dev)
    e1 = torch.tensor([1.0, 0.0, 0.0], device=dev)
    e2 = torch.cross(u, e1)
    pts = []
    for rr in radii:
        rad = rr * Rc
        for ih in range(n_h):
            zoff = (ih - 0.5) * 0.6 * Rc
            for a in range(n_az):
                th = 2.0 * math.pi * a / n_az
                p = (torch.as_tensor(c, device=dev, dtype=torch.float32)
                     + rad * (math.cos(th) * e1 + math.sin(th) * e2)
                     + zoff * u)
                pts.append(p)
    X = torch.stack(pts)
    # reject poses buried inside the point cloud (nearest-neighbour clearance);
    # the threshold must scale with the *innermost* ring, otherwise a compact
    # inner ring (T5-B) is rejected entirely.
    sub = P[:: max(1, P.shape[0] // 20000)]
    clr = float(clr) if clr else 0.5 * float(min(radii)) * Rc
    keep = []
    for i in range(0, X.shape[0], 64):
        chunk = X[i:i + 64]
        d = (chunk[:, None, :] - sub[None, :, :]).norm(dim=2).min(1).values
        keep.append(d > clr)
    keep = torch.cat(keep)
    X = X[keep]
    if X.shape[0] < 8:
        return None
    if X.shape[0] > n_pos:
        X = X[_fps(X, n_pos, seed)]
    return X


def _fps(pos, k, seed=0):
    """Greedy farthest-point sampling of k indices from (N,3) positions."""
    n = pos.shape[0]
    if n <= k:
        return torch.arange(n, device=pos.device)
    d = torch.full((n,), float("inf"), dtype=torch.float32, device=pos.device)
    idx = [int(torch.argmin(pos[:, 0] + pos[:, 2]))]          # deterministic start
    d = torch.minimum(d, (pos - pos[idx[0]]).pow(2).sum(1))
    for _ in range(k - 1):
        i = int(torch.argmax(d))
        idx.append(i)
        d = torch.minimum(d, (pos - pos[i]).pow(2).sum(1))
    return torch.as_tensor(idx, device=pos.device)


# --------------------------------------------------------------------------
def extract_auto_virtual_cams(
    xyz: torch.Tensor,          # (N,3) float32 CUDA, gaussian centres
    scales: torch.Tensor,       # (N,3) float32 CUDA, LINEAR scales
    quats: torch.Tensor,        # (N,4) float32 CUDA, normalised w,x,y,z
    opacities: torch.Tensor,    # (N,)  float32 CUDA, post-sigmoid
    n_views: int = 8,
    grid_res: int = 64,
    n_pos: int = 48,
    n_dirs: int = 6,
    fov_deg: float = 60.0,
    W: int = 1024,
    H: int = 768,
    cov_k: int = 3,
    use_zbuf: bool = True,
    use_scale_gate: bool = True,
    use_normal_gate: bool = False,
    # step-1 knobs
    alpha_min: float = 0.1,
    n_sub: int = 100_000,
    p_low: float = 0.02,
    p_high: float = 0.98,
    pad_frac: float = 0.25,
    r_target: float = 4.0,
    d_lo_frac: float = 0.20,
    d_hi_frac: float = 0.65,
    cavity_min6: int = 20,
    # step-2/3 knobs
    d_safe: int = 2,
    d_far: int = 10,
    n_erode: int = 12,
    tol_gamma: float = 2.0,
    r_px_min: float = 0.8,
    r_px_max: float = 35.0,
    Wz: int = 64,
    Hz: int = 48,
    batch_cams: int = 32,
    cavity_min: int = 500,
    cavity_frac: float = 0.75,
    surf_cap: int = 20_000,
    dedupe_deg: float = 35.0,
    near_frac: float = 0.10,
    near_deg: float = 25.0,
    seed: int = 0,
    noblas: bool = False,
    return_pool: bool = False,   # return ALL candidate cameras (PICS pool) instead
                                 # of the greedy-selected B views
    # ---- v2 modules (defaults reproduce v1 exactly) ----------------------
    core_aabb: float = 0.0,      # M1: keep the p_keep smallest in-plane-scale
                                 #     points; 0 = off (v1 2%/98% AABB)
    dmode: str = "sgeom",        # M2a: "sgeom" (v1) | "smid" (in-plane median)
    r_ref: float | None = None,  # M2a target projected radius (px)
    soft_scale: float = 0.0,     # M2b: sigma_r of the log-Gaussian scale kernel
                                 #      (>0 disables the hard scale gate)
    dir_mode: str = "axis",      # M3: "axis" (v1) | "gravity"
    pitches: str = "0",          # M3: comma-separated pitch degrees (e.g. "0,-15")
    grav_lower: float = 0.0,     # B2: re-vote the gravity axis on the lower shell
    cos_weight: float = 0.0,     # M4: gamma of the front-facing cosine coverage
                                 #      (>0 replaces the phi k-coverage objective)
    cmax: float = 2.0,           # M4: saturating coverage cap C_max
    pos_mode: str = "frontier",  # M5/M6: "frontier" (v1) | "vote" | "pca_traj"
    eta0: float = 1.6,           # M6: compact vs corridor threshold on sqrt(l1/l2)
    n_corr: int = 16,            # M6: number of corridor cross-sections
    dir_bins: int = 0,           # M7a: >0 enables the azimuth quadrant coverage
    aniso_weight: float = 0.0,   # M7b: beta of the anisotropy sensitivity weight
    weight_mode: str = "log",    # D3: "log" (v1 log1p(W_occ)) | "n" (N_v^0.5) | "mip"
    gate_rel: float = 0.0,       # audit fix: gate window relative to the actual
                                 # camera-to-scene distance (0 = use d_target)
    cover_target: str = "voxel",  # "voxel" (v1 surface-voxel coverage) | "gauss"
    n_gauss_probe: int = 50000,   # probe size for the gaussian-support objective
    cover_gate: bool = False,     # apply the kernel's r_px / z window in the objective
    gauss_rstar: float = 0.0,     # T1-B: >0 enables the r* soft quality objective
    gauss_rsigma: float = 1.0,    # T1-B: log-radius tolerance
    gauss_agg: str = "max",       # T1-B: "max" (facility location) | "cap"
    gauss_cap: float = 3.0,       # T1-B: saturation cap for agg="cap"
    phi_override: str = "",       # T4-A: explicit phi curve, e.g. "0,1,1.15,1.2"
    normal_mode: str = "kstep",   # T2-A: "kstep" (N0, v1) | "n2" (tensor + global)
    shell_ratios: str = "0.40,0.35,0.25",   # T1-A: inner/mid/outer slot fractions
    ring_radii: str = "1.0,2.0,3.5",        # T5-A: voxel-free ring radii (x R_core)
    diverse_quota: bool = False,            # T4-B: per-band quota on the 8 picks
    extra_cams: list | None = None,         # S2: extra candidates from Domain B
    unified_frac: float = 0.5,              # S3: interior-pocket share for "unified"
    pca_corridor_only: bool = False,  # M6c: keep the v1 frontier for compact scenes
    return_stats: bool = False,
    verbose: bool = False,
):
    """Returns (cams, stats) -- cams in `cams_fast` dict format."""
    dev = xyz.device
    t0 = torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else None
    stamps = {}

    def _mark(name):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        stamps[name] = time.perf_counter()

    with torch.no_grad():
        _mark("start")
        # ------------------------------------------------------------------
        # Step 1: core AABB + projection-scale self calibration
        # ------------------------------------------------------------------
        op = opacities.reshape(-1)
        keep = op > alpha_min
        if int(keep.sum()) < 1000:
            keep = torch.ones_like(keep, dtype=torch.bool)
        mu = xyz[keep]
        sc = scales[keep]
        qq = quats[keep]
        ov = op[keep]
        Nv = mu.shape[0]

        step = max(1, int(Nv // n_sub))
        sub = mu[::step]
        sub_sc = sc[::step]
        s_sorted = torch.sort(sub_sc, dim=1).values
        s_mid_sub = s_sorted[:, 1]                    # M2: in-plane median scale
        s_geom_sub = sub_sc.prod(1).clamp_min(1e-30).pow(1.0 / 3.0)

        fg_keep = 1.0
        fg_mask = None
        if 0.0 < core_aabb < 1.0:                     # M1: foreground-core AABB
            thr = torch.quantile(s_mid_sub, float(core_aabb))
            fg_mask = s_mid_sub <= thr
            if int(fg_mask.sum()) < 100:
                fg_mask = None
            else:
                fg_keep = float(fg_mask.float().mean().item())
        if fg_mask is None:
            lo, hi = _quantile_axis(sub, [p_low, p_high])
        else:
            fg = sub[fg_mask]
            ctr0 = fg.median(0).values
            rr = (fg - ctr0).norm(dim=1)
            r90 = torch.quantile(rr, 0.90)
            pts2 = fg[rr <= r90]
            if pts2.shape[0] < 100:
                pts2 = fg
            lo, hi = _quantile_axis(pts2, [0.03, 0.97])
        core_lo, core_hi = lo, hi
        L = (core_hi - core_lo).clamp_min(1e-6)
        Lmax = float(L.max().item())
        pad = pad_frac * Lmax
        vox_lo = core_lo - pad
        vox_hi = core_hi + pad
        Lvox = (vox_hi - vox_lo).clamp_min(1e-6)
        R = int(grid_res)
        Delta = float(Lvox.max().item()) / R

        in_core = ((sub >= core_lo) & (sub <= core_hi)).all(1)
        if int(in_core.sum()) < 100:
            in_core = torch.ones_like(in_core)
        s_use_sub = s_mid_sub if dmode == "smid" else s_geom_sub
        s_med = float(torch.median(s_use_sub[in_core]).item())
        fx = _intr(fov_deg, W)
        fy = fx
        rref = float(r_target if r_ref is None else r_ref)
        if dmode == "smid" and r_ref is None:
            rref = 1.5
        d_target = min(max(fx * s_med / rref, d_lo_frac * Lmax), d_hi_frac * Lmax)
        d_min = 0.25 * d_target
        d_max = 3.5 * d_target
        _mark("step1")

        # ------------------------------------------------------------------
        # Step 2: coarse occupancy, surface voxels, safe-distance field
        # ------------------------------------------------------------------
        u = torch.floor((mu - vox_lo[None, :]) / Delta).long()
        inside = ((u >= 0) & (u < R)).all(1)
        u = u[inside]
        lin = (u[:, 0] * R + u[:, 1]) * R + u[:, 2]
        nvox = R * R * R
        w_p = ov[inside]
        mu_p = mu[inside]
        sc_p = sc[inside]
        s_g = sc_p.prod(1).clamp_min(1e-30).pow(1.0 / 3.0)
        Rm = _quat_to_mat(qq[inside])
        axis_min = sc_p.argmin(1)
        nrm = Rm.gather(2, axis_min[:, None, None].expand(-1, 3, 1)).squeeze(2)

        Wocc = torch.zeros(nvox, device=dev)
        Wocc.scatter_add_(0, lin, w_p)
        cen_acc = torch.zeros(nvox, 3, device=dev)
        cen_acc.scatter_add_(0, lin[:, None].expand(-1, 3), mu_p * w_p[:, None])
        sca_acc = torch.zeros(nvox, device=dev)
        sca_acc.scatter_add_(0, lin, s_g * w_p)
        nrm_acc = torch.zeros(nvox, 3, device=dev)
        nrm_acc.scatter_add_(0, lin[:, None].expand(-1, 3), nrm * w_p[:, None])
        Nocc = torch.zeros(nvox, device=dev)
        Nocc.scatter_add_(0, lin, torch.ones_like(w_p))
        T_acc = None
        if normal_mode == "n2":
            # T2-A: nematic structure tensor over *flat* gaussians only
            _ss = torch.sort(sc_p, dim=1).values
            _flat = (_ss[:, 0] / _ss[:, 1].clamp_min(1e-9)) < 0.3
            _wf = w_p * _flat.float()
            _c = torch.stack([nrm[:, 0] * nrm[:, 0], nrm[:, 0] * nrm[:, 1],
                              nrm[:, 0] * nrm[:, 2], nrm[:, 1] * nrm[:, 1],
                              nrm[:, 1] * nrm[:, 2], nrm[:, 2] * nrm[:, 2]], 1)
            T_acc = torch.zeros(nvox, 6, device=dev)
            T_acc.scatter_add_(0, lin[:, None].expand(-1, 6), _c * _wf[:, None])
        if aniso_weight > 0.0:                        # M7b: ln(s_max / s_min)
            _kap = (sc_p.max(1).values
                    / sc_p.clamp_min(1e-6).min(1).values).clamp_min(1.0).log()
            kap_acc = torch.zeros(nvox, device=dev)
            kap_acc.scatter_add_(0, lin, _kap * w_p)
        else:
            kap_acc = None
        if weight_mode != "log":                      # D3: density-weight probes
            s_mid_p = torch.sort(sc_p, dim=1).values[:, 1]
            cnt_acc = torch.zeros(nvox, device=dev)
            cnt_acc.scatter_add_(0, lin, torch.ones_like(w_p))
            rho_acc = torch.zeros(nvox, device=dev)
            rho_acc.scatter_add_(0, lin, w_p / s_mid_p.clamp_min(1e-6) ** 2)
        else:
            cnt_acc = rho_acc = None

        Wocc3 = Wocc.view(R, R, R)
        nz = Wocc3[Wocc3 > 0]
        tau_occ = float(min(torch.quantile(nz, 0.20).item(), 0.5)) if nz.numel() else 0.5
        occ = (Wocc3 >= max(tau_occ, 1e-6))
        occf = occ.float()

        free = 1.0 - occf
        nb_free = F.max_pool3d(free[None, None], 3, stride=1, padding=1)[0, 0] > 0.5
        surf = occ & nb_free
        surf_idx = torch.nonzero(surf.reshape(-1), as_tuple=False).reshape(-1)
        n_surf = int(surf_idx.numel())

        # erosion-based discrete distance-to-surface on free voxels
        cur = free.clone()
        dist = torch.zeros_like(cur)
        for _ in range(n_erode):
            cur = 1.0 - F.max_pool3d(1.0 - cur[None, None], 3, stride=1,
                                     padding=1)[0, 0]
            dist += cur
        _mark("step2_vox")

        # interior cavity vs outer shell (step 2.4)
        box_lo = torch.floor((core_lo - vox_lo) / Delta).long().clamp(0, R - 1)
        box_hi = torch.ceil((core_hi - vox_lo) / Delta).long().clamp(0, R - 1)
        gx = torch.arange(R, device=dev)
        inx = (gx >= box_lo[0]) & (gx <= box_hi[0])
        iny = (gx >= box_lo[1]) & (gx <= box_hi[1])
        inz = (gx >= box_lo[2]) & (gx <= box_hi[2])
        in_box = (inx[:, None, None] & iny[None, :, None] & inz[None, None, :])

        hits = torch.zeros_like(occf)
        for d in range(3):
            T = occf.sum(dim=d, keepdim=True)
            C = torch.cumsum(occf, dim=d)
            hits += ((T - C) > 0.5).float()          # occupied strictly ahead
            hits += ((C - occf) > 0.5).float()       # occupied strictly behind

        free_b = free > 0.5
        cavity = free_b & in_box & (hits >= 4) & (dist >= d_safe)
        n_cav = int(cavity.sum().item())
        # fully enclosed (all six rays blocked) is a much stricter "is a room"
        # signal than "four of six": an outdoor ground plane already gives 5.
        cav6 = free_b & in_box & (hits >= 6) & (dist >= d_safe)
        n_cav6 = int(cav6.sum().item())

        def _unravel(idx):
            """flat voxel index -> (n,3) integer ijk, row-major over (R,R,R)."""
            i = torch.div(idx, R * R, rounding_mode="floor")
            j = torch.div(idx - i * (R * R), R, rounding_mode="floor")
            k = idx - i * (R * R) - j * R
            return torch.stack([i, j, k], 1)

        def _pos_from_idx(idx):
            return vox_lo[None, :] + (_unravel(idx).float() + 0.5) * Delta

        # ---- surface-voxel payload (shared by M5/M6 positions and Step 3) --
        s_ijk = _unravel(surf_idx)
        P = vox_lo[None, :] + (s_ijk.float() + 0.5) * Delta
        if n_surf > surf_cap:
            _wv = Wocc[surf_idx]
            _order = torch.argsort(_wv)
            _pick = _order[torch.linspace(0, _order.numel() - 1, surf_cap).long()]
            surf_idx = surf_idx[_pick]
            P = P[_pick]
        V = P.shape[0]
        s_ijk = _unravel(surf_idx)
        w_v = torch.log1p(Wocc[surf_idx])
        if weight_mode == "n":
            w_v = cnt_acc[surf_idx].clamp_min(1e-9).sqrt()
        elif weight_mode == "mip":
            w_v = rho_acc[surf_idx].clamp_min(1e-12).pow(0.35)
        s_v = sca_acc[surf_idx] / Wocc[surf_idx].clamp_min(1e-6)
        n_v = nrm_acc[surf_idx]
        n_v = n_v / n_v.norm(dim=1, keepdim=True).clamp_min(1e-9)
        if normal_mode == "n2":
            # T2-A: N2 = nematic structure-tensor axis + global orientation
            T_s = T_acc[surf_idx].detach().cpu().numpy().astype(np.float64)
            Mx = np.empty((V, 3, 3), np.float64)
            Mx[:, 0, 0] = T_s[:, 0]
            Mx[:, 0, 1] = Mx[:, 1, 0] = T_s[:, 1]
            Mx[:, 0, 2] = Mx[:, 2, 0] = T_s[:, 2]
            Mx[:, 1, 1] = T_s[:, 3]
            Mx[:, 1, 2] = Mx[:, 2, 1] = T_s[:, 4]
            Mx[:, 2, 2] = T_s[:, 5]
            _w, _v = np.linalg.eigh(Mx)
            ax = _v[:, :, 2]
            ax = ax / np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
            Pn = P.detach().cpu().numpy()
            _nv = Nocc[surf_idx].detach().cpu().numpy()
            _k = max(1, int(0.05 * _nv.shape[0]))
            _top = np.argsort(-_nv)[:_k]
            focus = (Pn[_top] * _nv[_top, None]).sum(0) / max(_nv[_top].sum(), 1e-9)
            ctr = Pn.mean(0)
            # indoor reference = the surface centroid (≈ room centre); outdoor =
            # pointing away from the dense focus.  n_cav cleanly separates the two
            # regimes on these scenes (indoor >= 1e4, outdoor <= a few hundred).
            ref = (Pn - focus[None, :]) if (n_cav < 2000) else (ctr[None, :] - Pn)
            sgn = np.sign((ax * ref).sum(1))
            sgn[sgn == 0] = 1.0
            n_out = torch.as_tensor(ax * sgn[:, None], dtype=torch.float32, device=dev)
        else:
            n_out = (n_v if (cos_weight <= 0.0 and pos_mode != "vote")
                     else _orient_normals(n_v, s_ijk, occ, R))
        if aniso_weight > 0.0:                    # M7b: anisotropy sensitivity
            w_v = w_v * (1.0 + float(aniso_weight)
                         * (kap_acc[surf_idx] / Wocc[surf_idx].clamp_min(1e-6))
                         .clamp(0.0, 3.0))

        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        def _sub(t, cap):
            if t.numel() <= cap:
                return t
            return t[torch.randperm(t.numel(), generator=gen, device=dev)[:cap].sort().values]

        cap = max(2000, 60 * n_pos)
        indoor = n_cav6 >= max(cavity_min6, int(0.02 * n_cav))
        if indoor:
            mode = "indoor"
            outer = free_b & (~in_box) & (dist >= d_safe) & (dist <= d_far)
            ci = _sub(torch.nonzero(cavity.reshape(-1), as_tuple=False).reshape(-1), cap)
            oi = _sub(torch.nonzero(outer.reshape(-1), as_tuple=False).reshape(-1), cap)
            parts = []
            if ci.numel():
                pc = _pos_from_idx(ci)
                nc = max(1, int(round(n_pos * cavity_frac)))
                parts.append(pc[_fps(pc, min(nc, pc.shape[0]), seed)])
            if oi.numel():
                po = _pos_from_idx(oi)
                no = max(1, n_pos - sum(int(x.shape[0]) for x in parts))
                parts.append(po[_fps(po, min(no, po.shape[0]), seed)])
            pos = torch.cat(parts)
            cand = torch.cat([ci, oi])
        else:
            mode = "outdoor"
            band = free_b & (dist >= d_safe) & (dist <= d_far)
            cand = _sub(torch.nonzero(band.reshape(-1), as_tuple=False).reshape(-1), cap)
            if cand.numel() >= 8:
                pc = _pos_from_idx(cand)
                pos = pc[_fps(pc, min(n_pos, pc.shape[0]), seed)]
            else:
                pos = None

        # ---- M5 / M6 alternative position generators ----------------------
        vote_field = None
        pca_eta = float("nan")
        pca_kind = ""
        if pos_mode == "vote":
            pv, vote_field, _nvote = _positions_vote(
                P, w_v, n_out, occ, dist, vox_lo, Delta, R, d_target,
                d_safe, n_pos, seed)
            if pv is not None and pv.shape[0] >= 8:
                pos = pv
                mode = mode + "+vote"
            cand = torch.zeros(0, dtype=torch.long, device=dev)
        elif pos_mode == "multishell":
            _pm = _positions_multishell(
                P, Nocc[surf_idx], free_b, dist, vox_lo, Delta, R, n_pos,
                ratios=([float(x) for x in str(shell_ratios).split(",")]
                        if shell_ratios else (0.40, 0.35, 0.25)),
                seed=seed)
            if _pm is not None:
                pos, _rc = _pm
                if pos.shape[0] >= 8:
                    mode = mode + "+mshell"
            cand = torch.zeros(0, dtype=torch.long, device=dev)
        elif pos_mode == "mixed":
            # T3-A: heterogeneous pool = outdoor frontier band + core shells +
            # point-cloud rings.  This is also the answer to *mixed* scenes: the
            # regime decision moves from the scene level down to the pose level.
            _parts = []
            band = free_b & (dist >= d_safe) & (dist <= d_far)
            _cb = _sub(torch.nonzero(band.reshape(-1), as_tuple=False).reshape(-1), cap)
            if _cb.numel() >= 8:
                _parts.append(_pos_from_idx(_cb))
            _ms = _positions_multishell(P, Nocc[surf_idx], free_b, dist,
                                        vox_lo, Delta, R, n_pos, seed=seed)
            if _ms is not None:
                _parts.append(_ms[0])
            _rg = _positions_rings(P, Nocc[surf_idx], dev, n_pos=n_pos, seed=seed)
            if _rg is not None:
                _parts.append(_rg)
            if _parts:
                _all = torch.cat(_parts)
                pos = _all[_fps(_all, min(n_pos, _all.shape[0]), seed)]
                mode = mode + "+mixed"
            cand = torch.zeros(0, dtype=torch.long, device=dev)
        elif pos_mode == "unified":
            # S3: delete the `cav6 >= threshold` binary branch.  Every scene gets
            # the same mixture: interior free-space pockets (high 6-axis ray
            # hit-count, safe distance) + the exterior frontier shell, at a fixed
            # ratio; a shortfall in one class is filled by the other.
            outer = free_b & (~in_box) & (dist >= d_safe) & (dist <= d_far)
            band = free_b & (dist >= d_safe) & (dist <= d_far)
            ci = _sub(torch.nonzero(cavity.reshape(-1), as_tuple=False).reshape(-1), cap)
            bi = _sub(torch.nonzero(band.reshape(-1), as_tuple=False).reshape(-1), cap)
            _oi = _sub(torch.nonzero(outer.reshape(-1), as_tuple=False).reshape(-1), cap)
            n_in = max(1, int(round(n_pos * float(unified_frac))))
            parts = []
            if ci.numel() >= 8:
                pp = _pos_from_idx(ci)
                parts.append(pp[_fps(pp, min(n_in, pp.shape[0]), seed)])
            got = sum(int(x.shape[0]) for x in parts)
            for idx, need in ((_oi, n_in - got), (bi, n_pos - got)):
                if need <= 0 or idx.numel() < 8:
                    continue
                pp = _pos_from_idx(idx)
                parts.append(pp[_fps(pp, min(need, pp.shape[0]), seed)])
                got = sum(int(x.shape[0]) for x in parts)
            if parts:
                pos = torch.cat(parts)
                mode = mode + "+unified"
            cand = torch.cat([ci, bi]) if (ci.numel() and bi.numel()) else cand
        elif pos_mode == "rings":
            _pr = _positions_rings(
                P, Nocc[surf_idx], dev, n_pos=n_pos,
                radii=([float(x) for x in str(ring_radii).split(",")]
                       if ring_radii else (1.0, 2.0, 3.5)),
                seed=seed)
            if _pr is not None and _pr.shape[0] >= 8:
                pos = _pr
                mode = mode + "+rings"
            cand = torch.zeros(0, dtype=torch.long, device=dev)
        elif pos_mode == "pca_traj":
            pp, pca_eta, pca_kind = _positions_pca(
                free_b, dist, vox_lo, Delta, R, d_target, d_safe, n_pos,
                eta0=eta0, n_corr=n_corr, seed=seed, surf_P=P)
            if pp is not None and pp.shape[0] >= 8:
                if pca_corridor_only and pca_kind != "corridor":
                    mode = mode + "+pca_skip_compact"
                else:
                    pos = pp
                    mode = mode + "+" + str(pca_kind)
            cand = torch.zeros(0, dtype=torch.long, device=dev)

        # fallback ladder (spec 5.2)
        if pos is None:
            for ds in (1, 0):
                band = free_b & (dist >= ds) & (dist <= d_far)
                cand = _sub(torch.nonzero(band.reshape(-1), as_tuple=False).reshape(-1), cap)
                if cand.numel() >= 8:
                    pc = _pos_from_idx(cand)
                    pos = pc[_fps(pc, min(n_pos, pc.shape[0]), seed)]
                    break
        fallback = False
        if pos is None:
            fallback = True
            mode = "fallback_sphere"
            ctr0 = 0.5 * (core_lo + core_hi)
            rad = 0.65 * Lmax
            kk = torch.arange(n_pos, device=dev, dtype=torch.float32)
            az = 2.0 * math.pi * kk / n_pos
            el = torch.where(kk % 2 > 0, 0.2094395, -0.2094395)  # +-12 deg
            pos = torch.stack([rad * torch.cos(el) * torch.cos(az),
                               rad * torch.sin(el) * 0.6,
                               rad * torch.cos(el) * torch.sin(az)], 1) + ctr0
        n_pos_eff = int(pos.shape[0])

        # ------------------------------------------------------------------
        # Step 3: candidate poses + coarse soft z-buffer visibility
        # ------------------------------------------------------------------
        dirs = _dir_set(n_dirs) if dir_mode == "axis" else None
        u_g = None
        per_pos = None
        if dir_mode == "gravity":            # M3: zero-roll yaw ring around gravity
            u_g, _gv, _gfrac = _gravity_axis(mu, sc, qq, lower_frac=grav_lower)
            pitches = [float(x) for x in str(pitches).split(",") if str(x).strip()]
            dirs = _dirs_gravity(n_dirs, pitches, u_g.detach().cpu().numpy())
        elif dir_mode == "vote":             # M5b: aim along the voted view direction
            per_pos = []
            cen = pos.mean(0)
            for k in range(n_pos_eff):
                g = torch.floor((pos[k] - vox_lo) / Delta).long().clamp(0, R - 1)
                d0 = vote_field[g[0], g[1], g[2]] if vote_field is not None else None
                if d0 is None or float(d0.norm()) < 1e-6:
                    d0 = (cen - pos[k])
                d0 = d0 / d0.norm().clamp_min(1e-9)
                d0 = d0.detach().cpu().numpy().astype(np.float64)
                up = np.array([0.0, -1.0, 0.0])
                ax = up - d0 * float(d0 @ up)
                ax = ax / max(np.linalg.norm(ax), 1e-9)
                bi = np.cross(ax, d0)
                trio = []
                for ang in (0.0, 0.6109, -0.6109):     # +-35 deg yaw
                    trio.append(math.cos(ang) * d0 + math.sin(ang) * bi)
                per_pos.append(np.stack(trio))
        eyes, mdirs, names = [], [], []
        for k in range(n_pos_eff):
            e = pos[k].detach().cpu().numpy().astype(np.float64)
            dl = per_pos[k] if per_pos is not None else dirs
            for j, d in enumerate(dl):
                eyes.append(e)
                mdirs.append(d)
                names.append(f"virt_av_{k:02d}_{j:02d}")
        if extra_cams:
            for c in extra_cams:
                eyes.append(np.asarray(c["C"], np.float64))
                mdirs.append(np.asarray(c["w2v"], np.float64)[2, :3])
                names.append(str(c["name"]))
            M = len(eyes)
        M = len(eyes)
        if u_g is not None:
            up_w = -u_g.detach().cpu().numpy()
            w2v_all = np.stack([_look_at_up(e, e + d, up_w)
                                for e, d in zip(eyes[:M - len(extra_cams or [])],
                                                mdirs[:M - len(extra_cams or [])])]
                               + [np.asarray(c["w2v"], np.float64)
                                  for c in (extra_cams or [])])
        else:
            w2v_all = np.stack([look_at(e, e + d)
                                for e, d in zip(eyes[:M - len(extra_cams or [])],
                                                mdirs[:M - len(extra_cams or [])])]
                               + [np.asarray(c["w2v"], np.float64)
                                  for c in (extra_cams or [])])
        n_domain_a = M - len(extra_cams or [])

        # surface voxel payload is computed right after Step 2 (shared with M5/M6)
        fx_low = fx * (Wz / W)
        fy_low = fy * (Hz / H)
        ctr_scene = 0.5 * (core_lo + core_hi)
        soft = float(soft_scale) > 0.0
        use_cos = float(cos_weight) > 0.0
        if soft:                       # M2b: the continuous kernel replaces the gate
            use_scale_gate = False
        A = torch.zeros((M, V), dtype=torch.bool, device=dev)
        Aw = (torch.zeros((M, V), dtype=torch.float32, device=dev)
              if (soft or use_cos) else None)
        BinArr = (torch.zeros((M, V), dtype=torch.int8, device=dev)
                  if dir_bins > 0 else None)
        if dir_bins > 0:                       # M7a: horizontal azimuth quadrant
            if u_g is not None:
                _up = u_g.detach()
            else:
                _up = torch.tensor([0.0, 1.0, 0.0], device=dev)
            _tmp = torch.tensor([1.0, 0.0, 0.0], device=dev) \
                if abs(float(_up[0])) < 0.9 else torch.tensor([0.0, 0.0, 1.0], device=dev)
            e1h = torch.cross(_tmp, _up)
            e1h = e1h / e1h.norm().clamp_min(1e-9)
            e2h = torch.cross(_up, e1h)
        n_vis = torch.zeros(M, device=dev)
        w2v_t = torch.as_tensor(w2v_all, dtype=torch.float32, device=dev)
        Ph = torch.cat([P, torch.ones(V, 1, device=dev)], 1)
        big = 1e6
        # ---- cover_target == "gauss": score views by the gaussians they see ----
        _Qq = None
        if cover_target == "gauss":
            _K = min(int(n_gauss_probe), int(mu.shape[0]))
            _gi = torch.linspace(0, int(mu.shape[0]) - 1, _K).long().to(dev)
            _smid = torch.sort(sc[_gi], 1).values[:, 1]
            if gauss_rstar > 0.0:
                # T1-B: smooth r* quality instead of the binary visibility test
                _Qq = _gauss_quality(mu[_gi], _smid, ov[_gi], w2v_t, fx, fy,
                                     float(W), float(H), dev,
                                     gauss_rstar, gauss_rsigma,
                                     batch=batch_cams)
                _Ag = _Qq > 0
                _wg = ov[_gi].clone()
            else:
                _Ag = _gauss_cov(mu[_gi], sc[_gi].amax(1), w2v_t, fx, fy,
                                 float(W), float(H), Wz, Hz, dev,
                                 batch=batch_cams, delta=float(Delta),
                                 gate=((r_px_min, r_px_max, d_min, d_max)
                                       if cover_gate else None))
                _wg = ov[_gi].clone()
            _vox_iter = range(0)          # skip the surface-voxel coverage loop
        else:
            _vox_iter = range(0, M, batch_cams)
        for s0 in _vox_iter:
            s1 = min(s0 + batch_cams, M)
            Wm = w2v_t[s0:s1]                                    # (B,4,4)
            if noblas:
                pc = (Ph[None, :, None, :] * Wm[:, None, :, :]).sum(-1)
            else:
                pc = torch.einsum("vj,bkj->bvk", Ph, Wm)          # (B,V,4)
            z = pc[..., 2]
            zc = z.clamp_min(1e-6)
            uu = fx_low * pc[..., 0] / zc + 0.5 * Wz
            vv = fy_low * pc[..., 1] / zc + 0.5 * Hz
            ui = torch.floor(uu).long()
            vi = torch.floor(vv).long()
            zpos = z > 1e-6
            inimg = (ui >= 0) & (ui < Wz) & (vi >= 0) & (vi < Hz) & zpos
            ok = inimg
            if use_scale_gate:
                rpx = fx * s_v[None, :] / zc
                ok = ok & (rpx >= r_px_min) & (rpx <= r_px_max)
                if gate_rel > 0.0:
                    # gate window centred on the *actual* camera-to-scene distance
                    cam_c0 = (-Wm[:, :3, :3].transpose(1, 2)
                              @ Wm[:, :3, 3][..., None])[..., 0]
                    dref = (cam_c0 - ctr_scene[None, :]).norm(dim=1)
                    ok = ok & (z >= (dref * 0.35)[:, None]) \
                            & (z <= (dref * 2.5)[:, None])
                else:
                    ok = ok & (z >= d_min) & (z <= d_max)
            if use_normal_gate:
                cam_c = (-Wm[:, :3, :3].transpose(1, 2) @ Wm[:, :3, 3][..., None])[..., 0]
                dvec = cam_c[:, None, :] - P[None, :, :]
                dvec = dvec / dvec.norm(dim=2, keepdim=True).clamp_min(1e-9)
                cosang = (dvec * n_v[None, :, :]).sum(2).abs()
                ok = ok & (cosang >= 0.173)
            if use_zbuf:
                pix = vi.clamp(0, Hz - 1) * Wz + ui.clamp(0, Wz - 1)
                zfill = torch.where(ok, z, torch.full_like(z, big))
                zbuf = torch.full((s1 - s0, Hz * Wz), big, device=dev)
                zbuf.scatter_reduce_(1, pix.clamp(0, Hz * Wz - 1), zfill,
                                     reduce="amin", include_self=True)
                zb = zbuf.view(s1 - s0, Hz, Wz)[:, None]
                zb = -F.max_pool2d(-zb, 3, stride=1, padding=1)[:, 0]
                Zt = zb.reshape(s1 - s0, Hz * Wz).gather(1, pix.clamp(0, Hz * Wz - 1))
                ok = ok & (z <= Zt + tol_gamma * Delta)
            A[s0:s1] = ok
            if BinArr is not None:
                cam_c = (-Wm[:, :3, :3].transpose(1, 2) @ Wm[:, :3, 3][..., None])[..., 0]
                dvp = P[None, :, :] - cam_c[:, None, :]
                t1 = (dvp * e1h[None, None, :]).sum(2)
                t2 = (dvp * e2h[None, None, :]).sum(2)
                ang = torch.atan2(t2, t1)
                bb = torch.floor((ang + math.pi) / (2.0 * math.pi / dir_bins))
                BinArr[s0:s1] = torch.where(
                    ok, bb.long().clamp(0, dir_bins - 1),
                    torch.zeros_like(bb, dtype=torch.long)).to(torch.int8)
            if cos_weight > 0.0:
                # M4: front-facing cosine weight, I_vis * max(0.1, max(0,-d.n_out)^g)
                cam_c = (-Wm[:, :3, :3].transpose(1, 2) @ Wm[:, :3, 3][..., None])[..., 0]
                dv = P[None, :, :] - cam_c[:, None, :]
                dv = dv / dv.norm(dim=2, keepdim=True).clamp_min(1e-9)
                cosv = -(dv * n_out[None, :, :]).sum(2).clamp_min(0.0)
                a_w = cosv.clamp_min(1e-3).pow(cos_weight).clamp_min(0.1)
                Aw[s0:s1] = torch.where(ok, a_w, torch.zeros_like(a_w))
            elif Aw is not None:
                # log-Gaussian scale kernel around rref (continuous, no hard cut)
                rpx_w = (fx * s_v[None, :] / zc).clamp_min(1e-6)
                lw = torch.log(rpx_w / max(rref, 1e-6)) / float(soft_scale)
                Aw[s0:s1] = torch.where(ok, torch.exp(-0.5 * lw * lw),
                                        torch.zeros_like(lw))
            n_vis[s0:s1] = ok.float().sum(1)
        if cover_target == "gauss":
            A = _Ag
            V = int(A.shape[1])
            w_v = _wg
            n_vis = A.float().sum(1)
            Aw = None
            BinArr = None
        _mark("step3_cov")
        # ------------------------------------------------------------------
        # Step 4: submodular k-coverage greedy selection
        # ------------------------------------------------------------------
        # phi(c) spec values: [0, 1.0, 1.5, 1.75] for k=3, capped thereafter
        if cov_k == 3:
            phi = [0.0, 1.0, 1.5, 1.75]
        elif cov_k == 1:
            phi = [0.0, 1.0]
        elif cov_k == 2:
            phi = [0.0, 1.0, 1.5]
        else:
            phi = [0.0] + [1.0 + 0.5 * (1.0 - 0.5 ** (c - 1)) for c in range(1, cov_k + 1)]
        if phi_override:
            phi = [float(x) for x in str(phi_override).split(",") if x != ""]
        delta_lut = torch.as_tensor(
            [phi[min(c + 1, len(phi) - 1)] - phi[min(c, len(phi) - 1)]
             for c in range(len(phi) + 1)], dtype=torch.float32, device=dev)
        cvec = torch.zeros(V, device=dev)
        chosen = []
        _curq = (torch.zeros(V, device=dev) if _Qq is not None else None)
        Af = Aw if Aw is not None else A.float()
        pos_t = torch.as_tensor(np.stack(eyes), dtype=torch.float32, device=dev)
        dir_t = torch.as_tensor(np.stack(mdirs), dtype=torch.float32, device=dev)
        ang_full = math.radians(dedupe_deg)
        ang_near = math.radians(near_deg)
        lit = None
        wquad = None
        if BinArr is not None:
            lit = torch.zeros(V, dir_bins, dtype=torch.bool, device=dev)
            wq = [1.0, 0.5, 0.25, 0.1][:dir_bins]
            while len(wq) < dir_bins:
                wq.append(0.1)
            wquad = torch.tensor(wq, dtype=torch.float32, device=dev)
        for b in range(n_views):
            if _Qq is not None:
                # T1-B: smooth r* quality, either facility-location (max) or a
                # saturating sum (cap); both are monotone submodular.
                if gauss_agg == "max":
                    gain = (_Qq - _curq[None, :]).clamp_min(0.0).sum(1)
                else:
                    head = (float(gauss_cap) - _curq).clamp_min(0.0)
                    gain = torch.minimum(_Qq, head[None, :]).sum(1)
            elif BinArr is not None:
                # M7a: marginal gain = newly lit azimuth quadrants, weight [1,.5,.25,.1]
                pop = lit.sum(1).clamp(max=dir_bins - 1)
                wb = wquad[pop]
                bi = BinArr.reshape(-1).long()
                vidx = torch.arange(V, device=dev).repeat(M)
                litg = lit[vidx, bi].reshape(M, V)
                newv = A & (~litg)
                gain = (newv.float() * (w_v * wb)[None, :]).sum(1)
            elif use_cos:
                # M4: F(S) = sum_v w_v * min(C_max, sum_c a_cv); marginal gain is
                # sum_v w_v * min(a_mv, headroom_v), monotone submodular.
                head = (float(cmax) - cvec).clamp(0.0, float(cmax))
                if noblas:
                    gain = (torch.minimum(Af, head[None, :])
                            * w_v[None, :]).sum(1)
                else:
                    gain = (torch.minimum(Af, head[None, :])
                            * w_v[None, :]).sum(1)
            else:
                c_int = cvec.long().clamp(max=delta_lut.numel() - 1)
                delta = delta_lut[c_int]
                if noblas:
                    gain = (Af * (w_v * delta)[None, :]).sum(1)
                else:
                    gain = Af @ (w_v * delta)
            gain[chosen] = -1e30
            if chosen:
                dpos = (pos_t - pos_t[chosen][:, None, :]).norm(dim=2)  # (|S|,M)
                ddir = (dir_t[None, :, :] * dir_t[chosen][:, None, :]).sum(2)
                same_place = dpos.min(0).values < 0.1 * Lmax
                ang_min = torch.acos(ddir.clamp(-1, 1)).min(0).values
                blocked = same_place & (ang_min < ang_full)
                near = same_place & (ang_min < ang_near)
                gain = torch.where(blocked, torch.full_like(gain, -1e30),
                                   torch.where(near, 0.5 * gain, gain))
            if float(gain.max()) <= 0.0:
                # nothing left to gain: stop rather than duplicate poses
                break
            m = int(torch.argmax(gain))
            chosen.append(m)
            cvec = cvec + Af[m]
            if _curq is not None:
                if gauss_agg == "max":
                    _curq = torch.maximum(_curq, _Qq[m])
                else:
                    _curq = _curq + _Qq[m]
            if lit is not None:
                vis_m = A[m]
                if bool(vis_m.any()):
                    lit[vis_m] = lit[vis_m].scatter(
                        1, BinArr[m][vis_m].long()[:, None], True)
            if verbose:
                print(f"[auto_virt] pick {b}: cam {m} gain {float(gain[m]):.3f}",
                      flush=True)
        if diverse_quota and len(chosen) >= 4:
            # T4-B: guarantee a minimum number of picks in each distance band
            # (near / mid / far from the density focus) so the 8 slots cannot all
            # collapse onto one radius ring.
            Pn = P.detach().cpu().numpy()
            nv = Nocc[surf_idx].detach().cpu().numpy()
            k5 = max(8, int(0.05 * Pn.shape[0]))
            top5 = np.argsort(-nv)[:k5]
            cc = (Pn[top5] * nv[top5, None]).sum(0) / max(nv[top5].sum(), 1e-9)
            r_all = np.linalg.norm(np.stack(eyes) - cc[None, :], axis=1)
            q1, q2 = np.quantile(r_all, [0.33, 0.66])
            bid = np.digitize(r_all, [q1, q2]).astype(int)
            quota = [2, 3, 3]
            cnt = [0, 0, 0]
            for m in chosen:
                cnt[bid[m]] += 1
            g_np = gain.detach().cpu().numpy()
            order_g = np.argsort(-g_np)
            for b in range(3):
                while cnt[b] < quota[b]:
                    over = max(range(3), key=lambda x: cnt[x] - quota[x])
                    if cnt[over] <= quota[over]:
                        break
                    pick = next((int(m) for m in order_g
                                 if m not in chosen and bid[m] == b
                                 and g_np[m] > -1e20), None)
                    if pick is None:
                        break
                    victim = next((mm for mm in reversed(chosen)
                                   if bid[mm] == over), None)
                    if victim is None:
                        break
                    chosen.remove(victim)
                    cnt[over] -= 1
                    chosen.append(pick)
                    cnt[b] += 1
        n_greedy_picks = len(chosen)
        if len(chosen) < n_views:
            # pad with the liveliest unused cameras (keeps B fixed)
            order = torch.argsort(n_vis, descending=True).tolist()
            for m in order:
                if len(chosen) >= n_views:
                    break
                if m not in chosen:
                    chosen.append(m)
        _mark("step4")

        cams = []
        for m in chosen:
            cams.append(dict(name=names[m], w2v=np.asarray(w2v_all[m], np.float64),
                             fx=float(fx), fy=float(fy), W=float(W), H=float(H),
                             C=np.asarray(eyes[m], np.float64)))
        if return_pool:
            cams = [dict(name=names[m], w2v=np.asarray(w2v_all[m], np.float64),
                         fx=float(fx), fy=float(fy), W=float(W), H=float(H),
                         C=np.asarray(eyes[m], np.float64)) for m in range(M)]

    stats = dict(
        mode=mode, n_views=len(cams), n_valid=int(Nv), n_surface=n_surf,
        n_cand=int(cand.numel()), n_pos=int(pos.shape[0]), n_cams=int(M),
        n_surf_used=int(V), Lmax=Lmax, Delta=Delta, s_med=s_med,
        d_target=d_target, d_min=d_min, d_max=d_max, fov=fov_deg,
        d_over_L=float(d_target / max(Lmax, 1e-9)),
        dmode=dmode, r_ref=rref, core_aabb=float(core_aabb),
        soft_scale=float(soft_scale), s_mid_med=float(torch.median(s_mid_sub).item()),
        pos_mode=pos_mode, pca_eta=pca_eta, pca_kind=pca_kind,
        dir_bins=int(dir_bins), aniso_weight=float(aniso_weight),
        tau_occ=tau_occ, n_cavity=n_cav, n_cavity6=n_cav6, fallback=fallback,
        fg_keep=fg_keep,
        cov_frac=float(A.any(0).float().mean().item()) if M else 0.0,
        n_greedy=n_greedy_picks,
        domain_split=(
            f"{sum(1 for m in chosen if m < n_domain_a)}/"
            f"{sum(1 for m in chosen if m >= n_domain_a)}"
            if extra_cams else ""),
        vis_frac=float(A.float().mean().item()) if M else 0.0,
        cam_spread=float(np.asarray(eyes)[chosen].std(0).mean()) if chosen else 0.0,
        n_vis=[float(x) for x in n_vis.tolist()],
        cover_mean=float(A.float().sum(1).mean().item()) if M else 0.0,
        t_step1_ms=(stamps.get("step1", 0) - stamps.get("start", 0)) * 1e3,
        t_step2_ms=(stamps.get("step2_vox", 0) - stamps.get("step1", 0)) * 1e3,
        t_step3_ms=(stamps.get("step3_cov", 0) - stamps.get("step2_vox", 0)) * 1e3,
        t_step4_ms=(stamps.get("step4", 0) - stamps.get("step3_cov", 0)) * 1e3,
        t_total_ms=(stamps.get("step4", 0) - stamps.get("start", 0)) * 1e3,
        chosen=chosen,
    )
    if return_stats:
        return cams, stats
    return cams
