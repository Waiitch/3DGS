"""PICS -- Pair-conditioned Information and Coverage Sampling for O2cal.

Implements the two-level structure of the 2026-11-16 plan (Part IV, 11.1):

  Level 1  cheap screen, all pairs x all fit cameras
           -> projected centre, conservative pixel radius, in-image footprint
              fraction q_{p,c} and in-image footprint area (a proxy for d)
  Level 2  exact O2cal patch statistics, only for the views that get selected
           -> d_{p,c} = sum_x G_m^2, n_{p,c} = sum_x G_m T_p, a_{p,c} = n/d

Selection strategies (plan 16.1 / 23):

  sup    P2  top-B by support fraction q   (only fixes zero-support)
  info   P3  top-B by in-image footprint area (scalar-FI proxy, d)
  strat  P4  trajectory-stratified: B bins over the shooting order, best
             supported camera inside each bin (keeps fast8's distribution)

`patch_stats` is the single exact primitive: given (pair, camera) rows it
returns d, n and the O2cal validity mask, so the offline evaluator and the
in-pipeline kernel share one implementation of the math (identical to
`o2gpu.opacity_o2cal_gpu`, which stays the reference for the fixed 8 views).
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
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


def _lmax2x2(S2):
    a, b = S2[..., 0, 0], S2[..., 0, 1]
    c, d = S2[..., 1, 0], S2[..., 1, 1]
    tr = a + d
    det = a * d - b * c
    disc = torch.sqrt(torch.clamp(tr * tr * 0.25 - det, min=0.0))
    return tr * 0.5 + disc


def _det2(S2):
    return S2[..., 0, 0] * S2[..., 1, 1] - S2[..., 0, 1] * S2[..., 1, 0]


def _gauss(mu2d, S2, px, py):
    a, b = S2[..., 0, 0], S2[..., 0, 1]
    c, d = S2[..., 1, 0], S2[..., 1, 1]
    det = torch.clamp(a * d - b * c, min=1e-12)
    i00 = d / det
    i11 = a / det
    i01 = -b / det
    dx = px - mu2d[..., 0:1]
    dy = py - mu2d[..., 1:2]
    qq = i00[..., None] * dx * dx + 2.0 * i01[..., None] * dx * dy \
        + i11[..., None] * dy * dy
    return torch.exp(-0.5 * torch.clamp(qq, 0.0, 60.0))


# --------------------------------------------------------------------------- #
class CamPool:
    """All fit cameras of one scene, stacked once on the device."""

    def __init__(self, cams, dev):
        self.dev = dev
        self.names = [c["name"] for c in cams]
        self.V = len(cams)
        self.M = None          # (V,V) track-overlap matrix, set by the caller
        self.w2v = torch.as_tensor(
            np.stack([np.asarray(c["w2v"], np.float32) for c in cams]), device=dev)
        self.fx = torch.as_tensor(np.asarray([c["fx"] for c in cams], np.float32),
                                  device=dev)
        self.fy = torch.as_tensor(np.asarray([c["fy"] for c in cams], np.float32),
                                  device=dev)
        self.W = torch.as_tensor(np.asarray([c["W"] for c in cams], np.float32),
                                 device=dev)
        self.H = torch.as_tensor(np.asarray([c["H"] for c in cams], np.float32),
                                 device=dev)
        self.fmax = torch.maximum(self.fx, self.fy)
        R = self.w2v[:, :3, :3]
        t = self.w2v[:, :3, 3]
        # camera centres C = -R^T t, written as elementwise multiply+sum: the
        # pipeline runs with --no-blas so that cuBLAS is never initialised, and a
        # 3x3 batched matmul here would silently pay its ~0.7 s handle creation.
        self.C = -torch.stack([R[:, 0, i] * t[:, 0] + R[:, 1, i] * t[:, 1]
                               + R[:, 2, i] * t[:, 2] for i in range(3)], -1)
        ax = R[:, 2, :]                                             # third row
        self.axis = ax / torch.clamp(ax.norm(dim=1, keepdim=True), min=1e-12)

    # -- cheap level 1 --------------------------------------------------- #
    @torch.no_grad()
    def screen(self, mu, sc, sigma_cut=2.0, chunk=16384, want_area=True):
        """Frustum + footprint screen of (P,) merged gaussians against all views.

        The pixel radius is the *conservative* bound
            r = sigma_cut * max(sc) * max(fx,fy) / z
        (the exact EWA radius can never exceed it), so nothing that a later
        exact test would accept is dropped here.

        Returns dict(q=(P,V) float16 in-image footprint fraction,
                     area=(P,V) float16 log1p(in-image footprint area px^2))
        """
        P = int(mu.shape[0])
        V = self.V
        q = torch.zeros((P, V), dtype=torch.float16, device=self.dev)
        area = (torch.zeros((P, V), dtype=torch.float16, device=self.dev)
                if want_area else None)
        rw = (sigma_cut * sc.amax(1)).to(torch.float32)            # (P,)
        R = self.w2v[:, :3, :3]
        t = self.w2v[:, :3, 3]
        for s0 in range(0, P, chunk):
            s1 = min(P, s0 + chunk)
            m = mu[s0:s1].to(torch.float32)
            r = rw[s0:s1, None]
            pc = (m[:, 0:1, None] * R[None, :, :, 0]
                  + m[:, 1:2, None] * R[None, :, :, 1]
                  + m[:, 2:3, None] * R[None, :, :, 2]) + t[None]   # (c,V,3)
            z = pc[..., 2]
            ok = z > 1e-6
            zc = torch.clamp(z, min=1e-9)
            px = self.fx[None] * pc[..., 0] / zc + 0.5 * self.W[None]
            py = self.fy[None] * pc[..., 1] / zc + 0.5 * self.H[None]
            rp = r * self.fmax[None] / zc                          # (c,V)
            x0 = torch.clamp(px - rp, min=0.0)
            x1 = torch.minimum(px + rp, self.W[None])
            y0 = torch.clamp(py - rp, min=0.0)
            y1 = torch.minimum(py + rp, self.H[None])
            wx = torch.clamp(x1 - x0, min=0.0)
            wy = torch.clamp(y1 - y0, min=0.0)
            tot = torch.clamp(4.0 * rp * rp, min=1e-6)
            frac = torch.where(ok, (wx * wy) / tot, torch.zeros_like(wx))
            frac = torch.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
            q[s0:s1] = frac.to(torch.float16)
            if area is not None:
                a_in = torch.where(ok, wx * wy, torch.zeros_like(wx))
                area[s0:s1] = torch.log1p(a_in).to(torch.float16)
            del pc, px, py, rp, wx, wy, frac
        return {"q": q, "area": area}

    # -- selection ------------------------------------------------------- #
    def _bins(self, B):
        return torch.div(torch.arange(self.V, device=self.dev) * B,
                         self.V, rounding_mode="floor")                   # (V,)

    @torch.no_grad()
    def select(self, scr, strategy="sup", B=8, qmin=0.05, fast_ids=None,
               nbins=None, pref_ids=None, track_lam=None):
        """Return (P,B) int32 camera ids; padding is -1.

        strategy:
            sup     top-B by support fraction q (ties -> lowest index)
            info    top-B by in-image footprint area
            strat   one camera per trajectory bin (nbins = B by default)
            fill    stratified fill: round-robin over nbins bins taking the 1st
                    best, then the 2nd best ... until B slots are used
            repair  strat, but keep the historical fast camera whenever it has
                    support and only repair the bins where it does not
            pref    keep the given `pref_ids` (e.g. the global cover set) in the
                    slots where they have support, then stratified-fill the rest
            guard   minimal intervention: use the historical fast set whenever it
                    has any support, and only fall back to `repair` for the pairs
                    fast cannot fit at all
            track   support gate + COLMAP track-redundancy penalty (needs pool.M)
        """
        q = scr["q"].to(torch.float32)
        # deterministic tie-break towards the lower camera index
        tie = -1e-6 * (torch.arange(self.V, device=self.dev, dtype=torch.float32)
                       [None] / max(self.V, 1))
        qs = q + tie
        P = q.shape[0]
        if strategy in ("sup", "info"):
            if strategy == "sup":
                score = qs
            else:
                if scr["area"] is None:
                    raise ValueError("strategy 'info' needs screen(want_area=True)")
                score = scr["area"].to(torch.float32) + tie * 1e3
            score = torch.where(q >= qmin, score,
                                torch.full_like(score, -1e8))
            vals, ids = torch.topk(score, B, dim=1)
            keep = vals > -1e7
            ids = ids.to(torch.int32)
            idx = torch.where(keep, ids, torch.full_like(ids, -1))
        elif strategy in ("strat", "repair", "fill"):
            if strategy == "repair" and fast_ids is None:
                raise ValueError("strategy 'repair' needs fast_ids")
            nb = int(nbins or (B if strategy != "fill" else max(1, min(B, 8))))
            bins = self._bins(nb)
            idx = torch.full((P, B), -1, dtype=torch.int32, device=self.dev)
            fmap = {}
            if fast_ids is not None:
                for fid in fast_ids:
                    fmap[int(bins[fid].item())] = int(fid)
            if strategy == "fill":
                per_bin = -(-B // nb)
                for b in range(nb):
                    cam_b = torch.nonzero(bins == b, as_tuple=False).flatten()
                    k = min(per_bin, int(cam_b.numel()))
                    vals, ids = qs[:, cam_b].topk(k, dim=1)
                    ids = cam_b[ids]
                    for r in range(k):
                        slot = r * nb + b
                        if slot >= B:
                            break
                        ok = vals[:, r] >= qmin
                        idx[:, slot] = torch.where(
                            ok, ids[:, r].to(torch.int32),
                            torch.full_like(ids[:, r], -1, dtype=torch.int32))
            else:
                big = torch.full_like(qs, -1e8)
                for b in range(nb):
                    sel_b = (bins == b).to(torch.float32).unsqueeze(0)    # (1,V)
                    sb = qs * sel_b + big * (1.0 - sel_b)
                    val, ids = sb.max(dim=1)
                    ok = val >= qmin
                    if strategy == "repair" and b in fmap:
                        fid = fmap[b]
                        use_f = q[:, fid] >= qmin
                        ids = torch.where(use_f, torch.full_like(ids, fid), ids)
                        ok = ok | use_f
                    idx[:, b] = torch.where(ok, ids.to(torch.int32),
                                            torch.full_like(ids, -1,
                                                            dtype=torch.int32))
        elif strategy == "pref":
            if pref_ids is None:
                raise ValueError("strategy 'pref' needs pref_ids")
            pref = [int(c) for c in pref_ids][:B]
            idx = torch.full((P, B), -1, dtype=torch.int32, device=self.dev)
            used = torch.zeros((P, self.V), dtype=torch.bool, device=self.dev)
            rows = torch.arange(P, device=self.dev)
            big = torch.full_like(qs, -1e8)
            bins = self._bins(int(nbins or max(1, min(B, 8))))
            # 1) keep every preferred camera that actually has support
            for j, c in enumerate(pref):
                okc = q[:, c] >= qmin
                idx[okc, j] = c
                used[rows[okc], c] = True
            # 2) repair each empty slot from its preferred camera's trajectory
            #    bin, so the spatial spread of the preferred set is preserved
            for j, c in enumerate(pref):
                empty = idx[:, j] < 0
                if not bool(empty.any()):
                    continue
                b = int(bins[c].item())
                sel_b = (bins == b).to(torch.float32).unsqueeze(0)       # (1,V)
                free = torch.where(used, big, qs)
                sb = free * sel_b + big * (1.0 - sel_b)
                val, ids = sb.max(dim=1)
                ok = empty & (val >= qmin)
                idx[ok, j] = ids[ok].to(torch.int32)
                used[rows[ok], ids[ok]] = True
            # 3) any slot still empty (its bin had nothing) -> round-robin fill
            slot = len(pref)
            while slot < B:
                progressed = False
                for b in range(int(bins.max().item()) + 1):
                    if slot >= B:
                        break
                    sel_b = (bins == b).to(torch.float32).unsqueeze(0)
                    free = torch.where(used, big, qs)
                    sb = free * sel_b + big * (1.0 - sel_b)
                    val, ids = sb.max(dim=1)
                    ok = val >= qmin
                    if not bool(ok.any()):
                        continue
                    progressed = True
                    idx[:, slot] = torch.where(
                        ok, ids.to(torch.int32),
                        torch.full_like(ids, -1, dtype=torch.int32))
                    used[rows[ok], ids[ok]] = True
                    slot += 1
                if not progressed:
                    break
        elif strategy == "all":
            # plan §15 H: every supported view (the a_p^all definition itself).
            # Per-pair variable count, -1 padded; the kernel already accepts that.
            cand = q >= qmin
            cnt = cand.sum(1)
            kmax = int(cnt.max().item()) if P else 0
            kmax = max(kmax, 1)
            idx = torch.full((P, kmax), -1, dtype=torch.int32, device=self.dev)
            pos = torch.cumsum(cand.to(torch.int64), 1) - 1
            cols = torch.arange(self.V, device=self.dev)[None, :].expand(P, self.V)
            rows = torch.arange(P, device=self.dev)[:, None].expand(P, self.V)
            keep = cand & (pos < kmax)
            idx[rows[keep], pos[keep].clamp(min=0)] = cols[keep].to(torch.int32)
        elif strategy == "dir":
            # plan §15 G7, pair-conditioned: support gate minus a penalty for
            # viewing-direction similarity (directional novelty term).
            lam = 1.0 if track_lam is None else float(track_lam)
            C = self.C.to(torch.float32)
            dirs = C - C.mean(0, keepdim=True)
            dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-9)
            cand = q >= qmin
            base = torch.where(cand, qs, torch.full_like(qs, -1e8))
            red = torch.zeros_like(qs)
            used = torch.zeros((P, self.V), dtype=torch.bool, device=self.dev)
            idx = torch.full((P, B), -1, dtype=torch.int32, device=self.dev)
            for step in range(B):
                score = torch.where(cand & ~used, base - lam * red,
                                    torch.full_like(base, -1e8))
                i = score.argmax(1)
                ok = score.gather(1, i[:, None])[:, 0] > -1e7
                if not bool(ok.any()):
                    break
                idx[ok, step] = i[ok].to(torch.int32)
                rows = torch.nonzero(ok, as_tuple=False).flatten()
                used[rows, i[rows]] = True
                red[rows] = torch.maximum(
                    red[rows], (dirs[i[rows]] @ dirs.transpose(0, 1)).abs())
        elif strategy == "track":
            # support gate + diminishing redundancy (plan 11.6 / 16.1 P6):
            # score = q - lam * max_{s in S} M[c,s], with M the COLMAP track
            # overlap between the candidate and the already selected cameras.
            if self.M is None:
                raise ValueError("strategy 'track' needs pool.M (track overlap)")
            lam = 1.0 if track_lam is None else float(track_lam)
            cand = q >= qmin
            base = torch.where(cand, qs, torch.full_like(qs, -1e8))
            red = torch.zeros_like(qs)
            used = torch.zeros((P, self.V), dtype=torch.bool, device=self.dev)
            idx = torch.full((P, B), -1, dtype=torch.int32, device=self.dev)
            for step in range(B):
                score = torch.where(cand & ~used, base - lam * red,
                                    torch.full_like(base, -1e8))
                i = score.argmax(1)
                ok = score.gather(1, i[:, None])[:, 0] > -1e7
                if not bool(ok.any()):
                    break
                idx[ok, step] = i[ok].to(torch.int32)
                rows = torch.nonzero(ok, as_tuple=False).flatten()
                used[rows, i[rows]] = True
                # accumulate the redundancy over ALL cameras of the pairs that
                # made a pick.  Masking this update per (pair, camera) -- e.g.
                # torch.where(used, maximum(red, M[i]), red) -- silently zeroes
                # every camera except the one just picked, so the penalty never
                # influences the next step (caught by the ground-truth loop).
                red[rows] = torch.maximum(red[rows], self.M[i[rows]].to(red.dtype))
        elif strategy == "guard":
            if fast_ids is None:
                raise ValueError("strategy 'guard' needs fast_ids")
            sel_rep = self.select(scr, strategy="repair", B=B, qmin=qmin,
                                  fast_ids=fast_ids, nbins=nbins)
            m = min(B, len(fast_ids))
            fid_t = torch.as_tensor([int(c) for c in fast_ids[:m]], device=self.dev,
                                    dtype=torch.int32)
            has = (q[:, fid_t.to(torch.int64)] >= qmin).any(1)          # (P,)
            base = torch.full((P, m), -1, dtype=torch.int32, device=self.dev)
            base[:, :m] = fid_t[None]
            idx = torch.where(has[:, None], base, sel_rep[:, :m])
        else:
            raise ValueError(f"unknown strategy {strategy}")
        return idx

    # -- level 2 --------------------------------------------------------- #
    @torch.no_grad()
    def signature(self, state, comp, pairs, pid, cid, grid=9, pool=3,
                  sigma_cut=2.0, dtype=torch.float32, chunk=65536):
        """Response-space signature psi_{p,c} (plan 11.3).

        IMPORTANT: the 4x4 sample grid the plan suggests is *not* usable with the
        adaptive patch -- the patch spans +-2 sigma_max in both image axes, so on
        a merged (strongly elongated) gaussian a 4-point grid has no sample on the
        ridge and the whole response map collapses to ~0 (measured: mean log d
        -4.4, rank-1 signature).  We therefore sample on the same 9x9 grid the
        fit itself uses and pool it to a 3x3 map.

        psi = [ pool3x3(G_m) (9), pool3x3(T_p) (9), log d, log1p(a),
                log l1, log l2, IoU(G_i,G_j), rr_rel ]        -- 24 raw dims
        Returns (K, 24) float32 and the O2cal validity mask (K,).
        """
        K = int(pid.numel())
        dev = self.dev
        g = torch.linspace(-1.0, 1.0, grid, device=dev, dtype=dtype)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        gx = gx.reshape(-1).to(torch.float64)
        gy = gy.reshape(-1).to(torch.float64)
        G = gx.numel()
        side = int(round(G ** 0.5))
        blk = side // pool
        ncell = pool * pool
        nscal = 6
        out = torch.zeros((K, 2 * ncell + nscal), dtype=dtype, device=dev)
        ok_out = torch.zeros(K, dtype=torch.bool, device=dev)
        op = state["op"]
        i_all = pairs[:, 0].to(torch.int64)
        j_all = pairs[:, 1].to(torch.int64)
        for s0 in range(0, K, chunk):
            s1 = min(K, s0 + chunk)
            sl = slice(s0, s1)
            pi = pid[sl].to(torch.int64)
            ci = cid[sl].to(torch.int64)
            w2v = self.w2v[ci].to(torch.float64)
            fx = self.fx[ci].to(torch.float64)
            fy = self.fy[ci].to(torch.float64)
            Wd = self.W[ci].to(torch.float64)
            Hd = self.H[ci].to(torch.float64)
            ii, jj = i_all[pi], j_all[pi]
            ai = op[ii].to(torch.float64)
            aj = op[jj].to(torch.float64)
            mu2d_i, S2_i, v_i = self._proj_inst(state["mu"][ii], state["sc"][ii],
                                                state["q"][ii], w2v, fx, fy, Wd, Hd,
                                                torch.float64)
            mu2d_j, S2_j, v_j = self._proj_inst(state["mu"][jj], state["sc"][jj],
                                                state["q"][jj], w2v, fx, fy, Wd, Hd,
                                                torch.float64)
            mu2d_m, S2_m, v_m = self._proj_inst(comp["mu_m"][pi], comp["sc_m"][pi],
                                                comp["q_m"][pi], w2v, fx, fy, Wd, Hd,
                                                torch.float64)
            rr = sigma_cut * torch.sqrt(torch.clamp(_lmax2x2(S2_m), min=1e-9))
            cx, cy = mu2d_m[:, 0], mu2d_m[:, 1]
            ok = v_i & v_j & v_m & (rr > 1e-3)
            ok &= torch.isfinite(cx) & torch.isfinite(cy) & torch.isfinite(rr)
            ok &= (cx > -rr) & (cx < Wd + rr) & (cy > -rr) & (cy < Hd + rr)
            ok_out[sl] = ok
            # eigenvectors of the merged 2-D covariance (log l1, log l2)
            tr = S2_m[:, 0, 0] + S2_m[:, 1, 1]
            det = (S2_m[:, 0, 0] * S2_m[:, 1, 1] - S2_m[:, 0, 1] * S2_m[:, 1, 0])
            disc = torch.sqrt(torch.clamp(tr * tr * 0.25 - det, min=0.0))
            l1 = tr * 0.5 + disc
            l2 = torch.clamp(tr * 0.5 - disc, min=1e-18)
            bv = torch.nonzero(ok, as_tuple=False).flatten()
            if bv.numel() == 0:
                continue
            rrk = rr[bv]
            px = torch.minimum(torch.maximum(cx[bv, None] + gx[None, :] * rrk[:, None],
                                             torch.zeros((), dtype=torch.float64,
                                                         device=dev)),
                               (Wd - 1.0)[bv, None])
            py = torch.minimum(torch.maximum(cy[bv, None] + gy[None, :] * rrk[:, None],
                                             torch.zeros((), dtype=torch.float64,
                                                         device=dev)),
                               (Hd - 1.0)[bv, None])
            Gi = _gauss(mu2d_i[bv], S2_i[bv], px, py)
            Gj = _gauss(mu2d_j[bv], S2_j[bv], px, py)
            Gm = _gauss(mu2d_m[bv], S2_m[bv], px, py)
            Tp = 1.0 - (1.0 - ai[bv][:, None] * Gi) * (1.0 - aj[bv][:, None] * Gj)
            dd = (Gm * Gm).sum(1)
            nn = (Gm * Tp).sum(1)
            iou = (torch.minimum(Gi, Gj).sum(1)
                   / torch.clamp(torch.maximum(Gi, Gj).sum(1), min=1e-9))
            def _pool2d(x):
                """(n, side*side) -> (n, pool*pool) block mean, any side >= pool."""
                grp = torch.div(torch.arange(side, device=x.device) * pool, side,
                                rounding_mode="floor")
                cell = (grp[:, None] * pool + grp[None, :]).reshape(-1)
                acc = torch.zeros((x.shape[0], pool * pool), dtype=x.dtype,
                                  device=x.device)
                acc.index_add_(1, cell, x)
                cnt = torch.zeros(pool * pool, dtype=x.dtype, device=x.device)
                cnt.index_add_(0, cell, torch.ones_like(cell, dtype=x.dtype))
                return acc / cnt.clamp(min=1.0)
            Gm_p = _pool2d(Gm)
            Tp_p = _pool2d(Tp)
            aa = nn / dd.clamp(min=1e-12)
            sig = torch.cat([Gm_p, Tp_p,
                             torch.log(dd.clamp(min=1e-12))[:, None],
                             torch.log1p(aa.clamp(0.0, 50.0))[:, None],
                             torch.log(l1[bv].clamp(min=1e-18))[:, None],
                             torch.log(l2[bv].clamp(min=1e-18))[:, None],
                             iou[:, None],
                             (rrk / torch.clamp(torch.sqrt(torch.max(Wd[bv], Hd[bv])),
                                                min=1e-9))[:, None]], 1)
            out[sl][bv] = sig.to(dtype)
        return out, ok_out

    # -- level 2 --------------------------------------------------------- #
    def _proj_inst(self, mu, sc, q, w2v, fx, fy, Wd, Hd, dtype):
        """EWA projection with one camera per instance.

        mu (K,3), sc (K,3), q (K,4), w2v (K,4,4), fx/fy/Wd/Hd (K,)
        -> mu2d (K,2), S2 (K,2,2), valid (K,)
        """
        mu = mu.to(dtype)
        sc = sc.to(dtype)
        q = q.to(dtype)
        ph = torch.cat([mu, torch.ones((mu.shape[0], 1), dtype=dtype,
                                      device=mu.device)], 1)
        # pc = ph @ w2v, per instance.  Every per-instance matrix product below
        # is expanded into broadcast multiply+sum: einsum/matmul on a stack of
        # tiny (3x3) matrices is one of the slowest paths in torch and this
        # projection dominates the pair-conditioned O2cal kernel.
        # pc = ph @ w2v, per instance.  Every per-instance matrix product below
        # is expanded into broadcast multiply+sum with (K,3)-shaped operands: a
        # (K,4,4) temporary is 0.5 GB per call at K=262k and dominated the
        # kernel's memory traffic (measured 4x slower than o2gpu otherwise).
        Rw = w2v[:, :3, :3]
        # pc = w2v @ ph  ->  pc[j] = sum_i w2v[j, i] * ph[i]  (columns of w2v)
        pc = (ph[:, 0:1] * w2v[:, :, 0] + ph[:, 1:2] * w2v[:, :, 1]
              + ph[:, 2:3] * w2v[:, :, 2] + ph[:, 3:4] * w2v[:, :, 3])   # (K,4)
        tz = pc[:, 2]
        valid = tz > 1e-6
        tzc = torch.clamp(tz, min=1e-9)
        tx = pc[:, 0] / tzc
        ty = pc[:, 1] / tzc
        mu2d = torch.stack([fx * tx + 0.5 * Wd, fy * ty + 0.5 * Hd], -1)
        j00 = fx / tzc
        j02 = -fx * tx / tzc
        j11 = fy / tzc
        j12 = -fy * ty / tzc
        # T = J @ Rw with J = [[j00, 0, j02], [0, j11, j12]]
        T = torch.empty((mu.shape[0], 2, 3), dtype=dtype, device=mu.device)
        T[:, 0, :] = j00[:, None] * Rw[:, 0, :] + j02[:, None] * Rw[:, 2, :]
        T[:, 1, :] = j11[:, None] * Rw[:, 1, :] + j12[:, None] * Rw[:, 2, :]
        # M = T @ Rc
        Rc = _rot(q)
        M = (T[:, :, 0:1] * Rc[:, 0:1, :]
             + T[:, :, 1:2] * Rc[:, 1:2, :]
             + T[:, :, 2:3] * Rc[:, 2:3, :])                       # (K,2,3)
        Mv = M * (sc * sc)[:, None, :]
        S2 = torch.empty((mu.shape[0], 2, 2), dtype=dtype, device=mu.device)
        S2[:, 0, 0] = (Mv[:, 0] * M[:, 0]).sum(-1)
        S2[:, 0, 1] = (Mv[:, 0] * M[:, 1]).sum(-1)
        S2[:, 1, 0] = S2[:, 0, 1]
        S2[:, 1, 1] = (Mv[:, 1] * M[:, 1]).sum(-1)
        S2 = S2 + 0.3 * torch.eye(2, dtype=dtype, device=mu.device)
        return mu2d, S2, valid

    @torch.no_grad()
    def patch_stats(self, state, comp, pairs, pid, cid, grid=9, sigma_cut=2.0,
                    dtype=torch.float64, chunk=65536, want_sums=False,
                    timers=None, want_t2=True, want_report=True,
                    sums_only=False):
        """Exact O2cal patch statistics for K (pair, camera) rows.

        pid, cid: (K,) int64, camera ids must already be un-padded (>= 0).
        want_sums adds the three arrays the DC SH correction needs
        (`v3`, `det_i`, `det_j`), which are already computed anyway.
        Returns dict with
            d   (K,) sum_x G_m^2                 (O2cal denominator term)
            n   (K,) sum_x G_m T_p                (O2cal numerator term)
            t2  (K,) sum_x T_p^2                  (needed for held-out residuals)
            ok  (K,) bool  the kernel's own validity mask (patch mask)
            cin (K,) bool  projected centre inside the image (report criterion)
            ov  (K,) bool  sigma_cut footprint overlaps the image
            a0  (K,)       fallback opacity for the pair
            rr  (K,)       sigma_cut pixel radius of the merged gaussian
        """
        K = int(pid.numel())
        dev = self.dev
        import time as _time

        g = torch.linspace(-1.0, 1.0, grid, device=dev, dtype=dtype)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)
        d = torch.zeros(K, dtype=dtype, device=dev)
        n = torch.zeros(K, dtype=dtype, device=dev)
        t2 = (torch.zeros(K, dtype=dtype, device=dev) if want_t2 else None)
        ok_out = torch.zeros(K, dtype=torch.bool, device=dev)
        cin_out = (torch.zeros(K, dtype=torch.bool, device=dev)
                   if want_report else None)
        ov_out = (torch.zeros(K, dtype=torch.bool, device=dev)
                  if want_report else None)
        a0_out = torch.zeros(K, dtype=dtype, device=dev)
        rr_out = torch.zeros(K, dtype=dtype, device=dev)
        if want_sums:
            v3_out = torch.zeros(K, dtype=torch.bool, device=dev)
            deti_out = torch.zeros(K, dtype=dtype, device=dev)
            detj_out = torch.zeros(K, dtype=dtype, device=dev)
        op = state["op"]
        i_all = pairs[:, 0].to(torch.int64)
        j_all = pairs[:, 1].to(torch.int64)
        for s0 in range(0, K, chunk):
            s1 = min(K, s0 + chunk)
            sl = slice(s0, s1)
            pi = pid[sl].to(torch.int64)
            ci = cid[sl].to(torch.int64)
            w2v = self.w2v[ci].to(dtype)
            fx = self.fx[ci].to(dtype)
            fy = self.fy[ci].to(dtype)
            Wd = self.W[ci].to(dtype)
            Hd = self.H[ci].to(dtype)
            ii = i_all[pi]
            jj = j_all[pi]
            ai = op[ii].to(dtype)
            aj = op[jj].to(dtype)
            a0 = torch.clamp(ai + aj - ai * aj, 0.0, 1.0)
            a0_out[sl] = a0

            _t0 = _time.perf_counter()
            mu2d_i, S2_i, v_i = self._proj_inst(state["mu"][ii], state["sc"][ii],
                                                state["q"][ii], w2v, fx, fy, Wd, Hd,
                                                dtype)
            mu2d_j, S2_j, v_j = self._proj_inst(state["mu"][jj], state["sc"][jj],
                                                state["q"][jj], w2v, fx, fy, Wd, Hd,
                                                dtype)
            mu2d_m, S2_m, v_m = self._proj_inst(comp["mu_m"][pi], comp["sc_m"][pi],
                                                comp["q_m"][pi], w2v, fx, fy, Wd, Hd,
                                                dtype)
            if timers is not None:
                timers["pc_proj"] = (timers.get("pc_proj", 0.0)
                                     + _time.perf_counter() - _t0)
            _t0 = _time.perf_counter()

            rr = sigma_cut * torch.sqrt(torch.clamp(_lmax2x2(S2_m), min=1e-9))
            cx, cy = mu2d_m[:, 0], mu2d_m[:, 1]
            fin = torch.isfinite(cx) & torch.isfinite(cy) & torch.isfinite(rr)
            cin = (v_m & fin & (cx >= 0) & (cx < Wd) & (cy >= 0) & (cy < Hd))
            ov = (v_m & fin & (cx > -rr) & (cx < Wd + rr)
                  & (cy > -rr) & (cy < Hd + rr))
            if want_report:
                cin_out[sl] = cin
                ov_out[sl] = ov
            ok = ov & v_i & v_j & (rr > 1e-3)
            ok_out[sl] = ok
            if want_sums:
                v3_out[sl] = v_i & v_j & v_m
                deti_out[sl] = _det2(S2_i)
                detj_out[sl] = _det2(S2_j)
            if sums_only:
                # decoupled SH accumulation: only the per-view weights are needed,
                # so skip the 81-sample grid entirely (plan 11.3 / shfix coupling)
                continue

            bv = torch.nonzero(ok, as_tuple=False).flatten()
            if bv.numel() == 0:
                continue
            rrk = rr[bv]
            rr_out[sl][bv] = rrk
            px = torch.minimum(torch.maximum(cx[bv, None] + gx[None, :] * rrk[:, None],
                                             torch.zeros((), dtype=dtype, device=dev)),
                               (Wd - 1.0)[bv, None])
            py = torch.minimum(torch.maximum(cy[bv, None] + gy[None, :] * rrk[:, None],
                                             torch.zeros((), dtype=dtype, device=dev)),
                               (Hd - 1.0)[bv, None])
            Gi = _gauss(mu2d_i[bv], S2_i[bv], px, py)
            Gj = _gauss(mu2d_j[bv], S2_j[bv], px, py)
            Gm = _gauss(mu2d_m[bv], S2_m[bv], px, py)
            Tp = 1.0 - (1.0 - ai[bv][:, None] * Gi) * (1.0 - aj[bv][:, None] * Gj)
            d[sl][bv] = (Gm * Gm).sum(1)
            n[sl][bv] = (Gm * Tp).sum(1)
            if want_t2:
                t2[sl][bv] = (Tp * Tp).sum(1)
            if timers is not None:
                timers["pc_grid"] = (timers.get("pc_grid", 0.0)
                                     + _time.perf_counter() - _t0)
        out = dict(d=d, n=n, ok=ok_out, a0=a0_out, rr=rr_out)
        if want_t2:
            out["t2"] = t2
        if want_report:
            out.update(cin=cin_out, ov=ov_out)
        if want_sums:
            out.update(v3=v3_out, det_i=deti_out, det_j=detj_out)
        return out


# --------------------------------------------------------------------------- #
def opacity_from_stats(d, n, pid, P, a0):
    """Least-squares opacity per pair from per-view normal-equation sums."""
    dev = d.device
    num = torch.zeros(P, dtype=d.dtype, device=dev).index_add_(0, pid, n)
    den = torch.zeros(P, dtype=d.dtype, device=dev).index_add_(0, pid, d)
    cnt = torch.zeros(P, dtype=torch.int32, device=dev).index_add_(
        0, pid, torch.ones_like(pid, dtype=torch.int32))
    am = torch.where(cnt > 0, num / torch.clamp(den, min=1e-9), a0)
    return am, cnt, num, den


@torch.no_grad()
def opacity_o2cal_gpu_pool(state, comp, pairs, sel, pool, grid=9, sigma_cut=2.0,
                           calibrate=False, batch=32768, stats=None, timers=None,
                           dtype=torch.float64, sh_sums=None, chunk=65536,
                           sh_ref_sel=None):
    """In-pipeline O2cal with per-pair camera sets (`sel`, (P,B), -1 = unused).

    `sh_sums` (dict with keys Si/Sj, float64 (P,) zeros) mirrors
    `o2gpu.opacity_o2cal_gpu`: it accumulates sum_v op/sqrt(det) over the views
    where all three gaussians project.

    `sh_ref_sel`, when given, is a *second* (P,R) selection used only for that
    accumulation.  This decouples the DC SH correction from the camera ablation:
    otherwise changing the fit views silently changes the colour correction too
    (measured on garden: the SH term alone moves by 0.22 dB between fast8 and the
    pair-conditioned set).
    """
    import time
    P = int(pairs.shape[0])
    B = int(sel.shape[1])
    dev = pool.dev
    a_i_all = state["op"][pairs[:, 0]]
    a_j_all = state["op"][pairs[:, 1]]
    a0 = torch.clamp(a_i_all + a_j_all - a_i_all * a_j_all, 0.0, 1.0)
    num = torch.zeros(P, dtype=dtype, device=dev)
    den = torch.zeros(P, dtype=dtype, device=dev)
    cnt = torch.zeros(P, dtype=torch.int32, device=dev)
    clock = [0.0, 0.0, 0.0]     # bill the patch work (selection happens before)
    if sh_sums is not None and sh_ref_sel is not None:
        # ---- decoupled SH accumulation over the reference view set ----------
        t0 = time.perf_counter()
        R = int(sh_ref_sel.shape[1])
        for b0 in range(0, P, batch):
            b1 = min(P, b0 + batch)
            s = sh_ref_sel[b0:b1].to(torch.int64)
            okc = s >= 0
            if not bool(okc.any()):
                continue
            rows = torch.nonzero(okc.reshape(-1), as_tuple=False).flatten()
            pid_r = (rows // R) + b0
            cid_r = s.reshape(-1)[rows]
            st_r = pool.patch_stats(state, comp, pairs, pid_r, cid_r,
                                    sigma_cut=sigma_cut, dtype=dtype,
                                    want_sums=True, want_t2=False,
                                    want_report=False, sums_only=True)
            v3 = st_r["v3"].to(dtype)
            sh_sums["Si"].index_add_(
                0, pid_r, (a_i_all[pid_r].to(dtype)
                           / torch.clamp(st_r["det_i"], min=1e-12).sqrt()) * v3)
            sh_sums["Sj"].index_add_(
                0, pid_r, (a_j_all[pid_r].to(dtype)
                           / torch.clamp(st_r["det_j"], min=1e-12).sqrt()) * v3)
        clock[2] += time.perf_counter() - t0
        if timers is not None:
            timers["o2_shref"] = timers.get("o2_shref", 0.0) + clock[2]
    for b0 in range(0, P, batch):
        b1 = min(P, b0 + batch)
        t0 = time.perf_counter()
        s = sel[b0:b1].to(torch.int64)                       # (b,B)
        okc = s >= 0
        if not bool(okc.any()):
            continue
        rows = torch.nonzero(okc.reshape(-1), as_tuple=False).flatten()
        pid = (rows // B) + b0
        cid = s.reshape(-1)[rows]
        st = pool.patch_stats(state, comp, pairs, pid, cid, grid=grid,
                             sigma_cut=sigma_cut, dtype=dtype,
                             want_sums=(sh_sums is not None), timers=timers,
                             want_t2=False, want_report=False, chunk=chunk)
        clock[0] += time.perf_counter() - t0
        t0 = time.perf_counter()
        if sh_sums is not None and sh_ref_sel is None:
            v3 = st["v3"].to(dtype)
            ai = a_i_all[pid].to(dtype)
            aj = a_j_all[pid].to(dtype)
            sh_sums["Si"].index_add_(
                0, pid, (ai / torch.clamp(st["det_i"], min=1e-12).sqrt()) * v3)
            sh_sums["Sj"].index_add_(
                0, pid, (aj / torch.clamp(st["det_j"], min=1e-12).sqrt()) * v3)
        keep = st["ok"]
        if not bool(keep.any()):
            continue
        pk = pid[keep]
        num.index_add_(0, pk, st["n"][keep])
        den.index_add_(0, pk, st["d"][keep])
        cnt.index_add_(0, pk, torch.ones_like(pk, dtype=torch.int32))
        clock[1] += time.perf_counter() - t0
    am = torch.where(cnt > 0, num / torch.clamp(den, min=1e-9), a0)
    am = torch.where(torch.isfinite(am), am, a0)
    if stats is not None:
        stats["clip_hi"] = stats.get("clip_hi", 0) + int((am > 1).sum())
        stats["clip_lo"] = stats.get("clip_lo", 0) + int((am < 0).sum())
        stats["o2_fallback"] = stats.get("o2_fallback", 0) + int((cnt == 0).sum())
        stats["o2_views"] = stats.get("o2_views", 0) + int(cnt.sum())
        stats["o2_pairs"] = stats.get("o2_pairs", 0) + P
        stats["am_mean_pre_cal"] = float(am.mean())
    if calibrate:
        f = a0.mean() / torch.clamp(am.mean(), min=1e-9)
        if stats is not None:
            stats["last_f"] = float(f)
            stats["o2_calib"] = stats.get("o2_calib", 0.0) + float(f)
            stats["o2_calib_n"] = stats.get("o2_calib_n", 0) + 1
        am = am * f
    return torch.clamp(am, 0.0, 1.0).to(torch.float32)
