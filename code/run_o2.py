#!/usr/bin/env python3
"""GPU backend for the official NanoGS training-free simplifier.

Algorithm, parameter semantics (target count from the original N0, opacity
pre-pruning rule, union of directed kNN -> unique undirected edges, per-pass
graph refresh, per-pass merge budget, MPMM and SH weighting) are exactly those
of the public code.  Only the execution backend of each operator is switched:

    --cost-device   cpu|gpu     edge costs      (nanogs.utils.cost.full_cost_pairs)
    --merge-device  cpu|gpu     MPMM merge      (nanogs.utils.merge.merge_pairs)
    --edges-device  cpu|gpu     graph dedup     (np.unique(axis=0) vs torch.unique)
    --match         exact|par   disjoint select (stable greedy vs parallel)
    --fast-io       0|1         vectorised PLY read/write (byte-identical)

`--cost-device cpu --merge-device cpu --edges-device cpu --match exact --fast-io 0`
is bit-for-bit the public pipeline (verified by sha256 of the output PLY).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

from nanogs.utils.params import RunParams, CostParams
from nanogs.utils.ply_utils import read_ply, store_ply
from nanogs.utils.cost import full_cost_pairs
from nanogs.utils.merge import merge_pairs
from nanogs.simplification import knn_indices as knn_indices_official, knn_undirected_edges


def knn_indices_tuned(mu, k, leafsize=32, variant=2, eps=0.0):
    """cKDTree with balanced_tree/compact_nodes off.

    Measured on truck pass 1 (1.04M points, k=16); every variant gives the
    **identical** undirected edge set (10,334,299 edges, key-for-key):

      variant 0 (official)                build 0.596  query 1.183  total 1.779
      variant 1 (nobal+leaf32)            build 0.303  query 0.991  total 1.295
      variant 2 (+ float64, copy_data=F)  build 0.248  query 0.447  total 0.696

    `copy_data=False` alone halves the query: scipy then searches the caller's
    contiguous array instead of its own rearranged copy.  cKDTree stores doubles,
    so converting once up front is also cheaper than converting per comparison.
    """
    from scipy.spatial import cKDTree
    if variant >= 2:
        mu = np.ascontiguousarray(mu, dtype=np.float64)
    tree = cKDTree(mu, balanced_tree=False, compact_nodes=False,
                   leafsize=leafsize, copy_data=False)
    _, idx = tree.query(mu, k=k + 1, workers=-1, eps=eps)
    return idx[:, 1:]


def knn_indices(mu, k, tuned=True, variant=1, eps=0.0):
    if not tuned:
        return knn_indices_official(mu, k)
    return knn_indices_tuned(mu, k, variant=variant, eps=eps)

import gpulib as G
from fastio import read_ply_fast, store_ply_fast

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Prof:
    def __init__(self, dev):
        self.dev = dev
        self.tot = {}
        self.cur = None

    def stage(self, name):
        prof = self

        @contextmanager
        def _ctx():
            if prof.dev is not None:
                torch.cuda.synchronize(prof.dev)
            t0 = time.perf_counter()
            try:
                yield
            finally:
                if prof.dev is not None:
                    torch.cuda.synchronize(prof.dev)
                dt = time.perf_counter() - t0
                prof.tot[name] = prof.tot.get(name, 0.0) + dt
                if prof.cur is not None:
                    prof.cur[name] = prof.cur.get(name, 0.0) + dt

        return _ctx()


def cpu_undirected_edges_fast(nbr):
    """Same edge set as nanogs.simplification.knn_undirected_edges, but dedups
    on packed int64 keys instead of np.unique(axis=0)."""
    N, k = nbr.shape
    ii = np.repeat(np.arange(N, dtype=np.int64), k)
    jj = nbr.reshape(-1).astype(np.int64)
    u = np.minimum(ii, jj)
    v = np.maximum(ii, jj)
    mask = u != v
    key = np.unique(u[mask] * np.int64(N) + v[mask])
    uu = key // N
    vv = key % N
    return np.stack([uu, vv], axis=1).astype(np.int32)


def cpu_edge_costs(edges, mu, sc, q, op, sh, cp, block_edges):
    M = edges.shape[0]
    w = np.empty((M,), dtype=np.float32)
    for e0 in range(0, M, block_edges):
        e1 = min(M, e0 + block_edges)
        uv = edges[e0:e1]
        u, v = uv[:, 0], uv[:, 1]
        mu_u, sc_u, q_u, op_u = mu[u], sc[u], q[u], op[u]
        mu_v, sc_v, q_v, op_v = mu[v], sc[v], q[v], op[v]
        if sh.shape[1]:
            sh_u, sh_v = sh[u], sh[v]
        else:
            sh_u = sh[u]
            sh_v = sh_u
        w[e0:e1] = full_cost_pairs(mu_u, sc_u, q_u, op_u, sh_u,
                                   mu_v, sc_v, q_v, op_v, sh_v, cp).astype(np.float32)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-r", "--ratio", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--select-mode", choices=["cost", "random", "identity", "cost-nodedup"],
                    default="cost",
                    help="cost = score + sort (default); random/identity skip both "
                         "and take the KNN edges in that order")
    ap.add_argument("--greedy-mode", choices=["serial", "blocked"], default="serial",
                    help="blocked = spatial-partition matching (parallelisable, "
                         "cross-block edges resolved serially at the end)")
    ap.add_argument("--greedy-blocks", type=int, default=8)
    ap.add_argument("--select-topn", type=int, default=0,
                    help="only the N cheapest edges enter the greedy (0 = all); "
                         "a coarse filter that keeps the gate but shortens the scan")
    ap.add_argument("--knn-eps", type=float, default=0.0,
                    help="eps-approximate cKDTree query (0 = exact).  0.2 is ~18%% "
                         "faster with an identical accepted-pair tail")
    ap.add_argument("--merge_cap", type=float, default=0.5)
    ap.add_argument("--opacity_threshold", type=float, default=0.1)
    ap.add_argument("--lam_geo", type=float, default=1.0)
    ap.add_argument("--lam_sh", type=float, default=1.0)
    ap.add_argument("--n-mc", type=int, default=1)
    ap.add_argument("--scorer", choices=["kl", "scale_norm", "maha_fast"], default="kl",
                    help="edge score used by the greedy; kl = native NanoGS MC-KL")
    ap.add_argument("--sh-mode", choices=["s0", "dc", "l1"], default="s0",
                    help="s0 = original weighted-mean SH merge; dc/l1 = contribution-"
                         "weighted residual correction of the merged SH (plan S3a/S3b)")
    ap.add_argument("--sh-lam", type=float, default=1e-2,
                    help="ridge on the SH residual normal equations")
    ap.add_argument("--cluster", type=int, default=2, choices=[2, 4],
                    help="2 = pairwise greedy (baseline); 4 = one-shot 4-source "
                         "cluster merge (plan route B), straight to the target")
    ap.add_argument("--o2-every", type=int, default=1,
                    help="run O2cal every k-th merge level (k>1 = cascade mode A: "
                         "intermediate levels keep the O0 opacity)")
    ap.add_argument("--cost-axis", type=float, default=0.0,
                    help="penalty on (1-|cos|) between the two major axes: expensive "
                         "for flat splats merged across their orientation")
    ap.add_argument("--cost-normal", type=float, default=0.0,
                    help="penalty on (d . n_minor)^2: a merge whose connecting line "
                         "runs along the local normal bridges a surface/edge")
    ap.add_argument("--cost-random", type=int, default=0,
                    help="replace the edge cost with random noise: a floor that "
                         "isolates how much the ORDERING matters")
    ap.add_argument("--cost-vis", type=float, default=0.0,
                    help="cheap visibility term: project the gaussians into the fit "
                         "views and penalise merging the ones actually seen")
    ap.add_argument("--cost-e4", type=float, default=0.0,
                    help="E4 guard: penalise merging two large gaussians that sit "
                         "in a LOW-detail area (over-smooth patch risk)")
    ap.add_argument("--detail-nv", type=float, default=0.0,
                    help="detail-aware merge risk: lambda on "
                         "max(normal_var_i, normal_var_j) * (1-|n_i.n_j|)")
    ap.add_argument("--cost-cg-view", type=int, default=0, choices=[0, 4, 8],
                    help="E1: replace the DC colour distance with a multi-view SH "
                         "descriptor evaluated at 4 or 8 fixed directions")
    ap.add_argument("--cost-cg", type=float, default=0.0,
                    help="gated colour veto: colour difference weighted by a "
                         "coplanarity gate, so it only refines near-coplanar pairs")
    ap.add_argument("--cost-cg-tau", type=float, default=0.5,
                    help="gate width in local-spacing units (default 0.5)")
    ap.add_argument("--cost-plane", type=float, default=0.0,
                    help="penalty on |(d . n_u)| / local_spacing: distance of v from "
                         "u's local plane in LOCAL-SPACING units")
    ap.add_argument("--cost-sratio", type=float, default=0.0,
                    help="penalty on |log(s_major_u / s_major_v)|: a big splat "
                         "swallowing a small one")
    ap.add_argument("--edge-risk", type=int, default=0, choices=[0, 1],
                    help="1: cache a per-gaussian edge-score and use it in the cost")
    ap.add_argument("--edge-op", choices=["sobel", "laplace"], default="sobel")
    ap.add_argument("--edge-lambda", type=float, default=0.0,
                    help="weight of the edge-risk penalty (base cost is normalised "
                         "by its mean first, so 0.1-1 is a sensible range)")
    ap.add_argument("--sh-risk-lambda", type=float, default=0.0,
                    help="penalty on ||f_dc_i - f_dc_j|| in the edge cost: steer the "
                         "greedy away from colour-divergent merges")
    ap.add_argument("--sh-sum", type=int, default=1, choices=[0, 1],
                    help="1: O2cal accumulates sum_v w_i / sum_v w_j inside its own "
                         "loop and the DC correction becomes O(P) (no extra pass)")
    ap.add_argument("--sh-min-diff", type=float, default=0.0,
                    help="skip merges whose source DC colours differ by less than this")
    ap.add_argument("--sh-cache", type=int, default=1, choices=[0, 1],
                    help="1: reuse the 2-D covariance determinants O2cal already "
                         "computed instead of projecting i/j/m again")
    ap.add_argument("--sh-freeze-score", type=int, default=1, choices=[0, 1],
                    help="1: the scorer keeps the uncorrected (S0) SH lineage, so the "
                         "merge sequence stays identical to S0 (plan 3.2)")
    ap.add_argument("--sh-clip", type=float, default=0.30,
                    help="max |delta| per SH coefficient")
    ap.add_argument("--jacobi", type=int, default=0, choices=[0, 1],
                    help="1: hand-written fixed-sweep 3x3 Jacobi eigh + explicit det, so "
                         "the process never initialises cuSOLVER either")
    ap.add_argument("--no-blas", type=int, default=0, choices=[0, 1],
                    help="1: use explicit multiply+sum instead of matmul/einsum on the "
                         "scale_norm+O2cal path so the process never initialises cuBLAS")
    ap.add_argument("--knn-variant", type=int, default=1, choices=[1, 2],
                    help="1: nobal+leaf32; 2: also float64 + copy_data=False (1.86x, "
                         "identical edge set)")
    ap.add_argument("--knn-tuned", type=int, default=1, choices=[0, 1],
                    help="1 = cKDTree(balanced_tree=False, compact_nodes=False, leafsize=32);"
                         " identical edge set, ~1.36x faster build+query")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-edges", type=int, default=100_000)
    ap.add_argument("--cost-device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--merge-device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--edges-device", choices=["cpu", "gpu", "cpu_fast"], default="gpu")
    ap.add_argument("--match", choices=["exact", "par"], default="exact")
    ap.add_argument("--fast-io", type=int, default=1, choices=[0, 1])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--dump-o2-pass", type=int, default=0,
                    help="dump the pre-merge state + pairs of pass K for O2 diagnosis")
    ap.add_argument("--o2-verbose-pass", type=int, default=0,
                    help="compare in-process CPU vs GPU O2 on pass K")
    ap.add_argument("--dump-pairs", action="store_true",
                    help="save the pass-1 accepted pairs for scorer-overlap analysis")
    ap.add_argument("--max-passes", type=int, default=0)
    # ---- O2cal ----------------------------------------------------------
    ap.add_argument("--opacity-mode", choices=["O0", "O2", "O2cal"], default="O0")
    ap.add_argument("--o2-device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--o2-scene", default=None, help="COLMAP scene dir for the fit views")
    ap.add_argument("--o2-views", type=int, default=8)
    ap.add_argument("--o2-grid", type=int, default=9)
    ap.add_argument("--o2-sigma-cut", type=float, default=2.0)
    ap.add_argument("--o2-batch", type=int, default=32768)
    ap.add_argument("--o2-cams", choices=["fast", "ref", "virtual", "auto", "cover", "pair",
                                          "auto_virt", "json"],
                    default="auto",
                    help="fast: struct parser (bit-identical); ref: 3DGS colmap_loader; "
                         "virtual: rig synthesised from the point cloud only; "
                         "auto_virt: camera-free 5-step voxel/set-cover rig "
                         "(o2_auto_virt.py, no COLMAP at all); "
                         "json: load an explicit camera list from --o2-cams-json; "
                         "pair: pair-conditioned selection from the whole fit pool (PICS)")
    ap.add_argument("--o2-cams-json", default="",
                    help="path of a camera list (JSON) for --o2-cams json")
    ap.add_argument("--o2-dump-cams", default="",
                    help="write the resolved camera list to this JSON path (round-trip "
                         "verification for any --o2-cams mode)")
    ap.add_argument("--o2-pool", choices=["real", "virt"], default="real",
                    help="candidate pool for --o2-cams pair: the real COLMAP fit "
                         "cameras (default) or the dense virtual rig produced by "
                         "o2_auto_virt (camera-free PICS; per-pair view choice)")
    ap.add_argument("--o2-rstar", type=float, default=5.0,
                    help="T3-B: ideal projected pixel radius for --o2-pick scale_opt")
    ap.add_argument("--o2-view-tau", type=float, default=0.0,
                    help="S4: per-view projection-conditioning weight r^2/(r^2+tau^2) "
                         "inside the O2cal least squares (0 = off; 0.5/1.0 px "
                         "correspond to the sqrt(0.3)=0.55 px EWA floor)")
    ap.add_argument("--o2-view-rmax", type=float, default=40.0,
                    help="S4: hard cap (px) above which a view's weight is zero")
    ap.add_argument("--o2-solver", choices=["grid", "analytic", "eigen"], default="grid",
                    help="O2cal solver: the 9x9 local grid (default), the exact "
                         "continuous L2 closed form, or the per-pair eigen-axis "
                         "closed form (zero camera, zero grid)")
    ap.add_argument("--o2-eigen-mode", choices=["E0", "E1", "E2", "E3", "E4"],
                    default="E1",
                    help="eigen solver variant: E0 = 8 fixed cube dirs, E1 = normal "
                         "(scale-free), E2 = normal + gamma, E3 = 3 principal "
                         "sections, E4 = normal + 35deg tilt cone")
    ap.add_argument("--o2-eigen-gamma", type=float, default=0.0,
                    help="eigen solver: pixel scale gamma = f_px/d_eff. 0 = auto "
                         "(f_px/(c_dist*d_core))")
    ap.add_argument("--o2-eigen-cdist", type=float, default=0.75,
                    help="eigen solver: d_eff as a fraction of the core AABB extent")
    ap.add_argument("--o2-eigen-eps", type=float, default=-1.0,
                    help="eigen solver: EWA constant (0 or 0.3). <0 = auto per mode")
    ap.add_argument("--o2-eigen-wmode", choices=["area", "eq", "det"], default="area",
                    help="eigen solver: cross-view weight")
    ap.add_argument("--o2-eigen-psi", type=float, default=35.0,
                    help="eigen solver E4: tilt angle in degrees")
    ap.add_argument("--o2-eigen-fpx", type=float, default=886.8,
                    help="eigen solver: focal length in pixels (default 512/tan30deg "
                         "for a 1024-wide 60deg view)")
    ap.add_argument("--o2-debug-dump", default="",
                    help="D1: dump per-(pair,view) O2cal intermediates of the first "
                         "batch to this .pt path (diagnostics only)")
    ap.add_argument("--o2-cf-fix", action="store_true",
                    help="BROKEN, do not use (clamp+mask combination blows alpha up "
                         "to 1e5). The core NaN guard alone is the working fix.")
    ap.add_argument("--o2-cf-safe", action="store_true",
                    help="BROKEN, do not use (see --o2-cf-fix)")
    ap.add_argument("--o2-cf-c0", type=float, default=0.15,
                    help="analytic ecc_floor: positive floor c0 (Fix 3)")
    ap.add_argument("--o2-clean-mask", action="store_true",
                    help="M2/M3: replace the sigma_cut footprint mask with a pure "
                         "near-plane + centre-in-frame + non-degenerate test")
    ap.add_argument("--o2-cf-bound", action="store_true",
                    help="M2/M3: drop views whose cross ratios leave the rigorous "
                         "bound [0, (2+beta)/(1+beta)]")
    ap.add_argument("--o2-fit-dc", action="store_true",
                    help="also refit the DC (RGB) SH coefficients in closed form, "
                         "with the transmittance-weighted least-squares solution")
    ap.add_argument("--o2-cf-beta", type=float, default=0.0,
                    help="analytic solver: core-focus exponent beta (0 = unweighted)")
    ap.add_argument("--o2-cf-agg", choices=["eq", "area", "ecc", "ecc_floor"],
                    default="eq",
                    help="analytic solver: per-view aggregation (eq = plain mean, "
                         "area = pixel area, ecc = 1/kappa, the grid's implicit weight)")
    ap.add_argument("--o2-pick", choices=["sup", "info", "strat", "repair", "fill", "pref", "guard", "track", "fixed", "all", "dir", "scale_opt"],
                    default="repair",
                    help="--o2-cams pair selection rule: sup=best support, "
                         "info=largest in-image footprint, strat=one per trajectory bin, "
                         "repair=strat but keep the fast view when it has support, "
                         "fill=stratified fill until the budget is used, "
                         "pref=keep the global cover set and repair its empty slots, guard=fast whenever it has any support (repair only the fallbacks), "
                         "track=support gate + COLMAP track redundancy penalty, "
                         "scale_opt=per-pair projected-radius matching to --o2-rstar")
    ap.add_argument("--o2-qmin", type=float, default=0.05,
                    help="minimum in-image footprint fraction for a view to count as support")
    ap.add_argument("--o2-nbins", type=int, default=0,
                    help="trajectory bins for the stratified rules (0 -> B, or 8 for fill)")
    ap.add_argument("--o2-track-lam", type=float, default=1.0,
                    help="weight of the COLMAP track-redundancy penalty "
                         "(only for --o2-pick track)")
    ap.add_argument("--o2-sh-ref", choices=["fit", "fast"], default="fit",
                    help="which view set the DC SH correction accumulates over: "
                         "fit = the pair's own selected views (historical), "
                         "fast = the fixed fast-B reference set, so the camera "
                         "ablation no longer moves the colour correction")
    ap.add_argument("--o2-fixed-ids", default="",
                    help="comma separated camera ids, used by --o2-pick fixed to "
                         "feed a *global* set through the PICS kernel (parity check "
                         "against --o2-cams cover / fast)")
    ap.add_argument("--o2-rig", choices=["orbit", "inner", "two_orbit"], default="orbit")
    ap.add_argument("--o2-fov", type=float, default=55.0)
    ap.add_argument("--o2-res-w", type=int, default=1024)
    ap.add_argument("--o2-res-h", type=int, default=768)
    # ---- camera-free auto_virt rig (camera_free_o2cal.md) ----------------
    ap.add_argument("--av-res", type=int, default=64, help="coarse voxel grid R")
    ap.add_argument("--av-npos", type=int, default=48, help="candidate positions (FPS)")
    ap.add_argument("--av-ndirs", type=int, default=6,
                    help="per-position headings: 6 axis / 8 cube corners for the "
                         "axis mode; any >=3 for the gravity ring (density sweep)")
    ap.add_argument("--av-fov", type=float, default=60.0, help="virtual FOV_x in degrees")
    ap.add_argument("--av-cov-k", type=int, default=3, help="k of the k-coverage phi cap")
    ap.add_argument("--av-dsafe", type=int, default=2, help="free-voxel clearance (voxels)")
    ap.add_argument("--av-dfar", type=int, default=10, help="outer shell band (voxels)")
    ap.add_argument("--av-rtarget", type=float, default=4.0,
                    help="target projected radius of the median gaussian (px)")
    ap.add_argument("--av-dlo", type=float, default=0.20,
                    help="lower clip of d_target as a fraction of L_max "
                         "(spec 0.08 pins every scene to the floor; 0.20 matches "
                         "the real camera distance on truck/garden/bicycle/room)")
    ap.add_argument("--av-dhi", type=float, default=0.65,
                    help="upper clip of d_target as a fraction of L_max")
    ap.add_argument("--av-cav6", type=int, default=20,
                    help="min fully-enclosed voxels (6/6 rays) to call it indoor")
    # ---- v2 stage-1 modules (see camera_free_o2cal_autovirt_v2.md) -------
    ap.add_argument("--av-core-aabb", type=float, default=0.0,
                    help="M1: keep the p_keep smallest in-plane-scale points for "
                         "the core AABB (0.80 / 0.70); 0 = v1 behaviour")
    ap.add_argument("--av-dmode", choices=["sgeom", "smid"], default="sgeom",
                    help="M2a: d_target from the geometric-mean scale (v1) or from "
                         "the in-plane median scale")
    ap.add_argument("--av-rref", type=float, default=None,
                    help="M2a: target projected radius in px (default 4.0 for "
                         "sgeom, 1.5 for smid)")
    ap.add_argument("--av-soft-scale", type=float, default=0.0,
                    help="M2b: sigma_r of the continuous log-Gaussian scale "
                         "kernel; >0 replaces the hard scale gate")
    ap.add_argument("--av-dir-mode", choices=["axis", "gravity", "vote"],
                    default="axis",
                    help="M3/M5b: candidate headings from the 6 world axes (v1), a "
                         "zero-roll yaw ring around the extracted gravity axis, or "
                         "the voted view direction (+-35 deg) of M5")
    ap.add_argument("--av-pitches", default="0",
                    help="M3: comma separated pitch degrees, e.g. \"0,-15\"")
    ap.add_argument("--av-grav-lower", type=float, default=0.0,
                    help="B2: re-vote the gravity axis on this lower fraction of "
                         "the flat gaussians (e.g. 0.35) to lock the floor")
    ap.add_argument("--av-cos-weight", type=float, default=0.0,
                    help="M4: gamma of the front-facing cosine coverage; >0 switches "
                         "the greedy objective to saturating continuous coverage")
    ap.add_argument("--av-cmax", type=float, default=2.0,
                    help="M4: coverage saturation cap C_max")
    ap.add_argument("--av-pos-mode", choices=["frontier", "vote", "pca_traj", "multishell",
                                              "rings", "mixed", "dual_domain", "unified"],
                    default="frontier",
                    help="M5/M6: candidate positions from the v1 frontier, from "
                         "surface-normal back-projection voting, or from a PCA "
                         "compact/corridor trajectory")
    ap.add_argument("--av-eta0", type=float, default=1.6,
                    help="M6: compact vs corridor threshold on sqrt(l1/l2)")
    ap.add_argument("--av-ncorr", type=int, default=16,
                    help="M6: number of corridor cross-sections")
    ap.add_argument("--av-dir-bins", type=int, default=0,
                    help="M7a: >0 enables the azimuth quadrant coverage objective")
    ap.add_argument("--av-aniso-weight", type=float, default=0.0,
                    help="M7b: beta of the anisotropy sensitivity voxel weight")
    ap.add_argument("--av-pca-corridor-only", action="store_true",
                    help="M6c: only use the PCA corridor branch; compact (eta<eta0) "
                         "scenes keep the v1 frontier positions")
    ap.add_argument("--av-weight-mode", choices=["log", "n", "mip"], default="log",
                    help="D3 density-weight probe: log1p(W_occ) (v1), N_v^0.5, or "
                         "rho_mip^0.35")
    ap.add_argument("--av-gate-rel", type=float, default=0.0,
                    help="audit fix: centre the visibility gate window on the actual "
                         "camera-to-scene distance instead of d_target (1.0 = on)")
    ap.add_argument("--av-cover-target", choices=["voxel", "gauss"], default="voxel",
                    help="greedy coverage objective: surface voxels (v1) or the "
                         "actual gaussians' visible support (audit §12 fix)")
    ap.add_argument("--av-gauss-probe", type=int, default=50000,
                    help="number of probe gaussians for --av-cover-target gauss")
    ap.add_argument("--av-cover-gate", action="store_true",
                    help="apply the kernel's r_px / depth window inside the "
                         "gaussian-support objective")
    ap.add_argument("--av-gauss-rstar", type=float, default=0.0,
                    help="T1-B: ideal projected pixel radius r* (>0 enables the "
                         "smooth quality objective instead of binary visibility)")
    ap.add_argument("--av-gauss-rsigma", type=float, default=1.0,
                    help="T1-B: log-radius tolerance sigma_r")
    ap.add_argument("--av-gauss-agg", choices=["max", "cap"], default="max",
                    help="T1-B: facility-location (max) or saturating sum (cap)")
    ap.add_argument("--av-gauss-cap", type=float, default=3.0,
                    help="T1-B: saturation cap for --av-gauss-agg cap")
    ap.add_argument("--av-phi", default="",
                    help="T4-A: explicit coverage curve phi, e.g. '0,1' (strict "
                         "1-coverage), '0,1,1.15,1.2' (soft)")
    ap.add_argument("--av-shell-ratios", default="0.40,0.35,0.25",
                    help="T1-A: inner/mid/outer shell slot fractions for "
                         "--av-pos-mode multishell")
    ap.add_argument("--av-ring-radii", default="1.0,2.0,3.5",
                    help="T5-A: voxel-free ring radii (x R_core) for "
                         "--av-pos-mode rings; small values give the T5-B "
                         "compact indoor inner ring (e.g. 0.25,0.45,0.65)")
    ap.add_argument("--av-diverse-quota", action="store_true",
                    help="T4-B: enforce a near/mid/far distance-band quota on the "
                         "8 selected cameras")
    ap.add_argument("--av-pos-b", type=int, default=24,
                    help="S2 dual_domain: number of Domain-B (full-AABB) positions")
    ap.add_argument("--av-unified-frac", type=float, default=0.5,
                    help="S3 unified: share of interior-pocket positions (rest from "
                         "the exterior frontier)")
    ap.add_argument("--av-normal-mode", choices=["kstep", "n2"], default="kstep",
                    help="T2-A: normal orientation for the vote module: 'kstep' "
                         "(N0, v1 local stepping) or 'n2' (structure tensor + "
                         "global focus/cavity orientation)")
    ap.add_argument("--av-cavity-frac", type=float, default=0.75,
                    help="share of indoor candidates taken from the interior cavity")
    ap.add_argument("--av-no-zbuf", action="store_true", help="disable soft z-buffer occlusion")
    ap.add_argument("--av-no-scale-gate", action="store_true",
                    help="disable the +0.3I projection-scale gate")
    ap.add_argument("--av-normal-gate", action="store_true", help="enable grazing-angle gate")
    ap.add_argument("--av-dump-json", default="",
                    help="write the rig + stats to this json path (Phase 1 diagnostics)")
    args = ap.parse_args()

    if args.opacity_mode != "O0" and not args.o2_scene and \
            args.o2_cams not in ("virtual", "auto_virt", "json"):
        raise SystemExit("--opacity-mode O2/O2cal requires --o2-scene "
                         "(except for --o2-cams virtual / auto_virt)")
    if args.o2_cams == "pair" and args.opacity_mode != "O0":
        if args.o2_device != "gpu" or args.merge_device != "gpu":
            raise SystemExit("--o2-cams pair requires --o2-device gpu --merge-device gpu")
        if args.edge_risk:
            raise SystemExit("--o2-cams pair does not support --edge-risk "
                             "(the edge maps assume one fixed view set)")
        if args.cluster > 2:
            raise SystemExit("--o2-cams pair does not support --cluster")
        if args.sh_mode != "s0" and not args.sh_sum:
            raise SystemExit("--o2-cams pair requires --sh-sum 1 (the per-view SH "
                             "cache assumes one fixed view set)")
    calibrate = args.opacity_mode == "O2cal"

    use_gpu = "gpu" in (args.cost_device, args.merge_device, args.edges_device)
    dev = None
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        torch.cuda.set_device(args.gpu)
        dev = torch.device(f"cuda:{args.gpu}")
        torch.cuda.reset_peak_memory_stats(dev)

    prof = Prof(dev)
    if args.match == "par":
        raise SystemExit("--match par is not implemented yet (exact is bit-faithful)")
    merge_cap = max(0.01, min(0.5, float(args.merge_cap)))
    cp = CostParams(lam_geo=args.lam_geo, lam_sh=args.lam_sh)
    read_fn = read_ply_fast if args.fast_io else read_ply
    write_fn = store_ply_fast if args.fast_io else store_ply

    t_start = time.perf_counter()

    with prof.stage("ply_read"):
        hdr, mu, op, sc, q, sh, app_names = read_fn(args.ply)
    N0 = int(mu.shape[0])
    target = max(int(math.ceil(N0 * args.ratio)), 1)

    with prof.stage("prune"):
        thr = min(args.opacity_threshold, float(np.median(op)))
        keep = np.nonzero(op >= thr)[0]
        mu, sc, q, op, sh = mu[keep], sc[keep], q[keep], op[keep], sh[keep]

    p_cap = max(1, int(merge_cap * N0))

    # canonical state: GPU when any operator runs there
    if dev is not None:
        with prof.stage("h2d_model"):
            t_mu = torch.as_tensor(mu, device=dev)
            t_sc = torch.as_tensor(sc, device=dev)
            t_q = torch.as_tensor(q, device=dev)
            t_op = torch.as_tensor(op, device=dev)
            t_sh = torch.as_tensor(sh, device=dev) if sh.shape[1] else sh
        N = int(t_mu.shape[0])
        mu = sc = q = op = None       # the numpy copies are no longer authoritative

    with prof.stage("setup_z"):
        Z = torch.as_tensor(
            np.random.default_rng(args.seed).standard_normal(size=(args.n_mc, 3)).astype(np.float32),
            device=dev)
    with prof.stage("setup_jit"):
        greedy_jit = (G.greedy_blocked_init() if args.greedy_mode == "blocked"
                      else G.greedy_numba_init())

    # One-time warm-up of the CUDA math libraries used by the pass loop.
    # Without it the *first* batched matmul / eigh / det in the process pays
    # ~0.7-1.1 s of cuBLAS+cuSOLVER handle creation, and that cost lands on
    # whichever stage calls it first.  For `scale_norm` the first matmul happens
    # inside MPMM, so the scorer appeared to carry 0.75 s of hidden downstream
    # cost that was really a cold-start artefact.
    with prof.stage("setup_cuda_warm"):
        if dev is not None:
            with torch.no_grad():
                _w = torch.ones(8, 3, 3, device=dev)
                if not args.no_blas:
                    # cuBLAS handle (~0.7 s once) 闂?only needed when the pipeline
                    # actually uses matmul/einsum
                    torch.matmul(_w, _w.transpose(1, 2))
                    _v = torch.ones(8, 3, 4, device=dev)
                    torch.bmm(_v, _v.transpose(1, 2))
                if not args.jacobi:
                    # cuSOLVER (MPMM's eigh) 闂?skipped when the Jacobi path is used
                    torch.linalg.eigh(0.5 * (_w + _w.transpose(1, 2)))
                    torch.linalg.det(_w)
            torch.cuda.synchronize(dev)

    t_sh_s0 = None
    t_e = None
    if (args.sh_mode != "s0" and args.sh_freeze_score and dev is not None
            and t_sh is not None and t_sh.shape[1]):
        # parallel S0 lineage, used only by the scorer's SH term
        t_sh_s0 = t_sh.clone()

    if args.scorer == "maha_fast" and dev is not None:
        # one-time warm-up of the batched float64 3x3 GEMM (cuBLAS handle +
        # kernel selection costs ~1.1 s on the first full-size call)
        with torch.no_grad():
            G.precompute_scorer(t_mu[:4096], t_sc[:4096], t_q[:4096], need_sig=True,
                                noblas=bool(args.no_blas))

    # ---- O2cal setup (cameras loaded once, uploaded once) ----------------
    o2cams = None
    o2pool = None
    o2fast_ids = None
    o2pref_ids = None
    o2shref = None
    o2stats = dict(clip_hi=0, clip_lo=0, o2_fallback=0, o2_views=0)
    if args.opacity_mode != "O0":
        with prof.stage("o2_cams"):
            from o2gpu import O2Cameras
            _o2_cams = args.o2_cams
            if _o2_cams == "auto":
                # SfM data is the normal case; a bare PLY must not fail here.
                _sp0 = os.path.join(args.o2_scene or "", "sparse", "0", "images.bin")
                _o2_cams = "fast" if os.path.exists(_sp0) else "virtual"
                print(f"o2-cams auto -> {_o2_cams} "
                      f"({'found ' + _sp0 if _o2_cams == 'fast' else 'no sparse/0, using a synthetic rig'})",
                      flush=True)
            if _o2_cams == "cover":
                from cams_fast import load_cams_fast, pick_cover_views
                _all = load_cams_fast(args.o2_scene, "fit", 0)
                _sel = pick_cover_views(_all, t_mu.detach().cpu().numpy(),
                                        k=args.o2_views)
                cams = [_all[i] for i in _sel]
                sp = None
                print(f"o2-cams cover: picked {len(cams)} of {len(_all)} fit views "
                      f"by max coverage", flush=True)
            elif _o2_cams == "fast":
                from cams_fast import load_cams_fast
                cams = load_cams_fast(args.o2_scene, "fit", args.o2_views)
                sp = None
            elif _o2_cams == "pair":
                # PICS: the whole fit pool is the candidate set; each merge pair
                # picks its own B views (plan Part IV). `o2cams` still holds the
                # historical fast-B views, which the SH/edge/reporting paths use.
                # NOTE: only one COLMAP parse -- the fast subset is the same
                # even-stride subsample that load_cams_fast(scene,"fit",B) gives,
                # so it is derived from the full list instead of re-read.
                from cams_fast import load_cams_fast
                from pics import CamPool
                if getattr(args, "o2_pool", "real") == "virt":
                    # camera-free PICS: the candidate pool is the dense virtual
                    # rig, every merge pair then picks its own B views from it
                    from o2_auto_virt import extract_auto_virtual_cams
                    _all, _pst = extract_auto_virtual_cams(
                        t_mu, t_sc, t_q, t_op,
                        n_views=int(args.o2_views), grid_res=args.av_res,
                        n_pos=args.av_npos, n_dirs=args.av_ndirs,
                        fov_deg=args.av_fov, W=args.o2_res_w, H=args.o2_res_h,
                        cov_k=args.av_cov_k, use_zbuf=not args.av_no_zbuf,
                        use_scale_gate=not args.av_no_scale_gate,
                        d_safe=args.av_dsafe, d_far=args.av_dfar,
                        r_target=args.av_rtarget, cavity_frac=args.av_cavity_frac,
                        d_lo_frac=args.av_dlo, d_hi_frac=args.av_dhi,
                        cavity_min6=args.av_cav6, noblas=bool(args.no_blas),
                        core_aabb=args.av_core_aabb, dmode=args.av_dmode,
                        r_ref=args.av_rref, soft_scale=args.av_soft_scale,
                        dir_mode=args.av_dir_mode, pitches=args.av_pitches,
                        grav_lower=args.av_grav_lower, cos_weight=args.av_cos_weight,
                        cmax=args.av_cmax, pos_mode=args.av_pos_mode,
                        eta0=args.av_eta0, n_corr=args.av_ncorr,
                        dir_bins=args.av_dir_bins,
                        aniso_weight=args.av_aniso_weight,
                        pca_corridor_only=bool(args.av_pca_corridor_only),
                        weight_mode=args.av_weight_mode, gate_rel=args.av_gate_rel,
                        seed=args.seed, return_stats=True, return_pool=True)
                    _n = len(_all)
                    o2fast_ids = list(_pst["chosen"])
                    o2pool = CamPool(_all, dev)
                    cams = [_all[i] for i in o2fast_ids]
                    sp = None
                    print(f"o2-cams pair(virt): pool {_n} virtual cameras "
                          f"(mode={_pst['mode']}), fast-B {o2fast_ids}", flush=True)
                else:
                    _all = load_cams_fast(args.o2_scene, "fit", 0)
                    _n = len(_all)
                    _k = int(args.o2_views)
                    if 0 < _k < _n:
                        _step = _n / _k
                        o2fast_ids = [int(i * _step) for i in range(_k)]
                    else:
                        o2fast_ids = list(range(_n))
                    o2pool = CamPool(_all, dev)
                    cams = [_all[i] for i in o2fast_ids]
                    sp = None
                if args.o2_pick == "pref":
                    # base set = the historical greedy coverage selection; every
                    # merge pair keeps it where it has support and repairs the
                    # rest (plan 16.2 step 5).
                    if getattr(args, "o2_pool", "real") == "virt":
                        o2pref_ids = list(o2fast_ids)
                    else:
                        from cams_fast import pick_cover_views
                        o2pref_ids = pick_cover_views(_all, t_mu.detach().cpu().numpy(),
                                                      k=args.o2_views)
                    print(f"o2-pick pref: base cover set {o2pref_ids}", flush=True)
                if args.o2_pick == "track":
                    if getattr(args, "o2_pool", "real") != "virt":
                        # photogrammetric redundancy proxy (plan 11.6): COLMAP track
                        # overlap between fit cameras, parsed once and cached.
                        import cams_track
                        _M, _ = cams_track.load_overlap(args.o2_scene, o2pool)
                        o2pool.M = torch.as_tensor(_M, device=dev)
                        del _M
                        print(f"o2-pick track: overlap matrix {tuple(o2pool.M.shape)}, "
                              f"lam {args.o2_track_lam}", flush=True)
                    else:
                        print("o2-pick track: not available for a virtual pool "
                              "(no COLMAP tracks); falling back to 'repair'",
                              flush=True)
                        args.o2_pick = "repair"
                if args.o2_sh_ref == "fast":
                    # decoupled DC SH correction: accumulate over the fixed fast-B
                    # views, never over the per-pair selection
                    o2shref = torch.as_tensor(o2fast_ids, device=dev,
                                              dtype=torch.int32)[None]
                    print(f"o2-sh-ref fast: SH sums use the fast-{len(o2fast_ids)} "
                          f"reference views", flush=True)
                print(f"o2-cams pair: pool {o2pool.V} fit cameras, budget "
                      f"{args.o2_views}, rule {args.o2_pick}, qmin {args.o2_qmin}",
                      flush=True)
            elif _o2_cams == "virtual":
                # rig synthesised from the point cloud: no COLMAP, no images
                from vircams import make_rig
                _mu_np = t_mu.detach().cpu().numpy()
                cams = make_rig(_mu_np, args.o2_rig, args.o2_views,
                                fov=args.o2_fov, W=args.o2_res_w, H=args.o2_res_h)
                sp = None
                print(f"virtual rig '{args.o2_rig}': {len(cams)} views, "
                      f"fov={args.o2_fov} {args.o2_res_w}x{args.o2_res_h}", flush=True)
            elif _o2_cams == "auto_virt":
                # camera-free rig (camera_free_o2cal.md): no COLMAP pose, no images
                import json as _json
                from o2_auto_virt import extract_auto_virtual_cams
                _extra = None
                if args.av_pos_mode == "dual_domain":
                    # S2: Domain B = the *full*-AABB rig (no core shrink); Domain A
                    # is the regular core_aabb run.  Both pools go into one greedy.
                    _kwB = dict(
                        n_views=args.o2_views, grid_res=args.av_res,
                        n_pos=args.av_pos_b, n_dirs=args.av_ndirs,
                        fov_deg=args.av_fov, W=args.o2_res_w, H=args.o2_res_h,
                        cov_k=args.av_cov_k, use_zbuf=not args.av_no_zbuf,
                        use_scale_gate=False, d_safe=args.av_dsafe,
                        d_far=args.av_dfar, r_target=args.av_rtarget,
                        cavity_frac=args.av_cavity_frac, d_lo_frac=0.20,
                        d_hi_frac=args.av_dhi, cavity_min6=args.av_cav6,
                        noblas=bool(args.no_blas), core_aabb=1.0,
                        dmode=args.av_dmode, r_ref=args.av_rref,
                        soft_scale=args.av_soft_scale,
                        dir_mode="gravity", pitches="0",
                        cos_weight=0.0, cmax=args.av_cmax,
                        pos_mode="frontier", dir_bins=0, aniso_weight=0.0,
                        seed=args.seed, return_stats=False, return_pool=True)
                    _extra = extract_auto_virtual_cams(t_mu, t_sc, t_q, t_op, **_kwB)
                    print(f"o2-cams dual_domain: Domain B pool = {len(_extra)} cams",
                          flush=True)
                cams, _avs = extract_auto_virtual_cams(
                    t_mu, t_sc, t_q, t_op,
                    n_views=args.o2_views, grid_res=args.av_res,
                    n_pos=args.av_npos, n_dirs=args.av_ndirs,
                    fov_deg=args.av_fov, W=args.o2_res_w, H=args.o2_res_h,
                    cov_k=args.av_cov_k,
                    use_zbuf=not args.av_no_zbuf,
                    use_scale_gate=not args.av_no_scale_gate,
                    use_normal_gate=args.av_normal_gate,
                    d_safe=args.av_dsafe, d_far=args.av_dfar,
                    r_target=args.av_rtarget, cavity_frac=args.av_cavity_frac,
                    d_lo_frac=args.av_dlo, d_hi_frac=args.av_dhi,
                    cavity_min6=args.av_cav6, noblas=bool(args.no_blas),
                    core_aabb=args.av_core_aabb, dmode=args.av_dmode,
                    r_ref=args.av_rref, soft_scale=args.av_soft_scale,
                    dir_mode=args.av_dir_mode, pitches=args.av_pitches,
                    grav_lower=args.av_grav_lower,
                    cos_weight=args.av_cos_weight, cmax=args.av_cmax,
                    pos_mode=args.av_pos_mode, eta0=args.av_eta0,
                    n_corr=args.av_ncorr, dir_bins=args.av_dir_bins,
                    aniso_weight=args.av_aniso_weight,
                    pca_corridor_only=bool(args.av_pca_corridor_only),
                    weight_mode=args.av_weight_mode,
                    gate_rel=args.av_gate_rel,
                    cover_target=args.av_cover_target,
                    n_gauss_probe=args.av_gauss_probe,
                    cover_gate=bool(args.av_cover_gate),
                    extra_cams=_extra,
                    unified_frac=float(args.av_unified_frac),
                    gauss_rstar=args.av_gauss_rstar,
                    gauss_rsigma=args.av_gauss_rsigma,
                    gauss_agg=args.av_gauss_agg,
                    gauss_cap=args.av_gauss_cap,
                    phi_override=args.av_phi,
                    normal_mode=args.av_normal_mode,
                    shell_ratios=args.av_shell_ratios,
                    ring_radii=args.av_ring_radii,
                    diverse_quota=bool(args.av_diverse_quota),
                    seed=args.seed, return_stats=True)
                sp = None
                print(f"o2-cams auto_virt: {len(cams)} views mode={_avs['mode']} "
                      f"cav={_avs['n_cavity']} surf={_avs['n_surface']} "
                      f"d_target={_avs['d_target']:.4f} "
                      f"Lmax={_avs['Lmax']:.3f} "
                      f"t={_avs['t_total_ms']:.1f}ms "
                      f"[s1={_avs['t_step1_ms']:.1f} s2={_avs['t_step2_ms']:.1f} "
                      f"s3={_avs['t_step3_ms']:.1f} s4={_avs['t_step4_ms']:.1f}]",
                      flush=True)
                if args.av_dump_json:
                    with open(args.av_dump_json, "w") as _fh:
                        _json.dump({"stats": _avs,
                                    "cams": [{"name": c["name"],
                                              "w2v": np.asarray(c["w2v"]).tolist(),
                                              "C": np.asarray(c["C"]).tolist(),
                                              "fx": c["fx"], "fy": c["fy"],
                                              "W": c["W"], "H": c["H"]}
                                             for c in cams]}, _fh, indent=1)
                    print(f"o2-cams auto_virt: rig dumped -> {args.av_dump_json}",
                          flush=True)
            elif _o2_cams == "json":
                # closed-loop verification: replay an explicit camera list
                if not args.o2_cams_json:
                    raise SystemExit("--o2-cams json requires --o2-cams-json")
                import json as _json
                with open(args.o2_cams_json) as _fh:
                    raw = _json.load(_fh)
                if isinstance(raw, dict):
                    raw = raw.get("cams", [])
                cams = [dict(name=r["name"],
                             w2v=np.asarray(r["w2v"], np.float64),
                             fx=float(r["fx"]), fy=float(r["fy"]),
                             W=float(r["W"]), H=float(r["H"]),
                             C=np.asarray(r["C"], np.float64)) for r in raw]
                sp = None
                print(f"o2-cams json: loaded {len(cams)} cameras from "
                      f"{args.o2_cams_json}", flush=True)
            else:
                from ops2 import split_names, load_cams
                sp = split_names(args.o2_scene)
                cams = load_cams(sp, "fit", args.o2_views)
            if args.o2_dump_cams:
                import json as _json
                with open(args.o2_dump_cams, "w") as _fh:
                    _json.dump([{"name": c["name"],
                                 "w2v": np.asarray(c["w2v"]).tolist(),
                                 "C": np.asarray(c["C"]).tolist(),
                                 "fx": float(c["fx"]), "fy": float(c["fy"]),
                                 "W": float(c["W"]), "H": float(c["H"])}
                                for c in cams], _fh, indent=1)
                print(f"o2-cams {_o2_cams}: dumped {len(cams)} cameras -> "
                      f"{args.o2_dump_cams}", flush=True)
            o2cams = O2Cameras(cams, dev)
        o2stats.update({"scene": args.o2_scene, "views": o2cams.V,
                        "names": o2cams.names, "grid": args.o2_grid,
                        "sigma_cut": args.o2_sigma_cut, "calibrate": bool(calibrate),
                        "device": args.o2_device})

    # edge / detail risk (plan route C): cache one float per gaussian once, from
    # the calibration views' edge maps sampled at each gaussian's projected centre
    if args.edge_risk and dev is not None and o2cams is not None:
        with prof.stage("edge_pre"):
            import edge_risk as _er
            _maps = _er.load_edge_maps(args.o2_scene, o2cams, args.edge_op)
            t_e = _er.gaussian_scores(t_mu, t_sc, t_q, t_op, o2cams, _maps)
        print(f"edge risk ({args.edge_op}): mean {float(t_e.mean()):.4f}"
              f" p90 {float(torch.quantile(t_e.double(), 0.9)):.4f}", flush=True)

    passes = []
    it = 0
    while True:
        N = int(t_mu.shape[0]) if dev is not None else int(mu.shape[0])
        if N <= target:
            break
        pst = {}
        prof.cur = pst
        t_pass = time.perf_counter()
        k_eff = min(max(1, args.k), max(1, N - 1))

        # numpy mirror of the state, only when a CPU operator needs it
        need_np = ((args.cost_device == "cpu") or (args.merge_device == "cpu")
                   or (args.opacity_mode != "O0" and args.o2_device == "cpu")
                   or (dev is None))
        if need_np:
            if dev is not None and (mu is None or args.merge_device == "gpu"):
                with prof.stage("state_d2h"):
                    mu = t_mu.cpu().numpy()
                    sc = t_sc.cpu().numpy()
                    q = t_q.cpu().numpy()
                    op = t_op.cpu().numpy()
                    sh = t_sh.cpu().numpy() if isinstance(t_sh, torch.Tensor) else sh

        if args.dump_dir and it == 0:
            d_mu = mu if mu is not None else t_mu.cpu().numpy()
            d_sc = sc if sc is not None else t_sc.cpu().numpy()
            d_q = q if q is not None else t_q.cpu().numpy()
            d_op = op if op is not None else t_op.cpu().numpy()
            if sh is not None and sh.shape[1]:
                d_sh = sh
            elif isinstance(t_sh, torch.Tensor):
                d_sh = t_sh.cpu().numpy()
            else:
                d_sh = np.zeros((N, 0), np.float32)
            snap = dict(mu=d_mu.copy(), sc=d_sc.copy(), q=d_q.copy(),
                        op=d_op.copy(), sh=d_sh.copy())

        # ---- graph construction -------------------------------------------
        with prof.stage("knn"):
            if mu is not None:
                mu_np = mu
            else:
                with prof.stage("knn_mu_d2h"):
                    mu_np = t_mu.cpu().numpy()
            nbr = knn_indices(mu_np, k=k_eff, tuned=bool(args.knn_tuned),
                                  variant=args.knn_variant, eps=args.knn_eps)

        # one-pass GPU path: the graph, the costs and the sort never leave the
        # device; only the greedy input (edges + stable order) is exported.
        gpu_fast = (args.edges_device == "gpu" and args.cost_device == "gpu")

        with prof.stage("edges"):
            if args.edges_device == "gpu":
                with torch.no_grad():
                    with prof.stage("edges_h2d"):
                        nbr_t = torch.as_tensor(nbr.astype(np.int32, copy=False), device=dev)
                        del nbr
                    edges_t = G.undirected_edges_gpu(nbr_t)
                    if not gpu_fast:
                        with prof.stage("edges_d2h"):
                            edges = edges_t.cpu().numpy()
                        del edges_t
            elif args.edges_device == "cpu":
                edges = knn_undirected_edges(nbr)
            else:
                edges = cpu_undirected_edges_fast(nbr)

        M = int(edges_t.shape[0]) if gpu_fast else int(edges.shape[0])

        # ---- costs ---------------------------------------------------------
        with prof.stage("cost"):
            if args.select_mode in ("random", "identity"):
                w_t = None          # selection ignores the cost entirely
            elif args.cost_device == "cpu":
                w = cpu_edge_costs(edges, mu, sc, q, op, sh, cp, args.block_edges)
            else:
                with torch.no_grad():
                    if not gpu_fast:
                        with prof.stage("cost_h2d"):
                            e_t = torch.as_tensor(edges, device=dev)
                    else:
                        e_t = edges_t
                    if args.scorer == "kl":
                        st = G.precompute(t_mu, t_sc, t_q, t_op)
                        w_t = G.edge_costs_gpu(e_t, st, t_sh, Z, args.lam_geo,
                                               args.lam_sh, args.block_edges)
                        del st
                    else:
                        with prof.stage("scorer_pre"):
                            st = {"mu": t_mu}          # only mu is shared
                            sst = G.precompute_scorer(
                                t_mu, t_sc, t_q,
                                need_sig=(args.scorer == "maha_fast"),
                                noblas=bool(args.no_blas),
                                need_axes=(args.cost_axis > 0 or args.cost_normal > 0
                                           or args.cost_sratio > 0
                                           or args.cost_plane > 0
                                           or args.cost_cg > 0
                                           or args.detail_nv > 0
                                           or args.cost_e4 > 0))
                        dnv_scale = 1.0
                        e4_scale = 1.0
                        if args.detail_nv > 0.0 or args.cost_e4 > 0.0:
                            with prof.stage("dnv"):
                                sst["dnv"] = G.normal_variation(
                                    sst["mnr"], nbr_t)
                            with prof.stage("dnv_scale"):
                                ns = min(200000, e_t.shape[0])
                                sub = e_t[:ns]
                                pn = 1.0 - (sst["mnr"][sub[:, 0]]
                                            * sst["mnr"][sub[:, 1]]).sum(-1).abs()
                                dn_ = torch.maximum(sst["dnv"][sub[:, 0]],
                                                    sst["dnv"][sub[:, 1]]) * pn
                                dnv_scale = max(float(dn_.mean()), 1e-9)
                        if args.cost_vis > 0.0:
                            with prof.stage("vis"):
                                from cams_fast import load_cams_fast as _lcf
                                _vc = _lcf(args.o2_scene, "fit", args.o2_views)
                                cnt = torch.zeros(N, device=dev)
                                _mu = t_mu.detach().cpu().numpy()
                                for _c in _vc:
                                    _P = _c["w2v"]
                                    _pc = _mu @ _P[:3, :3].T + _P[:3, 3]
                                    _z = _pc[:, 2]
                                    _ok = _z > 1e-3
                                    _zz = np.where(_ok, _z, 1.0)
                                    _px = _c["fx"] * _pc[:, 0] / _zz + _c["W"] / 2
                                    _py = _c["fy"] * _pc[:, 1] / _zz + _c["H"] / 2
                                    _in = (_ok & (_px >= 0) & (_px < _c["W"])
                                           & (_py >= 0) & (_py < _c["H"]))
                                    cnt += torch.as_tensor(_in, device=dev).float()
                                sst["vis"] = (cnt / max(len(_vc), 1))
                        if args.cost_e4 > 0.0:
                            with prof.stage("e4"):
                                ns = min(200000, N)
                                sub = torch.randperm(N, device=dev)[:ns]
                                vol = t_sc.prod(1)
                                vp = torch.quantile(vol[sub], 0.99).clamp(min=1e-12)
                                np_ = torch.quantile(sst["dnv"][sub], 0.99).clamp(min=1e-12)
                                sst["vol"] = (vol / vp).clamp(0, 1)
                                sst["dnv"] = (sst["dnv"] / np_).clamp(0, 1)
                                ns2 = min(200000, e_t.shape[0])
                                su = e_t[:ns2]
                                q_ = (torch.maximum(sst["vol"][su[:, 0]], sst["vol"][su[:, 1]])
                                      * (1.0 - torch.maximum(sst["dnv"][su[:, 0]],
                                                             sst["dnv"][su[:, 1]])))
                                e4_scale = max(float(q_.mean()), 1e-9)
                        if args.cost_cg_view > 0 and t_sh is not None and t_sh.shape[1]:
                            with prof.stage("viewcol"):
                                sst["viewcol"] = G.view_colour_descriptor(
                                    t_sh_s0 if t_sh_s0 is not None else t_sh,
                                    args.cost_cg_view)
                        cg_scale = 1.0
                        if args.cost_cg > 0.0:
                            # robust unit normalisation: divide the colour term by
                            # its own mean over a candidate subsample, so lambda_cg
                            # means the same thing for DC and for view-SH
                            with prof.stage("cg_scale"):
                                ns = min(200000, e_t.shape[0])
                                sub = e_t[:ns]
                                sh_c = t_sh_s0 if t_sh_s0 is not None else t_sh
                                if "viewcol" in sst:
                                    dv = sst["viewcol"][sub[:, 0]] - sst["viewcol"][sub[:, 1]]
                                    d_c = (dv * dv).mean(-1).sum(-1).sqrt()
                                else:
                                    d_c = torch.linalg.vector_norm(
                                        (sh_c[sub[:, 0]] - sh_c[sub[:, 1]])[:, 0:3], dim=1)
                                cg_scale = max(float(d_c.mean()), 1e-9)
                        if args.cost_plane > 0 or args.cost_cg > 0:
                            with prof.stage("rho"):
                                import localdensity
                                rho_t = localdensity.local_spacing(t_mu, nbr_t)
                        else:
                            rho_t = None
                        w_t = G.edge_costs_scorer_gpu(
                            e_t, st, sst,
                            t_sh_s0 if t_sh_s0 is not None else t_sh,
                            args.lam_geo, args.lam_sh,
                            args.scorer, args.block_edges,
                            lam_risk=args.sh_risk_lambda,
                            lam_axis=args.cost_axis,
                            lam_normal=args.cost_normal,
                            lam_sratio=args.cost_sratio,
                            lam_plane=args.cost_plane,
                            rho=rho_t,
                            lam_cg=args.cost_cg,
                            tau=args.cost_cg_tau,
                            cg_scale=cg_scale,
                            lam_dnv=args.detail_nv,
                            dnv_scale=dnv_scale,
                            lam_e4=args.cost_e4,
                            e4_scale=e4_scale,
                            lam_vis=args.cost_vis)
                        if args.cost_random:
                            w_t = torch.rand_like(w_t)
                        if t_e is not None and args.edge_lambda > 0.0:
                            with prof.stage("edge_cost"):
                                base = w_t / w_t.mean().clamp(min=1e-12)
                                w_t = base + args.edge_lambda * 0.5 * (
                                    t_e[e_t[:, 0]] + t_e[e_t[:, 1]])
                        del st, sst
                    if not gpu_fast:
                        del e_t
                        with prof.stage("cost_d2h"):
                            w = w_t.cpu().numpy()
                        del w_t

        merges_needed = N - target
        if args.cluster > 2:
            # one-shot cluster merge: each output consumes `cluster` sources and
            # removes cluster-1 gaussians, so this needs (cluster-1)*C sources.
            n_clusters = -(-merges_needed // (args.cluster - 1))
            P = min(2 * n_clusters, p_cap) if merges_needed > 0 else None
        else:
            P = min(merges_needed, p_cap) if merges_needed > 0 else None

        # ---- selection -----------------------------------------------------
        with prof.stage("sort"):
            if args.select_mode in ("random", "identity"):
                _M = int(edges_t.shape[0]) if gpu_fast else int(edges.shape[0])
                if args.select_mode == "random":
                    order = np.random.default_rng(0).permutation(_M).astype(np.int64)
                else:
                    order = np.arange(_M, dtype=np.int64)
            elif args.cost_device == "gpu":
                with torch.no_grad():
                    valid_t = torch.isfinite(w_t)
                    idx_t = torch.nonzero(valid_t, as_tuple=False).flatten()
                    order = idx_t[torch.argsort(w_t[idx_t], stable=True)] \
                        .to(torch.int32).cpu().numpy()
            else:
                valid = np.isfinite(w)
                idx = np.nonzero(valid)[0]
                order = idx[np.argsort(w[idx], kind="mergesort")]

        if int(getattr(args, "select_topn", 0)) > 0 and args.select_mode in ("cost", "cost-nodedup"):
            order = order[:int(args.select_topn)]

        if gpu_fast:
            with prof.stage("greedy_edges_d2h"):
                # cast on the GPU: halves the transfer (int64 -> int32)
                edges = edges_t.to(torch.int32).cpu().numpy()
            del edges_t
        if (args.dump_dir and it == 0 and args.cost_device == "gpu"
                and args.select_mode in ("cost", "cost-nodedup")):
            w = w_t.cpu().numpy()

        with prof.stage("greedy"):
            _e32 = edges.astype(np.int32, copy=False)
            if args.select_mode == "cost-nodedup":
                # literally skip the greedy: take the cheapest P pairs as-is,
                # WITHOUT enforcing disjointness (some gaussians will be reused)
                _P = len(order) if P is None else min(int(P), len(order))
                _take = order[:_P]
                pairs = np.stack([_e32[_take, 0], _e32[_take, 1]], 1).astype(np.int32)
            elif args.greedy_mode == "blocked":
                # the cloud changes every pass, so the partition is rebuilt
                _blk = G.spatial_blocks(mu_np, args.greedy_blocks)
                pairs = greedy_jit(_e32, order, N,
                                   -1 if P is None else int(P), _blk)
            else:
                pairs = greedy_jit(_e32, order, N, -1 if P is None else int(P))
            if args.dump_pairs and it == 0:
                os.makedirs(args.dump_dir or "dump", exist_ok=True)
                np.save(os.path.join(args.dump_dir or "dump",
                                     f"pairs_{args.scorer}.npy"), pairs.astype(np.int32))

        # ---- MPMM merge (+ O2cal opacity) -----------------------------------
        with prof.stage("mpmm"):
            if args.merge_device == "cpu":
                if args.opacity_mode == "O0":
                    mu, sc, q, op, sh = merge_pairs(mu, sc, q, op, sh, pairs)
                else:
                    # same algorithm on CPU: geometry from geom_merge_w (returns the
                    # merged components), opacity from the reference O2cal
                    from replay_merge import geom_merge_w
                    from o2gpu import opacity_o2cal_cpu
                    i64 = pairs[:, 0].astype(np.int64)
                    j64 = pairs[:, 1].astype(np.int64)
                    with prof.stage("mpmm_comp"):
                        mu_m, s_m, q_m, Wm, w_i, w_j = geom_merge_w(
                            mu, sc, q, i64, j64, op[i64], op[j64])
                        sh_m = ((w_i[:, None] * sh[i64] + w_j[:, None] * sh[j64])
                                / Wm[:, None]).astype(np.float32) if sh.shape[1] else sh
                    with prof.stage("o2"):
                        new_op = opacity_o2cal_cpu(
                            {"mu": mu, "sc": sc, "q": q, "op": op},
                            {"mu_m": mu_m, "sc_m": s_m, "q_m": q_m},
                            pairs, cams, grid=args.o2_grid,
                            sigma_cut=args.o2_sigma_cut, calibrate=calibrate,
                            stats=o2stats, timers=prof.tot)
                        pst["o2_mean_alpha"] = float(np.asarray(new_op).mean())
                    with prof.stage("mpmm_assemble"):
                        used = np.zeros(mu.shape[0], dtype=bool)
                        used[i64] = True
                        used[j64] = True
                        keep = ~used
                        mu = np.concatenate([mu[keep], mu_m], 0)
                        sc = np.concatenate([sc[keep], s_m], 0)
                        q = np.concatenate([q[keep], q_m], 0)
                        op = np.concatenate([op[keep], np.asarray(new_op, np.float32)], 0)
                        if sh.shape[1]:
                            sh = np.concatenate([sh[keep], sh_m], 0)
                if dev is not None:
                    t_mu = torch.as_tensor(mu, device=dev)
                    t_sc = torch.as_tensor(sc, device=dev)
                    t_q = torch.as_tensor(q, device=dev)
                    t_op = torch.as_tensor(op, device=dev)
                    t_sh = torch.as_tensor(sh, device=dev) if sh.shape[1] else sh
            else:
                with torch.no_grad():
                    sh_cache = None
                    with prof.stage("mpmm_h2d"):
                        p_t = torch.as_tensor(pairs, device=dev, dtype=torch.int64)
                    if args.cluster > 2:
                        # ---- one-shot small-cluster merge (plan route B) ------
                        with prof.stage("cluster"):
                            p_np = np.asarray(pairs, np.int64)
                            cen = (0.5 * (t_mu[p_t[:, 0]] + t_mu[p_t[:, 1]])
                                   ).cpu().numpy()
                            from scipy.spatial import cKDTree
                            _, nb = cKDTree(cen).query(cen, k=2, workers=-1)
                            nn = nb[:, 1]
                            free = np.ones(len(cen), dtype=bool)
                            rows = []
                            for a in range(len(cen)):
                                b = int(nn[a])
                                if free[a] and b != a and free[b]:
                                    rows.append((a, b))
                                    free[a] = free[b] = False
                            if not rows:
                                raise SystemExit("cluster: no groups")
                            rows = np.asarray(rows, np.int64)
                            g = np.concatenate([p_np[rows[:, 0]], p_np[rows[:, 1]]], 1)
                            groups = torch.as_tensor(g, device=dev, dtype=torch.int64)
                            comp = G.merge_clusters(t_mu, t_sc, t_q, t_op, t_sh,
                                                    groups, jacobi=bool(args.jacobi))
                            comp["op_m"] = G.cluster_alpha(t_mu, t_sc, t_q, t_op,
                                                           comp, groups, o2cams)
                        with prof.stage("mpmm_assemble"):
                            t_mu, t_sc, t_q, t_op, t_sh = G.merge_assemble(
                                t_mu, t_sc, t_q, t_op, t_sh,
                                groups.reshape(-1, 2), comp)
                        pst["clusters"] = int(groups.shape[0])
                        pst["N_after"] = int(t_mu.shape[0])
                        print(f"  cluster level: {int(groups.shape[0])} clusters of "
                              f"{args.cluster} -> N {N} -> {int(t_mu.shape[0])}",
                              flush=True)
                        continue
                    comp = G.merge_components(t_mu, t_sc, t_q, t_op, t_sh, p_t,
                                              timers=prof.cur,
                                              noblas=bool(args.no_blas),
                                              jacobi=bool(args.jacobi))
                    # cascade mode A (plan 2.4): O2cal only every k-th level, and
                    # always on the level that reaches the target, so intermediate
                    # gaussians keep the O0 (MPMM) opacity.
                    n_after = N - int(pairs.shape[0])
                    do_o2 = (args.opacity_mode != "O0") and (
                        (it + 1) % args.o2_every == 0 or n_after <= target)
                    sh_cache = None
                    sh_sums = None
                    if do_o2:
                        with prof.stage("o2"):
                            if args.sh_mode != "s0" and t_sh is not None and t_sh.shape[1]:
                                import shfix
                                if args.sh_sum:
                                    sh_sums = shfix.make_sums(int(p_t.shape[0]), dev)
                                else:
                                    sh_cache = shfix.make_cache(int(p_t.shape[0]),
                                                                o2cams.V, dev)
                            if args.o2_device == "gpu":
                                if o2pool is not None:
                                    # pair-conditioned view selection (PICS)
                                    import pics as _pics
                                    _t0 = time.perf_counter()
                                    _scr = o2pool.screen(
                                        comp["mu_m"], comp["sc_m"],
                                        sigma_cut=args.o2_sigma_cut,
                                        want_area=(args.o2_pick in ("info", "scale_opt")))
                                    if args.o2_pick == "fixed":
                                        _ids = [int(v) for v in
                                                args.o2_fixed_ids.split(",") if v != ""]
                                        _sel = torch.as_tensor(
                                            _ids, device=dev, dtype=torch.int32
                                        )[None].repeat(int(comp["mu_m"].shape[0]), 1)
                                    else:
                                        _sel = o2pool.select(
                                            _scr, strategy=args.o2_pick,
                                            B=args.o2_views, qmin=args.o2_qmin,
                                            fast_ids=o2fast_ids,
                                            nbins=(args.o2_nbins or None),
                                            pref_ids=o2pref_ids,
                                            track_lam=args.o2_track_lam,
                                            rstar=args.o2_rstar)
                                    del _scr
                                    _nsel = (_sel >= 0).sum(1).float()
                                    pst["o2_sel_views"] = float(_nsel.mean())
                                    pst["o2_sel_zero"] = float((_nsel == 0).float().mean())
                                    del _nsel
                                    if o2fast_ids is not None and _sel.shape[0] > 0:
                                        # R3: how often does the selector actually
                                        # deviate from the historical fast-B set?
                                        _fi = torch.as_tensor(
                                            o2fast_ids, device=dev,
                                            dtype=torch.int32)
                                        _k = min(_sel.shape[1], _fi.numel())
                                        _a = _sel[:, :_k]                    # (P,k)
                                        # order-insensitive: how often is each
                                        # fast camera kept in the selected set?
                                        _keep = [_a.eq(_fi[j]).any(1).float().mean()
                                                 for j in range(_k)]
                                        _kept = float(sum(float(x) for x in _keep)) / _k
                                        pst["o2_reroute"] = 1.0 - _kept
                                        print(f"  [o2-pick {args.o2_pick}] fast-B kept "
                                              f"{100*_kept:.1f}% of slots "
                                              f"(re-routed {100*pst['o2_reroute']:.1f}%)",
                                              flush=True)
                                    prof.tot["o2_pick"] = (prof.tot.get("o2_pick", 0.0)
                                                           + time.perf_counter() - _t0)
                                    new_op = _pics.opacity_o2cal_gpu_pool(
                                        {"mu": t_mu, "sc": t_sc, "q": t_q, "op": t_op},
                                        comp, p_t, _sel, o2pool,
                                        grid=args.o2_grid,
                                        sigma_cut=args.o2_sigma_cut,
                                        calibrate=calibrate, batch=args.o2_batch,
                                        stats=o2stats, timers=prof.tot,
                                        sh_sums=sh_sums,
                                        chunk=int(args.o2_batch) * int(args.o2_views),
                                        sh_ref_sel=o2shref)
                                    del _sel
                                else:
                                    from o2gpu import opacity_o2cal_gpu as _o2
                                    _o2_dump = {} if args.o2_debug_dump else None
                                    _dc_buf = None
                                    if args.o2_fit_dc:
                                        _dc_buf = torch.zeros(int(p_t.shape[0]), 3,
                                                              dtype=torch.float64,
                                                              device=dev)
                                    _eg_mode = str(args.o2_eigen_mode)
                                    _eg_eps = (0.0 if _eg_mode in ("E0", "E1")
                                               else 0.3) if args.o2_eigen_eps < 0 else \
                                              float(args.o2_eigen_eps)
                                    _eg_gamma = float(args.o2_eigen_gamma)
                                    if _eg_mode in ("E1",):
                                        _eg_gamma, _eg_eps = 1.0, 0.0
                                    if _eg_gamma <= 0.0:
                                        # core AABB extent from the point cloud only
                                        # (no voxel grid, no camera): 5-95% box
                                        _q05 = torch.quantile(t_mu, 0.05, dim=0)
                                        _q95 = torch.quantile(t_mu, 0.95, dim=0)
                                        _Lcore = float((_q95 - _q05).norm())
                                        _dcore = 0.20 * _Lcore
                                        _deff = max(float(args.o2_eigen_cdist) * _dcore, 1e-6)
                                        _eg_gamma = float(args.o2_eigen_fpx) / _deff
                                        print(f"  [o2-eigen] L_core={_Lcore:.3f} "
                                              f"d_core={_dcore:.3f} d_eff={_deff:.3f} "
                                              f"gamma={_eg_gamma:.3f} px/unit", flush=True)
                                    new_op = _o2({"mu": t_mu, "sc": t_sc, "q": t_q,
                                                  "op": t_op},
                                                 comp, p_t, o2cams, grid=args.o2_grid,
                                                 sigma_cut=args.o2_sigma_cut,
                                                 calibrate=calibrate,
                                                 batch=args.o2_batch,
                                                 stats=o2stats, timers=prof.tot,
                                                 noblas=bool(args.no_blas),
                                                 cache=sh_cache, sh_sums=sh_sums,
                                                 view_tau=float(args.o2_view_tau),
                                                 view_rmax=float(args.o2_view_rmax),
                                                 solver=str(args.o2_solver),
                                                 cf_beta=float(args.o2_cf_beta),
                                                 cf_agg=str(args.o2_cf_agg),
                                                 cf_fix=bool(args.o2_cf_fix),
                                                 cf_safe=bool(args.o2_cf_safe),
                                                 cf_c0=float(args.o2_cf_c0),
                                                 clean_mask=bool(args.o2_clean_mask),
                                                 bound_guard=bool(args.o2_cf_bound),
                                                 sh_dc=(t_sh[:, :3] if args.o2_fit_dc
                                                        and t_sh is not None
                                                        and t_sh.shape[1] >= 3 else None),
                                                 dc_out=_dc_buf,
                                                 eg_mode=str(args.o2_eigen_mode),
                                                 eg_gamma=_eg_gamma,
                                                 eg_eps=_eg_eps,
                                                 eg_wmode=str(args.o2_eigen_wmode),
                                                 eg_psi=float(args.o2_eigen_psi),
                                                 debug_dump=_o2_dump)
                                    if _o2_dump:
                                        torch.save(_o2_dump, args.o2_debug_dump)
                                        print(f"  [o2-debug] dumped "
                                              f"{list(_o2_dump.keys())} -> "
                                              f"{args.o2_debug_dump}", flush=True)
                                        _o2_dump = None
                            else:
                                from o2gpu import opacity_o2cal_cpu as _o2
                                new_op = _o2({"mu": mu, "sc": sc, "q": q, "op": op},
                                             {k: comp[k].cpu().numpy()
                                              for k in ("mu_m", "sc_m", "q_m")},
                                             pairs, cams, grid=args.o2_grid,
                                             sigma_cut=args.o2_sigma_cut,
                                             calibrate=calibrate, stats=o2stats,
                                             timers=prof.tot)
                        comp["op_m"] = new_op.to(device=dev, dtype=torch.float32)
                        if args.o2_fit_dc and _dc_buf is not None and \
                                comp.get("sh_m") is not None:
                            # `_dc_buf` is a transmittance-weighted average of the
                            # RAW f_dc coefficients.  rgb = SH_C0*dc + 0.5 is affine,
                            # so averaging the raw coefficients is exactly the
                            # weighted average of the colours: write them straight in.
                            _has = _dc_buf.abs().sum(1) > 0
                            _shm = comp["sh_m"]
                            if int(_has.sum()):
                                _shm[_has, :3] = _dc_buf[_has].to(torch.float32)
                            pst["o2_dc_fit_frac"] = float(_has.float().mean())
                        pst["o2_mean_alpha"] = float(new_op.mean())
                        if args.o2_verbose_pass and (it + 1) == args.o2_verbose_pass:
                            from o2gpu import opacity_o2cal_cpu as _ocpu
                            st_np = {"mu": t_mu.cpu().numpy(), "sc": t_sc.cpu().numpy(),
                                     "q": t_q.cpu().numpy(), "op": t_op.cpu().numpy()}
                            cmp_np = {k: comp[k].cpu().numpy()
                                      for k in ("mu_m", "sc_m", "q_m")}
                            a_c = _ocpu(st_np, cmp_np, pairs, cams, grid=args.o2_grid,
                                        sigma_cut=args.o2_sigma_cut, calibrate=False,
                                        stats=dict(clip_hi=0, clip_lo=0,
                                                   o2_fallback=0, o2_views=0))
                            a_c = np.asarray(a_c, np.float64)
                            print(f"VERBOSE pass {it+1} P={pairs.shape[0]} "
                                  f"gpu_am_mean={o2stats.get('am_mean_pre_cal')} "
                                  f"gpu_am_max={o2stats.get('am_max_pre_cal')} "
                                  f"gpu_f={o2stats.get('last_f')} "
                                  f"cpu_am_mean={a_c.mean():.6f} cpu_am_max={a_c.max():.4f} "
                                  f"gpu_out_mean={pst['o2_mean_alpha']:.6f}", flush=True)
                        if args.dump_o2_pass and (it + 1) == args.dump_o2_pass:
                            dd = args.dump_dir or "dump"
                            os.makedirs(dd, exist_ok=True)
                            np.savez(os.path.join(dd, f"o2pass{it+1}.npz"),
                                     mu=t_mu.cpu().numpy(), sc=t_sc.cpu().numpy(),
                                     q=t_q.cpu().numpy(), op=t_op.cpu().numpy(),
                                     sh=(t_sh.cpu().numpy() if t_sh is not None
                                         and t_sh.shape[1] else np.zeros((1, 0))),
                                     sh_m=comp["sh_m"].cpu().numpy(),
                                     pairs=np.asarray(pairs, np.int32),
                                     mu_m=comp["mu_m"].cpu().numpy(),
                                     sc_m=comp["sc_m"].cpu().numpy(),
                                     q_m=comp["q_m"].cpu().numpy(),
                                     grid=args.o2_grid, sigma_cut=args.o2_sigma_cut,
                                     calibrate=int(calibrate), n=int(t_mu.shape[0]))
                    sh_m_s0 = None
                    have_sums = (sh_sums is not None) or (sh_cache is not None)
                    if (args.sh_mode != "s0" and have_sums
                            and t_sh is not None and t_sh.shape[1]):
                        with prof.stage("shfix"):
                            import shfix
                            sh_stats = {}
                            if args.sh_sum and args.sh_mode == "dc":
                                comp["sh_m"] = shfix.correct_dc_from_sums(
                                    (t_mu, t_sc, t_q, t_op, t_sh), comp, p_t,
                                    sh_sums, lam=args.sh_lam, clip=args.sh_clip,
                                    min_diff=args.sh_min_diff, stats=sh_stats)
                            else:
                                comp["sh_m"] = shfix.correct_sh(
                                    (t_mu, t_sc, t_q, t_op, t_sh), comp, p_t, o2cams,
                                    mode=args.sh_mode, lam=args.sh_lam,
                                    clip=args.sh_clip, stats=sh_stats,
                                    cache=sh_cache if args.sh_cache else None,
                                    min_diff=args.sh_min_diff)
                            pst["sh_fix"] = sh_stats.get("sh_correction")
                    e_m = None
                    if t_e is not None:
                        # edge risk follows the same lineage as the model
                        wi_e = (G.TWO_PI_15 * t_op[p_t[:, 0]]
                                * t_sc[p_t[:, 0]].prod(-1) + 1e-12)
                        wj_e = (G.TWO_PI_15 * t_op[p_t[:, 1]]
                                * t_sc[p_t[:, 1]].prod(-1) + 1e-12)
                        e_m = ((wi_e * t_e[p_t[:, 0]] + wj_e * t_e[p_t[:, 1]])
                               / torch.clamp(wi_e + wj_e, min=1e-12))
                    # S0 SH lineage used by the scorer: must advance on *every*
                    # level, including the ones where O2cal (and thus the SH
                    # correction) is skipped by the cascade schedule.
                    sh_m_s0 = None
                    if args.sh_freeze_score and t_sh_s0 is not None:
                        w_i = (G.TWO_PI_15 * t_op[p_t[:, 0]]
                               * t_sc[p_t[:, 0]].prod(-1) + 1e-12)
                        w_j = (G.TWO_PI_15 * t_op[p_t[:, 1]]
                               * t_sc[p_t[:, 1]].prod(-1) + 1e-12)
                        Wm = torch.clamp(w_i + w_j, min=1e-12)
                        sh_m_s0 = ((w_i[:, None] * t_sh_s0[p_t[:, 0]]
                                    + w_j[:, None] * t_sh_s0[p_t[:, 1]])
                                   / Wm[:, None])
                    n_pre = int(t_mu.shape[0])
                    t_mu, t_sc, t_q, t_op, t_sh = G.merge_assemble(
                        t_mu, t_sc, t_q, t_op, t_sh, p_t, comp)
                    if e_m is not None:
                        keep_e = torch.ones(n_pre, dtype=torch.bool, device=dev)
                        keep_e[p_t[:, 0]] = False
                        keep_e[p_t[:, 1]] = False
                        t_e = torch.cat([t_e[keep_e], e_m], 0)
                    if sh_m_s0 is not None and t_sh_s0 is not None:
                        keep = torch.ones(n_pre, dtype=torch.bool, device=dev)
                        keep[p_t[:, 0]] = False
                        keep[p_t[:, 1]] = False
                        t_sh_s0 = torch.cat([t_sh_s0[keep], sh_m_s0], 0)
                    del comp

        pst["N"] = N
        pst["M"] = M
        pst["pairs"] = int(pairs.shape[0])
        pst["need"] = int(merges_needed)
        pst["pass_s"] = round(time.perf_counter() - t_pass, 3)
        passes.append({"pass": it + 1, **{k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in pst.items()}})

        if args.dump_dir and it == 0:
            os.makedirs(args.dump_dir, exist_ok=True)
            np.savez(os.path.join(args.dump_dir, "pass1.npz"),
                     edges=edges.astype(np.int32, copy=False),
                     w=w.astype(np.float64),
                     order=np.asarray(order, dtype=np.int64),
                     pairs=np.asarray(pairs, dtype=np.int32),
                     mu=snap["mu"], sc=snap["sc"], q=snap["q"],
                     op=snap["op"], sh=snap["sh"],
                     N=np.int64(N), M=np.int64(M), lam_geo=args.lam_geo,
                     lam_sh=args.lam_sh, n_mc=args.n_mc, seed=args.seed)
            del snap

        it += 1
        if args.max_passes and it >= args.max_passes:
            break

    prof.cur = None

    with prof.stage("d2h_final"):
        if dev is not None:
            mu = t_mu.cpu().numpy()
            sc = t_sc.cpu().numpy()
            q = t_q.cpu().numpy()
            op = t_op.cpu().numpy()
            sh = t_sh.cpu().numpy() if isinstance(t_sh, torch.Tensor) else sh

    with prof.stage("ply_write"):
        op = np.clip(op, 0.0, 1.0).astype(np.float32)
        write_fn(args.output, hdr, mu, op, sc, q, sh, app_names)

    total = time.perf_counter() - t_start
    rep = {
        "input": args.ply, "output": args.output,
        "N0": N0, "target": target, "final": int(mu.shape[0]),
        "args": vars(args),
        "passes": passes,
        "stage_s": {k: round(v, 4) for k, v in prof.tot.items()},
        "o2": o2stats if args.opacity_mode != "O0" else None,
        "end_to_end_s": round(total, 3),
        "peak_gpu_MB": (round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)
                        if dev is not None else 0.0),
        "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "gpu_name": (torch.cuda.get_device_name(dev) if dev is not None else None),
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(rep, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
