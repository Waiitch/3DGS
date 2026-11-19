"""Export camera / selection / support data for the Stage-0 figures (plan 14.2).

Writes one small JSON per scene with
    C        (V,3) camera centres
    axis     (V,3) world-space optical axes
    ids      {fast, cover, track} camera indices into C
    pairs    (n,3) merged-Gaussian midpoints of a pair sample
    support  (n,)   how many of the 8 fast views actually fit that pair
The plotting itself runs elsewhere (this venv has no matplotlib).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pics                                        # noqa: E402
import cams_track                                  # noqa: E402
from cams_fast import load_cams_fast, pick_cover_views   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figdata")
    ap.add_argument("--n-pairs", type=int, default=6000)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    os.makedirs(args.out, exist_ok=True)
    cases = [
        ("truck", "/mnt/home1/wyq/3dgs/data/truck", "dump_p2c1/truck_o2pass1.npz"),
        ("room", "/mnt/home1/wyq/3dgs/data/room", "dump_room_p2c1/room_o2pass1.npz"),
        ("garden", "/mnt/home1/wyq/3dgs/data/garden", "dump_garden/garden_o2pass1.npz"),
        ("bicycle", "/mnt/home1/wyq/3dgs/data/bicycle", "dump_bicycle/bicycle_o2pass1.npz"),
    ]
    for tag, scene, dump in cases:
        z = np.load(dump)
        state = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
                 for k in ("mu", "sc", "q", "op")}
        comp = {k: torch.as_tensor(np.ascontiguousarray(z[k]), device=dev)
                for k in ("mu_m", "sc_m", "q_m")}
        pairs = torch.as_tensor(np.ascontiguousarray(z["pairs"]).astype(np.int64),
                                device=dev)
        allc = load_cams_fast(scene, "fit", 0)
        fastc = load_cams_fast(scene, "fit", args.B)
        pool = pics.CamPool(allc, dev)
        n2i = {n: i for i, n in enumerate(pool.names)}
        fast_ids = [n2i[c["name"]] for c in fastc]
        M, _ = cams_track.load_overlap(scene, pool)
        pool.M = torch.as_tensor(M, device=dev)
        cover_ids = pick_cover_views(allc, state["mu"].cpu().numpy(), k=args.B)

        n = min(args.n_pairs, int(pairs.shape[0]))
        pil = torch.linspace(0, int(pairs.shape[0]) - 1, n).round().to(torch.int64).to(dev)
        scr = pool.screen(comp["mu_m"], comp["sc_m"], want_area=False)
        scrub = {"q": scr["q"][pil], "area": None}
        track_ids = pool.select(scrub, strategy="track", B=args.B,
                                qmin=0.05).cpu().numpy()
        # per-pair fast8 support count on the same sample (exact patch mask)
        pid = torch.arange(n, device=dev).repeat_interleave(len(fast_ids))
        cid = torch.as_tensor(fast_ids, device=dev).repeat(n)
        sub = {k: v[pil] for k, v in comp.items()}
        st = pool.patch_stats(state, sub, pairs[pil], pid, cid)
        supp = st["ok"].to(torch.int32).reshape(n, len(fast_ids)).sum(1).cpu().numpy()
        supp_np = np.asarray(supp)
        pmid = comp["mu_m"][pil].cpu().numpy()
        zero_idx = np.nonzero(supp_np == 0)[0]
        if zero_idx.size:              # the zero-support pair closest to the centre
            cen = pmid.mean(0)
            ex = int(zero_idx[np.argmin(((pmid[zero_idx] - cen) ** 2).sum(1))])
        else:
            ex = 0
        out = {
            "scene": tag,
            "example": {"pair": comp["mu_m"][pil][ex].cpu().numpy().round(4).tolist(),
                        "support": int(supp[ex]),
                        "ids": [int(v) for v in np.asarray(track_ids)[ex]]},
            "C": pool.C.cpu().numpy().round(4).tolist(),
            "axis": pool.axis.cpu().numpy().round(4).tolist(),
            "names": pool.names,
            "ids": {"fast": [int(v) for v in fast_ids],
                    "cover": [int(v) for v in cover_ids],
                    "track": [int(v) for v in np.asarray(track_ids)[0]]},
            "pairs": comp["mu_m"][pil].cpu().numpy().round(4).tolist(),
            "support": supp.tolist(),
        }
        fp = os.path.join(args.out, f"figdata_{tag}.json")
        with open(fp, "w") as f:
            json.dump(out, f)
        print(f"{tag}: V={pool.V} pairs={n} zero-support {float((supp==0).mean())*100:.1f}% "
              f"-> {fp}", flush=True)


if __name__ == "__main__":
    main()
