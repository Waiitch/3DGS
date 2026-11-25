#!/usr/bin/env bash
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
/mnt/home1/wyq/nanogs-work/venv/bin/python - <<'PY'
import numpy as np, sys
sys.path.insert(0, '/mnt/home1/wyq/nanogs-work/o2gpu5x')
from fastio import read_ply_fast

F = {
    "ours O0            (19.72)": "/tmp/fp_ours_o0.ply",
    "ours M2 analytic   (21.42)": "/mnt/home1/wyq/nanogs-work/o2gpu5x/out/mn_truck_M2.ply",
    "ghap orig          (19.31)": "/mnt/home1/wyq/ghap-work/ft_truck/truck10_orig/point_cloud/iteration_30000/point_cloud.ply",
    "ghap NanoGS        (21.26)": "/mnt/home1/wyq/ghap-work/ft_truck/truck10_nanogs_bc/point_cloud/iteration_30000/point_cloud.ply",
    "ghap GHAP          (15.39)": "/mnt/home1/wyq/ghap-work/ft_truck/truck10_ghap/point_cloud/iteration_30000/point_cloud.ply",
    "ghap O2cal(old)    (15.61)": "/mnt/home1/wyq/ghap-work/ft_truck/truck10_o2cal/point_cloud/iteration_30000/point_cloud.ply",
}
data = {}
print(f"{'file':28s} {'N':>8s} {'op_mean':>8s} {'op_p50':>8s} {'op>0.95%':>9s} {'op>0.99%':>9s} {'sc_p50':>8s}")
for tag, p in F.items():
    try:
        hdr, mu, op, sc, q, sh, names = read_ply_fast(p)
    except Exception as e:
        print(f"{tag:28s}  ERROR {e}"); continue
    op = np.asarray(op, dtype=np.float64).ravel()
    sc = np.asarray(sc, dtype=np.float64)
    data[tag] = (mu, op, sc)
    sca = sc.ravel()
    print(f"{tag:28s} {len(op):8d} {op.mean():8.4f} {np.median(op):8.4f} "
          f"{100*(op>0.95).mean():8.2f}% {100*(op>0.99).mean():8.2f}% "
          f"{np.median(sca):8.5f}")

print("\n=== 位置/属性是否同源（与 ghap orig 对比）===")
if "ghap orig          (19.31)" in data:
    mu0, op0, sc0 = data["ghap orig          (19.31)"]
    for tag, (mu, op, sc) in data.items():
        if tag.startswith("ghap orig"):
            continue
        n = min(len(mu0), len(mu))
        if mu0.shape == mu.shape:
            dmu = float(np.abs(mu0 - mu).max())
            dop = float(np.abs(op0 - op).mean())
            dsc = float(np.abs(sc0 - sc).mean())
            print(f"{tag:28s} shape 相同 | max|Δμ|={dmu:.3e}  mean|Δop|={dop:.4f}  mean|Δsc|={dsc:.4f}")
        else:
            print(f"{tag:28s} shape 不同 {mu0.shape} vs {mu.shape}")
PY
