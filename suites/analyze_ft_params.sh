#!/usr/bin/env bash
# Decompose what the appearance fine-tune changed: geometry? opacity? SH?
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
D=/mnt/home1/wyq/ghap-work/ft_truck/truck10_nanogs_bc
M=/mnt/home1/wyq/3dgs/mpl_las/out

$PY - <<'PY'
import numpy as np, sys
sys.path.insert(0, '/mnt/home1/wyq/nanogs-work/o2gpu5x')
from fastio import read_ply_fast, store_ply_fast

D = "/mnt/home1/wyq/ghap-work/ft_truck/truck10_nanogs_bc"
p30 = f"{D}/point_cloud/iteration_30000/point_cloud.ply"
p45 = f"{D}/point_cloud/iteration_45000/point_cloud.ply"
h30, mu30, op30, sc30, q30, sh30, a30 = read_ply_fast(p30)
h45, mu45, op45, sc45, q45, sh45, a45 = read_ply_fast(p45)
print(f"N30={len(mu30)} N45={len(mu45)}  sh30={sh30.shape} sh45={sh45.shape}")

print("\n=== 几何是否被冻结（逐元素最大差）===")
print(f"  max|Δμ|  = {np.abs(mu30-mu45).max():.3e}")
print(f"  max|Δsc| = {np.abs(sc30-sc45).max():.3e}")
print(f"  max|Δq|  = {np.abs(q30-q45).max():.3e}")
print("  → 三个都为 0（或 fp 误差级）即证明几何冻结")

op30 = op30.ravel().astype(np.float64); op45 = op45.ravel().astype(np.float64)
d = op45 - op30
print("\n=== 不透明度变化 ===")
print(f"  mean {op30.mean():.4f} -> {op45.mean():.4f}  (Δmean {op45.mean()-op30.mean():+.4f})")
print(f"  |Δ| mean={np.abs(d).mean():.4f} p50={np.median(np.abs(d)):.4f} p95={np.quantile(np.abs(d),0.95):.4f} max={np.abs(d).max():.4f}")
print(f"  下降占比 {100*(d<0).mean():.1f}%  上升占比 {100*(d>0).mean():.1f}%")

sh30f = sh30.reshape(len(mu30), -1).astype(np.float64)
sh45f = sh45.reshape(len(mu45), -1).astype(np.float64)
ds = sh45f - sh30f
c3 = 3
print("\n=== SH 变化 ===")
print(f"  DC(前3):  mean|Δ|={np.abs(ds[:, :c3]).mean():.5f}  与 op 的相关性={np.corrcoef(ds[:, :c3].ravel(), np.repeat(d, c3))[0,1]:+.3f}")
if sh30f.shape[1] > c3:
    print(f"  高阶(其余 {sh30f.shape[1]-c3}): mean|Δ|={np.abs(ds[:, c3:]).mean():.5f}")
print(f"  整体 SH 平均绝对变化 {np.abs(ds).mean():.5f}（原始典型幅度 {np.abs(sh30f).mean():.5f}）")

# transplant: 45k with one group reverted to 30k
store_ply_fast("/tmp/ft45_op30.ply", h45, mu45, op30.astype(np.float32), sc45, q45, sh45, a45)
store_ply_fast("/tmp/ft45_sh30.ply", h45, mu45, op45.astype(np.float32), sc45, q45, sh30, a45)
store_ply_fast("/tmp/ft45_both30.ply", h45, mu45, op30.astype(np.float32), sc45, q45, sh30, a45)
print("\nwrote /tmp/ft45_op30.ply (op@30k), ft45_sh30.ply (sh@30k), ft45_both30.ply (=30k)")
PY

echo "===== 移植消融评测 ====="
for spec in "45k=both@45k|$D/point_cloud/iteration_45000/point_cloud.ply" \
            "op@30k|/tmp/ft45_op30.ply" \
            "sh@30k|/tmp/ft45_sh30.ply" \
            "both@30k|/tmp/ft45_both30.ply"; do
  IFS='|' read -r TAG PLY <<< "$spec"
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  bash $EV "$PLY" "$D" "tx_${TAG}" "$GPU" > /tmp/tx_${TAG}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' /tmp/tx_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  ss=$(grep -oE '"SSIM": [0-9.]+' /tmp/tx_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  echo "  $TAG : PSNR=$p SSIM=$ss"
done
echo TX_DONE
