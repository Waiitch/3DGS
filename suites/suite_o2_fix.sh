#!/usr/bin/env bash
# Stage 3 of o2cal_1.md: end-to-end matrix with the overflow / clamp / weight fixes.
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
CSV=timing/o2_analytic_fix.csv
[ -f "$CSV" ] || echo "run_id,scene,exp,psnr,ssim,lpips,am_mean,o2_s" > "$CSV"
COMMON="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
  --o2-views 8 --o2-sigma-cut 1.25 --sh-mode s0 --sh-clip 0.05 --cost-plane 2.0 --cost-cg 1.0
  --opacity-mode O2cal --o2-cams cover"
EXPS="
G_def|--o2-solver grid
CF_eq_safe|--o2-solver analytic --o2-cf-agg eq --o2-cf-fix --o2-cf-safe
CF_ecc_safe|--o2-solver analytic --o2-cf-agg ecc --o2-cf-fix --o2-cf-safe
CF_eccf_safe|--o2-solver analytic --o2-cf-agg ecc_floor --o2-cf-c0 0.15 --o2-cf-fix --o2-cf-safe
CF_eccf20_safe|--o2-solver analytic --o2-cf-agg ecc_floor --o2-cf-c0 0.20 --o2-cf-fix --o2-cf-safe
CF_eq_nofix|--o2-solver analytic --o2-cf-agg eq
"
for sn in truck room; do
  echo "$EXPS" | while IFS='|' read -r TAG FLAGS; do
    [ -z "$TAG" ] && continue
    RID="fx_${sn}_${TAG}"
    grep -q "^${RID}," "$CSV" 2>/dev/null && { echo "  [$RID] skip"; continue; }
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
    CUDA_VISIBLE_DEVICES=$GPU $PY -u run_o2.py \
      --ply $M/${sn}_vanilla/point_cloud/iteration_30000/point_cloud.ply \
      -o out/${RID}.ply $COMMON --o2-scene $D/$sn $FLAGS \
      --json results/av/${RID}.json > results/av/${RID}.log 2>&1
    if [ $? -ne 0 ]; then echo "  [$RID] FAIL"; tail -2 results/av/${RID}.log; continue; fi
    bash $EV out/${RID}.ply $M/${sn}_vanilla "$RID" $GPU > eval/${RID}.log 2>&1
    p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    ss=$(grep -oE '"SSIM": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    lp=$(grep -oE '"LPIPS": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    am=$(grep -oE '"am_mean_pre_cal":[ ]*[0-9.]+' results/av/${RID}.json|tail -1|grep -oE '[0-9.]+$')
    o2=$(grep -oE '"o2":[ ]*[0-9.]+' results/av/${RID}.json|tail -1|grep -oE '[0-9.]+$')
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$RID" "$sn" "$TAG" "$p" "$ss" "$lp" "${am:-0}" "${o2:-0}" >> "$CSV"
    echo "  [$RID] PSNR=$p am=${am}"
  done
done
echo FIX_DONE
