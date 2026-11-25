#!/usr/bin/env bash
# Phase 1 minimalist ablation, truck + room (o2cal_2.md).
#   M0 = S2-b12 + analytic + p2c1 + sigma_cut mask   (already measured)
#   M1 = M0 minus p2c1 (pure scale_norm)
#   M2 = M0 with clean mask + rigorous bound guard
#   M3 = M1 + clean mask + bound guard   <- the minimalist release config
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
CSV=timing/o2_minimalist_verify.csv
[ -f "$CSV" ] || echo "run_id,phase,exp_id,scene,rho,cams_mode,n_views,use_p2c1,use_sigma_cut,solver_impl,psnr,ssim,lpips,t_o2_ms,run_s" > "$CSV"
S2="--av-core-aabb 0.80 --av-no-scale-gate --av-cover-target gauss --av-gauss-probe 20000 --av-pos-mode dual_domain --av-pos-b 12"
BASE="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
  --o2-views 8 --o2-sigma-cut 1.25 --sh-mode s0 --sh-clip 0.05
  --opacity-mode O2cal --o2-cams auto_virt $S2"
P2C1="--cost-plane 2.0 --cost-cg 1.0"
CLEAN="--o2-clean-mask --o2-cf-bound"
# exp | p2c1? | clean? | plain
EXPS="
M0|1|0|S2-b12 + analytic + p2c1 + sigma_cut
M1|0|0|S2-b12 + analytic (no p2c1)
M2|1|1|S2-b12 + analytic + p2c1 + clean mask + bound
M3|0|1|S2-b12 + analytic + clean mask + bound (minimal)
"
for sn in truck room; do
  echo "$EXPS" | while IFS='|' read -r EXP P2 CLEANF DESC; do
    [ -z "$EXP" ] && continue
    RID="mn_${sn}_${EXP}"
    grep -q "^${RID}," "$CSV" 2>/dev/null && { echo "  [$RID] skip"; continue; }
    F="$BASE --o2-solver analytic --o2-cf-agg eq"
    [ "$P2" = "1" ] && F="$F $P2C1"
    [ "$CLEANF" = "1" ] && F="$F $CLEAN"
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
    T0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU $PY -u run_o2.py \
      --ply $M/${sn}_vanilla/point_cloud/iteration_30000/point_cloud.ply \
      -o out/${RID}.ply $F --o2-scene $D/$sn \
      --json results/av/${RID}.json > results/av/${RID}.log 2>&1
    RC=$?; T1=$(date +%s)
    if [ $RC -ne 0 ]; then echo "  [$RID] FAIL"; tail -3 results/av/${RID}.log; continue; fi
    bash $EV out/${RID}.ply $M/${sn}_vanilla "$RID" $GPU > eval/${RID}.log 2>&1
    p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    ss=$(grep -oE '"SSIM": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    lp=$(grep -oE '"LPIPS": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    o2=$(grep -oE '"o2":[ ]*[0-9.]+' results/av/${RID}.json|tail -1|grep -oE '[0-9.]+$')
    zs=$(grep -oE '"o2_fallback":[ ]*[0-9]+' results/av/${RID}.json|tail -1|grep -oE '[0-9]+$')
    printf '%s,1,%s,%s,0.10,auto_virt,8,%s,%s,analytic,%s,%s,%s,%s,%s\n' \
      "$RID" "$EXP" "$sn" "$P2" "$((1-CLEANF))" "$p" "$ss" "$lp" "${o2:-0}" "$((T1-T0))" >> "$CSV"
    echo "  [$RID] PSNR=$p o2=${o2}s zero_sup=${zs}"
  done
done
echo MN_DONE
