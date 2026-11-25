#!/usr/bin/env bash
# Three-way stacking: analytic O2cal solver + S2-b12 camera rig + dc SH mode.
#   s0: S2b12+analytic  (S2b12+grid already measured: 21.4064 / 27.3036 / ...)
#   dc: S2b12+grid  vs  S2b12+analytic
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
CSV=timing/o2_stack.csv
[ -f "$CSV" ] || echo "run_id,scene,sh_mode,solver,psnr,ssim,lpips,o2_s,run_s" > "$CSV"
S2="--av-core-aabb 0.80 --av-no-scale-gate --av-cover-target gauss --av-gauss-probe 20000 --av-pos-mode dual_domain --av-pos-b 12"
COMMON="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
  --o2-views 8 --o2-sigma-cut 1.25 --sh-clip 0.05 --cost-plane 2.0 --cost-cg 1.0
  --opacity-mode O2cal --o2-cams auto_virt $S2"
for spec in "s0|analytic" "dc|analytic" "dc|grid"; do
  IFS='|' read -r SHM SOLVER <<< "$spec"
  for sn in truck room garden bicycle playroom train drjohnson; do
    case $sn in train) md=tandt_train_vanilla;; drjohnson) md=drjohnson_vanilla;; *) md=${sn}_vanilla;; esac
    RID="stk_${sn}_${SHM}_${SOLVER}"
    grep -q "^${RID}," $CSV 2>/dev/null && { echo "  [$RID] skip"; continue; }
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
    T0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU $PY -u run_o2.py \
      --ply $M/$md/point_cloud/iteration_30000/point_cloud.ply \
      -o out/${RID}.ply $COMMON --sh-mode $SHM --o2-scene $D/$sn \
      --o2-solver $SOLVER --o2-cf-agg eq \
      --json results/av/${RID}.json > results/av/${RID}.log 2>&1
    RC=$?; T1=$(date +%s)
    if [ $RC -ne 0 ]; then
      printf '%s,%s,%s,%s,FAIL,FAIL,FAIL,0,%s\n' "$RID" "$sn" "$SHM" "$SOLVER" "$((T1-T0))" >> $CSV
      echo "  [$RID] FAIL"; tail -3 results/av/${RID}.log; continue; fi
    bash $EV out/${RID}.ply $M/$md "$RID" $GPU > eval/${RID}.log 2>&1
    p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    ss=$(grep -oE '"SSIM": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    lp=$(grep -oE '"LPIPS": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    o2=$(grep -oE '"o2":[ ]*[0-9.]+' results/av/${RID}.json|tail -1|grep -oE '[0-9.]+$')
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$RID" "$sn" "$SHM" "$SOLVER" "$p" "$ss" "$lp" "${o2:-0}" "$((T1-T0))" >> $CSV
    echo "  [$RID] PSNR=$p o2=${o2}s"
  done
done
echo STACK_DONE
