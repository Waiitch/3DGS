#!/usr/bin/env bash
# Phase 2A: grid vs analytic O2cal on truck / room, real cover8 cameras.
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
CSV=timing/o2_analytic_truck_room.csv
[ -f "$CSV" ] || echo "run_id,scene,exp,solver,grid,sigma_cut,beta,agg,psnr,ssim,lpips,run_s" > "$CSV"
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' '; }
COMMON="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
        --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
        --o2-views 8 --sh-mode s0 --sh-clip 0.05 --cost-plane 2.0 --cost-cg 1.0
        --opacity-mode O2cal --o2-cams cover"
# tag | solver | grid | sigma_cut | beta | agg
EXPS="
G_def|grid|9|1.25|0|eq
G_fine|grid|13|2.0|0|eq
CF_eq_b0|analytic|9|1.25|0.0|eq
CF_eq_b05|analytic|9|1.25|0.5|eq
CF_eq_b10|analytic|9|1.25|1.0|eq
CF_eq_b15|analytic|9|1.25|1.5|eq
CF_area_b0|analytic|9|1.25|0.0|area
CF_area_b10|analytic|9|1.25|1.0|area
"
SCENES=${1:-truck,room}
echo "$EXPS" | while IFS='|' read -r TAG SOLVER GR SC BETA AGG; do
  [ -z "$TAG" ] && continue
  for sn in ${SCENES//,/ }; do
    RID="cf_${sn}_${TAG}"
    grep -q "^${RID}," "$CSV" 2>/dev/null && { echo "  [$RID] skip"; continue; }
    GR_GPU=$(pick_gpu); T0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GR_GPU $PY -u run_o2.py \
      --ply $M/${sn}_vanilla/point_cloud/iteration_30000/point_cloud.ply \
      -o out/${RID}.ply $COMMON --o2-scene $D/$sn \
      --o2-solver $SOLVER --o2-grid $GR --o2-sigma-cut $SC \
      --o2-cf-beta $BETA --o2-cf-agg $AGG \
      --json results/av/${RID}.json > results/av/${RID}.log 2>&1
    rc=$?; T1=$(date +%s)
    if [ $rc -ne 0 ]; then
      printf '%s,%s,%s,%s,%s,%s,%s,%s,FAIL,FAIL,FAIL,%s\n' \
        "$RID" "$sn" "$TAG" "$SOLVER" "$GR" "$SC" "$BETA" "$AGG" "$((T1-T0))" >> "$CSV"
      echo "  [$RID] FAIL"; tail -3 results/av/${RID}.log; continue; fi
    bash $EV out/${RID}.ply $M/${sn}_vanilla "$RID" $GR_GPU > eval/${RID}.log 2>&1
    p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    s=$(grep -oE '"SSIM": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    l=$(grep -oE '"LPIPS": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$RID" "$sn" "$TAG" "$SOLVER" "$GR" "$SC" "$BETA" "$AGG" "$p" "$s" "$l" "$((T1-T0))" >> "$CSV"
    echo "  [$RID] PSNR=$p"
  done
done
echo CF_DONE
