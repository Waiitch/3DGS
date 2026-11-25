#!/usr/bin/env bash
# One experiment on truck: does the closed-form DC (RGB) refit add on top of the
# opacity refit?  M2 = opacity only; M2c = opacity + DC colour.
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
S2="--av-core-aabb 0.80 --av-no-scale-gate --av-cover-target gauss --av-gauss-probe 20000 --av-pos-mode dual_domain --av-pos-b 12"
BASE="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
  --o2-views 8 --o2-sigma-cut 1.25 --sh-mode s0 --sh-clip 0.05 --cost-plane 2.0 --cost-cg 1.0
  --opacity-mode O2cal --o2-cams auto_virt $S2 --o2-scene $D/truck
  --o2-solver analytic --o2-cf-agg eq --o2-clean-mask --o2-cf-bound"
for spec in "M2|" "M2c|--o2-fit-dc"; do
  IFS='|' read -r TAG EXTRA <<< "$spec"
  RID="dct_${TAG}"
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU $PY -u run_o2.py \
    --ply $M/truck_vanilla/point_cloud/iteration_30000/point_cloud.ply \
    -o out/${RID}.ply $BASE $EXTRA --json results/av/${RID}.json \
    > results/av/${RID}.log 2>&1
  RC=$?; T1=$(date +%s)
  [ $RC -ne 0 ] && { echo "  [$TAG] FAIL"; tail -3 results/av/${RID}.log; continue; }
  bash $EV out/${RID}.ply $M/truck_vanilla "$RID" "$GPU" > eval/${RID}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
  ss=$(grep -oE '"SSIM": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
  lp=$(grep -oE '"LPIPS": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
  frac=$($PY -c "
import json;d=json.load(open('results/av/${RID}.json'))
p=d.get('pst',d);print(p.get('o2_dc_fit_frac','-'))" 2>/dev/null)
  echo "  [$TAG] PSNR=$p SSIM=$ss LPIPS=$lp dc_fit_frac=$frac  wall=$((T1-T0))s"
done
echo DC_DONE
