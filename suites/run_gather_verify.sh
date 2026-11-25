#!/usr/bin/env bash
# Verify the gather optimisation: PSNR must be unchanged, o2 must drop.
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
COMMON="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
  --o2-views 8 --o2-sigma-cut 1.25 --sh-mode s0 --sh-clip 0.05 --cost-plane 2.0 --cost-cg 1.0
  --opacity-mode O2cal --o2-cams cover"
for sn in truck room garden bicycle playroom train drjohnson; do
  case $sn in train) md=tandt_train_vanilla;; drjohnson) md=drjohnson_vanilla;; *) md=${sn}_vanilla;; esac
  RID="gth_${sn}_CF"
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  CUDA_VISIBLE_DEVICES=$GPU $PY -u run_o2.py \
    --ply $M/$md/point_cloud/iteration_30000/point_cloud.ply \
    -o out/${RID}.ply $COMMON --o2-scene $D/$sn --o2-solver analytic --o2-cf-agg eq \
    --json results/av/${RID}.json > results/av/${RID}.log 2>&1
  if [ $? -ne 0 ]; then echo "  [$RID] FAIL"; tail -2 results/av/${RID}.log; continue; fi
  bash $EV out/${RID}.ply $M/$md "$RID" $GPU > eval/${RID}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
  $PY -c "
import json;d=json.load(open('results/av/${RID}.json'));o=d['o2'];s=d['stage_s']
print(f\"  ${sn}: PSNR=$p cf={s.get('o2_cf')} tot={s.get('o2')} calls={o.get('o2_calls')} batches={o.get('o2_batches')}\")"
done
echo GTH_DONE
