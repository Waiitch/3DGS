#!/usr/bin/env bash
# Dump playroom (slow CF) and room (fast CF) on the same batch shape, then bench.
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
COMMON="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
  --o2-views 8 --o2-sigma-cut 1.25 --sh-mode s0 --sh-clip 0.05 --cost-plane 2.0 --cost-cg 1.0
  --opacity-mode O2cal --o2-cams cover"
for sn in playroom room bicycle; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  CUDA_VISIBLE_DEVICES=$GPU $PY -u run_o2.py \
    --ply $M/${sn}_vanilla/point_cloud/iteration_30000/point_cloud.ply \
    -o /tmp/bn_${sn}.ply $COMMON --o2-scene $D/$sn --o2-solver grid \
    --o2-debug-dump /tmp/bn_${sn}.pt --json /tmp/bn_${sn}.json > /tmp/bn_${sn}.log 2>&1
  echo "  dumped $sn rc=$?"
done
echo "=== micro-benchmark (same shape, different data) ==="
$PY -u bench_o2_cf.py /tmp/bn_playroom.pt /tmp/bn_room.pt /tmp/bn_bicycle.pt 2>&1 | tail -10
