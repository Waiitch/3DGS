#!/usr/bin/env bash
# Isolation proof: the eigen solver must be bit-identical under any camera set.
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
COMMON="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
  --o2-sigma-cut 1.25 --sh-mode s0 --sh-clip 0.05 --cost-plane 2.0 --cost-cg 1.0
  --opacity-mode O2cal --o2-scene $D/truck --o2-solver eigen --o2-eigen-mode E1"
for C in cover auto_virt virtual; do
  GR=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  CUDA_VISIBLE_DEVICES=$GR $PY -u run_o2.py \
    --ply $M/truck_vanilla/point_cloud/iteration_30000/point_cloud.ply \
    -o /tmp/iso_$C.ply $COMMON --o2-cams $C --json /tmp/iso_$C.json > /tmp/iso_$C.log 2>&1
  if [ $? -ne 0 ]; then echo "  cams=$C FAILED"; tail -3 /tmp/iso_$C.log; continue; fi
  bash $EV /tmp/iso_$C.ply $M/truck_vanilla "iso_$C" $GR > /tmp/isoeval_$C.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' /tmp/isoeval_$C.log|tail -1|grep -oE '[0-9.]+')
  md5=$(md5sum /tmp/iso_$C.ply | cut -c1-12)
  echo "  cams=$C  PSNR=$p  ply_md5=$md5"
done
echo ISO_DONE
