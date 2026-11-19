#!/usr/bin/env bash
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
source /mnt/home1/wyq/3dgs/miniconda3/etc/profile.d/conda.sh
conda activate gs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

echo "=== REFINE @ 1% (truck + room) ==="
R=/mnt/home1/wyq/nanogs-work/o2gpu5x/refine_out
cd /mnt/home1/wyq/baselines/REFINE || exit 1
for sn in truck room; do
  case $sn in
    truck) ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)  ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
  esac
  D=$R/refine_${sn}_k1; mkdir -p $D
  t0=$(date +%s)
  python REFINE_pruning.py --start_pointcloud $ORIG/point_cloud/iteration_30000/point_cloud.ply \
    --json_path $ORIG/cameras.json --output_path $D/point_cloud.ply --prune_percent 0.99 \
    > logs_${sn}_k1.log 2>&1
  t1=$(date +%s)
  n=$(head -c 2000 $D/point_cloud.ply | tr -d '\0' | grep -m1 'element vertex' | awk '{print $3}')
  echo "  REFINE $sn k1: $((t1-t0))s  N=$n"
done
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
for sn in truck room; do
  case $sn in
    truck) SRC=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)  SRC=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
  esac
  bash $EV $R/refine_${sn}_k1/point_cloud.ply $SRC refine_${sn}_k1 1 > eval/refine_${sn}_k1.log 2>&1
  echo "  REFINE $sn k1: $(grep -o 'PSNR": [0-9.]*' eval/refine_${sn}_k1.log|tail -1) $(grep -o 'SSIM": [0-9.]*' eval/refine_${sn}_k1.log|tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/refine_${sn}_k1.log|tail -1)"
done

echo "=== 鎴戜滑鐨?room @ 0.01锛歜ase vs ax10 ==="
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
RM=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla; RMS=/mnt/home1/wyq/3dgs/data/room
SH="--sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05"
P2="--cost-plane 2.0 --cost-cg 1.0"
for tag in base ax10; do
  extra=""; [ $tag = "ax10" ] && extra="--cost-axis 1.0"
  T=exr_${tag}
  $PY -u run_o2.py --ply $RM/point_cloud/iteration_30000/point_cloud.ply -o out/${T}.ply \
    --ratio 0.01 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --o2-scene $RMS --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-batch 32768 --o2-cams cover --o2-views 8 --o2-sigma-cut 1.25 \
    $SH $P2 $extra --json results/${T}.json > results/${T}.log 2>&1
  echo "  run $T rc=$?"
done
for tag in base ax10; do
  bash $EV out/exr_${tag}.ply $RM exr_${tag} 3 > eval/exr_${tag}.log 2>&1
  echo "  room 0.01 $tag: $(grep -o 'PSNR": [0-9.]*' eval/exr_${tag}.log|tail -1) $(grep -o 'SSIM": [0-9.]*' eval/exr_${tag}.log|tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/exr_${tag}.log|tail -1)"
done
echo EXTREME2_DONE
