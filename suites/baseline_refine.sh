#!/usr/bin/env bash
# REFINE (rendering-free, zero-shot) at keep 50% / 10% / 5%, with timing
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
source /mnt/home1/wyq/3dgs/miniconda3/etc/profile.d/conda.sh
conda activate gs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
OUT=/mnt/home1/wyq/nanogs-work/o2gpu5x/refine_out
mkdir -p $OUT
cd /mnt/home1/wyq/baselines/REFINE || exit 1
printf 'scene,keep,prune,seconds,gaussians,mb\n' > /mnt/home1/wyq/nanogs-work/o2gpu5x/timing/refine.csv

for spec in "truck:0.5:50" "truck:0.9:10" "truck:0.95:5" \
            "room:0.5:50"  "room:0.9:10"  "room:0.95:5"; do
  IFS=: read sn pc keep <<< "$spec"
  case $sn in
    truck) ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)  ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
  esac
  D=$OUT/refine_${sn}_k${keep}
  mkdir -p $D
  t0=$(date +%s)
  python REFINE_pruning.py \
      --start_pointcloud $ORIG/point_cloud/iteration_30000/point_cloud.ply \
      --json_path $ORIG/cameras.json \
      --output_path $D/point_cloud.ply \
      --prune_percent $pc > logs_${sn}_k${keep}.log 2>&1
  rc=$?
  t1=$(date +%s)
  sec=$((t1 - t0))
  if [ -f $D/point_cloud.ply ]; then
    n=$(head -c 2000 $D/point_cloud.ply | tr -d '\0' | grep -m1 'element vertex' | awk '{print $3}')
    mb=$(du -m $D/point_cloud.ply | cut -f1)
  else
    n=0; mb=0
  fi
  echo "$sn keep=$keep: rc=$rc ${sec}s ${n} gaussians ${mb} MB"
  echo "$sn,$keep,$pc,$sec,$n,$mb" >> /mnt/home1/wyq/nanogs-work/o2gpu5x/timing/refine.csv
  grep -iE "throughput|reduction|time" logs_${sn}_k${keep}.log | tail -3
done
echo "=== REFINE evals ==="
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
i=0
for spec in "truck:50" "truck:10" "truck:5" "room:50" "room:10" "room:5"; do
  IFS=: read sn keep <<< "$spec"
  case $sn in
    truck) SRC=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)  SRC=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
  esac
  f=$OUT/refine_${sn}_k${keep}/point_cloud.ply
  [ -f "$f" ] || { echo "$sn k$keep MISSING"; continue; }
  gpu=$(( i % 2 == 0 ? 1 : 3 ))
  bash $EV "$f" "$SRC" "refine_${sn}_k${keep}" $gpu > eval/refine_${sn}_k${keep}.log 2>&1 &
  i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
for spec in "truck:50" "truck:10" "truck:5" "room:50" "room:10" "room:5"; do
  IFS=: read sn keep <<< "$spec"
  echo "REFINE $sn keep=$keep%: $(grep -o 'PSNR": [0-9.]*' eval/refine_${sn}_k${keep}.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/refine_${sn}_k${keep}.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/refine_${sn}_k${keep}.log | tail -1)"
done
echo "=== PUP prune-only 重测 ==="
P=/mnt/home1/wyq/gaussian-splatting-pup/experiments
for t in pupnr_truck_k50 pupnr_truck_k10 pupnr_truck_k05; do
  f=$P/$t/point_cloud/iteration_30002/point_cloud.ply
  bash $EV "$f" /mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla "nr2_$t" 3 > eval/nr2_$t.log 2>&1
  echo "PUP-norefine $t: $(grep -o 'PSNR": [0-9.]*' eval/nr2_$t.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/nr2_$t.log | tail -1)"
done
echo REFINE_ALL_DONE
