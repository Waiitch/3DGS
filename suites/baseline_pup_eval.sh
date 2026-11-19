#!/usr/bin/env bash
# evaluate the six PUP models with the same eval script
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
P=/mnt/home1/wyq/gaussian-splatting-pup/experiments
echo "=== PUP model sizes ==="
for t in pup_truck_k50 pup_truck_k10 pup_truck_k05 pup_room_k50 pup_room_k10 pup_room_k05; do
  f=$P/$t/point_cloud/iteration_35000/point_cloud.ply
  if [ -f "$f" ]; then
    n=$(head -c 2000 "$f" | tr -d '\0' | grep -m1 'element vertex' | awk '{print $3}')
    echo "$t: $n gaussians, $(du -m "$f" | cut -f1) MB"
  else
    echo "$t: MISSING"
  fi
done
i=0
for t in pup_truck_k50 pup_truck_k10 pup_truck_k05 pup_room_k50 pup_room_k10 pup_room_k05; do
  f=$P/$t/point_cloud/iteration_35000/point_cloud.ply
  [ -f "$f" ] || continue
  gpu=2
  bash $EV "$f" "$P/$t" "pup_$t" $gpu > eval/pup_$t.log 2>&1 &
  i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
for t in pup_truck_k50 pup_truck_k10 pup_truck_k05 pup_room_k50 pup_room_k10 pup_room_k05; do
  echo "$t: $(grep -o 'PSNR": [0-9.]*' eval/pup_$t.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/pup_$t.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/pup_$t.log | tail -1)"
done
echo PUP_EVAL_DONE
