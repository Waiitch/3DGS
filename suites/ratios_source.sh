#!/usr/bin/env bash
# rho = 1.0 reference: evaluate the source 30k models with the same script
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
MD_T=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla
MD_R=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla
bash $EV $MD_T/point_cloud/iteration_30000/point_cloud.ply $MD_T rt_src_truck 1 > eval/rt_src_truck.log 2>&1 &
bash $EV $MD_R/point_cloud/iteration_30000/point_cloud.ply $MD_R rt_src_room 3 > eval/rt_src_room.log 2>&1 &
wait
for s in truck room; do
  echo "src $s: $(grep -o 'PSNR": [0-9.]*' eval/rt_src_${s}.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/rt_src_${s}.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/rt_src_${s}.log | tail -1)"
done
ls -l $MD_T/point_cloud/iteration_30000/point_cloud.ply $MD_R/point_cloud/iteration_30000/point_cloud.ply | awk '{print $5, $9}'
echo SRC_DONE
