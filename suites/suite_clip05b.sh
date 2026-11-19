#!/usr/bin/env bash
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
SH="--sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05"
for spec in "truck:cover" "room:cover" "garden:cover" "bicycle:cover" "truck:track" "room:track"; do
  IFS=: read sn arm <<< "$spec"
  case $sn in
    truck)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;  SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)    MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;   SC=/mnt/home1/wyq/3dgs/data/room;;
    garden)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
    bicycle) MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla; SC=/mnt/home1/wyq/3dgs/data/bicycle;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  case $arm in
    cover) CAM="--o2-cams cover --o2-views 8";;
    track) CAM="--o2-cams pair --o2-pick track --o2-views 8";;
  esac
  T=c05_${sn}_${arm}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 $CAM $SH \
    --cost-plane 2.0 --cost-cg 1.0 --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  echo "  $T done"
done
for spec in "truck:cover" "room:cover" "garden:cover" "bicycle:cover" "truck:track" "room:track"; do
  IFS=: read sn arm <<< "$spec"
  case $sn in
    truck)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)    MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
    garden)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla;;
    bicycle) MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla;;
  esac
  bash $EV out/c05_${sn}_${arm}.ply $MD c05_${sn}_${arm} 3 > eval/c05_${sn}_${arm}.log 2>&1
  echo "$sn $arm (clip .05): $(grep -o '"PSNR": [0-9.]*' eval/c05_${sn}_${arm}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/c05_${sn}_${arm}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/c05_${sn}_${arm}.log|tail -1)"
done
echo CLIP05B_DONE
