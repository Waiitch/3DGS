#!/usr/bin/env bash
# Is the "optimal combination" already saturated?  Test what can still be added on
# top of clip=0.05 + p2c1 + cover8:
#   e4    : --cost-e4 0.5        (changes the pair sequence)
#   cov12 : cover with 12 views  (changes the camera budget)
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
SH="--sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05"
P2="--cost-plane 2.0 --cost-cg 1.0"
printf 'scene,arm,e2e,o2,final\n' > timing/best.csv
run() {
  sn=$1; arm=$2; shift 2
  case $sn in
    truck)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;  SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)    MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;   SC=/mnt/home1/wyq/3dgs/data/room;;
    garden)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
    bicycle) MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla; SC=/mnt/home1/wyq/3dgs/data/bicycle;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=bst_${sn}_${arm}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 $P2 $SH "$@" \
    --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s']
print('${sn},${arm},%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/best.csv
  echo "  run $T done"
}
for sn in truck room garden bicycle; do
  run $sn cover_e4 --o2-cams cover --o2-views 8 --cost-e4 0.5
done
for sn in truck room garden; do
  run $sn cover12  --o2-cams cover --o2-views 12
done
cat timing/best.csv
echo "=== evals (2-way parallel) ==="
i=0
for spec in "truck:cover_e4" "room:cover_e4" "garden:cover_e4" "bicycle:cover_e4" \
            "truck:cover12" "room:cover12" "garden:cover12"; do
  IFS=: read sn arm <<< "$spec"
  case $sn in
    truck)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)    MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
    garden)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla;;
    bicycle) MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla;;
  esac
  gpu=$(( i % 2 == 0 ? 1 : 3 ))
  bash $EV out/bst_${sn}_${arm}.ply $MD bst_${sn}_${arm} $gpu > eval/bst_${sn}_${arm}.log 2>&1 &
  i=$((i+1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
echo "=== results ==="
for spec in "truck:cover_e4" "room:cover_e4" "garden:cover_e4" "bicycle:cover_e4" \
            "truck:cover12" "room:cover12" "garden:cover12"; do
  IFS=: read sn arm <<< "$spec"
  echo "$sn $arm: $(grep -o '"PSNR": [0-9.]*' eval/bst_${sn}_${arm}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/bst_${sn}_${arm}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/bst_${sn}_${arm}.log|tail -1)"
done
echo BEST_DONE
