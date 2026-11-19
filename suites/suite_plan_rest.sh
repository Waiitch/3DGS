#!/usr/bin/env bash
# plan §18.1 (fixed budget scan, end-to-end) + §19.2 (camera x merge-strategy coupling)
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
SH="--sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05"
P2="--cost-plane 2.0 --cost-cg 1.0"
printf 'scene,arm,e2e,o2,final\n' > timing/plan_rest.csv
run() {
  sn=$1; tag=$2; cams=$3; views=$4; mer="$5"
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;  SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;   SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=pr_${sn}_${tag}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --o2-scene $SC --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-batch 32768 --o2-sigma-cut 1.25 \
    --o2-cams $cams --o2-views $views $mer $SH \
    --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s']
print('${sn},${tag},%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/plan_rest.csv
  echo "  run $T done"
}
# §18.1 fixed budget scan (p2c1 + cover @ B)
for sn in truck room garden; do
  run $sn b4  cover 4  "$P2"
  run $sn b6  cover 6  "$P2"
  run $sn b12 cover 12 "$P2"
done
# §19.2 coupling: camera gain under the *plain* scale_norm cost (no p2c1)
for sn in truck room garden; do
  run $sn sn_cover cover 8 ""
done
cat timing/plan_rest.csv
echo "=== evals (3-way) ==="
i=0
for spec in truck:b4 truck:b6 truck:b12 room:b4 room:b6 room:b12 garden:b4 garden:b6 garden:b12 \
            truck:sn_cover room:sn_cover garden:sn_cover; do
  IFS=: read sn tag <<< "$spec"
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla;;
  esac
  gpu=$(( i % 3 )); [ $gpu -eq 2 ] && gpu=3
  bash $EV out/pr_${sn}_${tag}.ply $MD pr_${sn}_${tag} $gpu > eval/pr_${sn}_${tag}.log 2>&1 &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "=== results ==="
for spec in truck:b4 truck:b6 truck:b12 room:b4 room:b6 room:b12 garden:b4 garden:b6 garden:b12 \
            truck:sn_cover room:sn_cover garden:sn_cover; do
  IFS=: read sn tag <<< "$spec"
  echo "$sn $tag: $(grep -o '"PSNR": [0-9.]*' eval/pr_${sn}_${tag}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/pr_${sn}_${tag}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/pr_${sn}_${tag}.log|tail -1)"
done
echo PLAN_REST_DONE
