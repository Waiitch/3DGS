#!/usr/bin/env bash
# Plan 13.4: coverage-optimised O2cal fit views vs the default even subsample.
set -o pipefail
W=/mnt/home1/wyq/nanogs-work/o2gpu5x
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
cd $W || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
printf 'scene,arm,rc,e2e,o2_s,cams_s,final\n' > timing/o2cover.csv

run () {
  local sn=$1 IN=$2 SC=$3 tag=$4 cam=$5
  $PY -u run_o2.py --ply $IN -o out/oc_${sn}_${tag}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 --o2-cams $cam \
    --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --cost-plane 2.0 --cost-cg 1.0 \
    --json /tmp/oc_${sn}_${tag}.json > /tmp/oc_${sn}_${tag}.log 2>&1
  local rc=$?
  $PY -c "
import json;d=json.load(open('/tmp/oc_${sn}_${tag}.json'));s=d['stage_s']
print('${sn},${tag},$rc,%.3f,%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0) or 0,s.get('o2_cams',0) or 0,d['final']))" >> timing/o2cover.csv
  echo "  $sn $tag rc=$rc"
}

for sn in truck room garden; do
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;  SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  run $sn $IN $SC fast  fast
  run $sn $IN $SC cover cover
done
cat timing/o2cover.csv
echo "===== evals ====="
for sn in truck room garden; do
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla;;
  esac
  for tag in fast cover; do
    bash $EV out/oc_${sn}_${tag}.ply $MD oc_${sn}_${tag} 2 > eval/oc_${sn}_${tag}.log 2>&1
    echo "$sn $tag: $(grep -o '"PSNR": [0-9.]*' eval/oc_${sn}_${tag}.log | tail -1) $(grep -o '"SSIM": [0-9.]*' eval/oc_${sn}_${tag}.log | tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/oc_${sn}_${tag}.log | tail -1)"
  done
done
echo "O2COVER_DONE"
