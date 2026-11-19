#!/usr/bin/env bash
# PICS end-to-end suite, round 2: after the one-time-cost fixes, with the
# smaller budgets that the offline budget sweep showed to be sufficient.
#   arms: fast8 (control) | repair6 | repair8 | repair12
# Same pair sequence for every arm (the scorer does not see the opacity), same
# final Gaussian count, same evaluator as round 1.
set -o pipefail
W=/mnt/home1/wyq/nanogs-work/o2gpu5x
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
cd $W || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${RUN_GPU:-0}
mkdir -p timing results eval
printf 'scene,arm,rep,rc,e2e,o2,o2pick,cams,views,zero,final\n' > timing/pics2.csv

run () {
  local sn=$1 IN=$2 SC=$3 arm=$4 rep=$5 cams=$6 extra=$7
  local tag=p2_${sn}_${arm}_${rep}
  $PY -u run_o2.py --ply $IN -o out/${tag}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 --o2-cams $cams $extra \
    --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --cost-plane 2.0 --cost-cg 1.0 \
    --json /tmp/${tag}.json > /tmp/${tag}.log 2>&1
  local rc=$?
  $PY -c "
import json
d=json.load(open('/tmp/${tag}.json')); s=d['stage_s']; ps=d.get('passes',[])
v=sum(p.get('o2_sel_views',0) for p in ps)/max(len(ps),1)
z=sum(p.get('o2_sel_zero',0) for p in ps)/max(len(ps),1)
print('${sn},${arm},${rep},$rc,%.3f,%.3f,%.3f,%.3f,%.2f,%.3f,%d'%(
  d['end_to_end_s'], s.get('o2',0) or 0, s.get('o2_pick',0) or 0,
  s.get('o2_cams',0) or 0, v, z, d['final']))" >> timing/pics2.csv
  echo "  $sn $arm rep=$rep rc=$rc"
}

for sn in truck room garden; do
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;  SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  for rep in 1 2 3; do
    run $sn $IN $SC fast     $rep fast    ""
    run $sn $IN $SC repair6  $rep pair    "--o2-pick repair --o2-views 6"
    run $sn $IN $SC repair8  $rep pair    "--o2-pick repair --o2-views 8"
  done
  for arm in fast repair6 repair8; do
    bash $EV out/p2_${sn}_${arm}_1.ply $MD p2_${sn}_${arm} ${EVAL_GPU:-2} > eval/p2_${sn}_${arm}.log 2>&1
    echo "$sn $arm: $(grep -o '"PSNR": [0-9.]*' eval/p2_${sn}_${arm}.log | tail -1) \
$(grep -o '"SSIM": [0-9.]*' eval/p2_${sn}_${arm}.log | tail -1) \
$(grep -o '"LPIPS": [0-9.]*' eval/p2_${sn}_${arm}.log | tail -1)"
  done
done
echo "PICS_SUITE2_DONE"
cat timing/pics2.csv
