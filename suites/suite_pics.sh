#!/usr/bin/env bash
# PICS end-to-end suite: fixed pair sequence, only the O2cal camera rule changes.
#   arms: fast8 (control) | cover8 (historical) | repair8 | repair12
# Mirrors suite_o2cover.sh so numbers are comparable with the earlier reports.
set -o pipefail
W=/mnt/home1/wyq/nanogs-work/o2gpu5x
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
cd $W || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
mkdir -p timing results eval
printf 'scene,arm,rep,rc,e2e,o2,o2pick,views,zero,final\n' > timing/pics.csv

run () {
  local sn=$1 IN=$2 SC=$3 arm=$4 rep=$5 cams=$6 extra=$7
  local tag=pc_${sn}_${arm}_${rep}
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
print('${sn},${arm},${rep},$rc,%.3f,%.3f,%.3f,%.2f,%.3f,%d'%(
  d['end_to_end_s'], s.get('o2',0) or 0, s.get('o2_pick',0) or 0, v, z, d['final']))" >> timing/pics.csv
  echo "  $sn $arm rep=$rep rc=$rc"
}

ARMS="fast cover repair8 repair12"
for sn in truck room garden; do
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;  SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  for rep in 1 2; do
    run $sn $IN $SC fast     $rep fast    ""
    run $sn $IN $SC cover    $rep cover   ""
    run $sn $IN $SC repair8  $rep pair    "--o2-pick repair --o2-views 8"
    run $sn $IN $SC repair12 $rep pair    "--o2-pick repair --o2-views 12"
  done
  # evaluate the rep-1 outputs only (the pipeline is deterministic)
  for arm in $ARMS; do
    bash $EV out/pc_${sn}_${arm}_1.ply $MD pc_${sn}_${arm} 3 > eval/pc_${sn}_${arm}.log 2>&1
    echo "$sn $arm: $(grep -o '"PSNR": [0-9.]*' eval/pc_${sn}_${arm}.log | tail -1) \
$(grep -o '"SSIM": [0-9.]*' eval/pc_${sn}_${arm}.log | tail -1) \
$(grep -o '"LPIPS": [0-9.]*' eval/pc_${sn}_${arm}.log | tail -1)"
  done
done
echo "PICS_SUITE_DONE"
cat timing/pics.csv
