#!/usr/bin/env bash
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
printf 'scene,arm,rep,e2e,o2,o2pick,cams,views,final\n' > timing/track.csv
run() {
  sn=$1; B=$2; rep=$3
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;  SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;   SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
    bicycle) MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla; SC=/mnt/home1/wyq/3dgs/data/bicycle;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=tk_${sn}_${B}_${rep}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 \
    --o2-cams pair --o2-pick track --o2-views $B \
    --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --cost-plane 2.0 --cost-cg 1.0 \
    --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s'];ps=d.get('passes',[])
v=sum(p.get('o2_sel_views',0) for p in ps)/max(len(ps),1)
print('${sn},${B},${rep},%.3f,%.3f,%.3f,%.3f,%.2f,%d'%(d['end_to_end_s'],s.get('o2',0),s.get('o2_pick',0),s.get('o2_cams',0),v,d['final']))" >> timing/track.csv
  echo "  $sn B=$B rep=$rep"
}
for sn in truck room garden; do for B in 6 8; do for rep in 1 2; do run $sn $B $rep; done; done; done
for sn in truck room garden; do
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla;;
  esac
  for B in 6 8; do
    bash $EV out/tk_${sn}_${B}_1.ply $MD tk_${sn}_${B} 2 > eval/tk_${sn}_${B}.log 2>&1
    echo "$sn B=$B: $(grep -o '"PSNR": [0-9.]*' eval/tk_${sn}_${B}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/tk_${sn}_${B}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/tk_${sn}_${B}.log|tail -1)"
  done
done
echo TRACK_SUITE_DONE
