#!/usr/bin/env bash
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
printf 'scene,arm,rep,e2e,o2,o2pick,cams,views,final\n' > timing/time3.csv
for sn in truck room garden; do
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;  SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  for rep in 1 2 3; do
    for arm in fast cover guard repair8; do
      case $arm in
        fast)   CAM="--o2-cams fast --o2-views 8";;
        cover)  CAM="--o2-cams cover --o2-views 8";;
        guard)  CAM="--o2-cams pair --o2-pick guard --o2-views 8";;
        repair8) CAM="--o2-cams pair --o2-pick repair --o2-views 8";;
      esac
      T=tt_${sn}_${arm}_${rep}
      $PY -u run_o2.py --ply $IN -o /tmp/${T}.ply --ratio 0.1 \
        --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
        --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
        --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 $CAM \
        --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --cost-plane 2.0 --cost-cg 1.0 \
        --json /tmp/${T}.json > /tmp/${T}.log 2>&1
      $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s'];ps=d.get('passes',[])
v=sum(p.get('o2_sel_views',0) for p in ps)/max(len(ps),1)
print('${sn},${arm},${rep},%.3f,%.3f,%.3f,%.3f,%.2f,%d'%(d['end_to_end_s'],s.get('o2',0),s.get('o2_pick',0),s.get('o2_cams',0),v,d['final']))" >> timing/time3.csv
    done
    echo "  $sn rep=$rep done"
  done
done
echo TIME3_DONE
