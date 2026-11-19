#!/usr/bin/env bash
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla
SC=/mnt/home1/wyq/3dgs/data/bicycle
IN=$MD/point_cloud/iteration_30000/point_cloud.ply
printf 'scene,arm,e2e,o2,o2pick,cams,views,final\n' > timing/bicycle.csv
for arm in fast guard; do
  case $arm in
    fast)  CAM="--o2-cams fast --o2-views 8";;
    guard) CAM="--o2-cams pair --o2-pick guard --o2-views 8";;
  esac
  T=bk_${arm}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 $CAM \
    --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --cost-plane 2.0 --cost-cg 1.0 \
    --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s'];ps=d.get('passes',[])
v=sum(p.get('o2_sel_views',0) for p in ps)/max(len(ps),1)
print('${arm},%.3f,%.3f,%.3f,%.3f,%.2f,%d'%(d['end_to_end_s'],s.get('o2',0),s.get('o2_pick',0),s.get('o2_cams',0),v,d['final']))" >> timing/bicycle.csv
  echo "  bike $arm done"
done
for arm in fast guard; do
  CUDA_VISIBLE_DEVICES=0 bash $EV out/bk_${arm}.ply $MD bk_${arm} 0 > eval/bk_${arm}.log 2>&1
  echo "bike $arm: $(grep -o '"PSNR": [0-9.]*' eval/bk_${arm}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/bk_${arm}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/bk_${arm}.log|tail -1)"
done
echo BICYCLE_DONE
