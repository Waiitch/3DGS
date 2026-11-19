#!/usr/bin/env bash
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
printf 'scene,clip,e2e,o2,final\n' > timing/clipsweep.csv
for spec in "truck:0.02" "truck:0.10" "truck:0.20" "room:0.02" "room:0.10" "room:0.20"; do
  IFS=: read sn cl <<< "$spec"
  case $sn in
    truck) MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;  SC=/mnt/home1/wyq/3dgs/data/room;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=cl_${sn}_${cl}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 --o2-cams fast --o2-views 8 \
    --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip $cl \
    --cost-plane 2.0 --cost-cg 1.0 --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s']
print('${sn},${cl},%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/clipsweep.csv
  echo "  $T done"
done
for spec in "truck:0.02" "truck:0.10" "truck:0.20" "room:0.02" "room:0.10" "room:0.20"; do
  IFS=: read sn cl <<< "$spec"
  case $sn in
    truck) MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
  esac
  bash $EV out/cl_${sn}_${cl}.ply $MD cl_${sn}_${cl} 3 > eval/cl_${sn}_${cl}.log 2>&1
  echo "$sn clip=$cl: $(grep -o '"PSNR": [0-9.]*' eval/cl_${sn}_${cl}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/cl_${sn}_${cl}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/cl_${sn}_${cl}.log|tail -1)"
done
echo CLIPSWEEP_DONE
