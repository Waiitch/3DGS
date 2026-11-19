#!/usr/bin/env bash
# same-protocol baseline row: the plain merge (O0 opacity, fast cameras, no O2cal)
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
P2="--cost-plane 2.0 --cost-cg 1.0"
TR=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; TRS=/mnt/home1/wyq/3dgs/data/truck
RM=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;  RMS=/mnt/home1/wyq/3dgs/data/room
for sn in truck room; do
  case $sn in
    truck) MD=$TR; SC=$TRS;;
    room)  MD=$RM; SC=$RMS;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  for rho in 0.5 0.1 0.05; do
    T=rt_${sn}_${rho}_base
    $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio $rho \
      --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
      --o2-scene $SC --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
      --opacity-mode O0 --o2-cams fast --o2-views 8 $P2 \
      --json results/${T}.json > results/${T}.log 2>&1
    $PY -q -c "
import json
d=json.load(open('results/${T}.json')); s=d['stage_s']
print('${sn},${rho},base,%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/ratios.csv
    echo "  run $T done"
  done
done
i=0
for spec in truck:0.5 truck:0.1 truck:0.05 room:0.5 room:0.1 room:0.05; do
  IFS=: read sn rho <<< "$spec"
  case $sn in truck) MD=$TR;; room) MD=$RM;; esac
  gpu=$(( i % 2 == 0 ? 1 : 3 ))
  bash $EV out/rt_${sn}_${rho}_base.ply $MD rt_${sn}_${rho}_base $gpu > eval/rt_${sn}_${rho}_base.log 2>&1 &
  i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
echo "=== base results ==="
for spec in truck:0.5 truck:0.1 truck:0.05 room:0.5 room:0.1 room:0.05; do
  IFS=: read sn rho <<< "$spec"
  echo "$sn rho=$rho base: $(grep -o 'PSNR": [0-9.]*' eval/rt_${sn}_${rho}_base.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/rt_${sn}_${rho}_base.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/rt_${sn}_${rho}_base.log | tail -1)"
done
echo BASE_DONE
