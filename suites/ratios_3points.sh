#!/usr/bin/env bash
# same-protocol table part A: our method at rho = 0.5 / 0.1 / 0.05 on truck + room
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
SH="--sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05"
P2="--cost-plane 2.0 --cost-cg 1.0"
printf 'scene,rho,arm,e2e,o2,final\n' > timing/ratios.csv
run() { sn=$1; rho=$2; tag=$3; MD=$4; SC=$5; shift 5
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=rt_${sn}_${rho}_${tag}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio $rho \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --o2-scene $SC --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-batch 32768 $P2 $SH "$@" \
    --json results/${T}.json > results/${T}.log 2>&1
  $PY -q -c "
import json
d=json.load(open('results/${T}.json')); s=d['stage_s']
print('${sn},${rho},${tag},%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/ratios.csv
  echo "  run $T done"
}
TR=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; TRS=/mnt/home1/wyq/3dgs/data/truck
RM=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;  RMS=/mnt/home1/wyq/3dgs/data/room
for sn in truck room; do
  case $sn in
    truck) MD=$TR; SC=$TRS;;
    room)  MD=$RM; SC=$RMS;;
  esac
  for rho in 0.5 0.1 0.05; do
    run $sn $rho ours $MD $SC --o2-cams cover --o2-views 8 --o2-sigma-cut 1.25
  done
done
echo "=== runs done ==="; cat timing/ratios.csv
echo "=== evals ==="
i=0
for spec in truck:0.5 truck:0.1 truck:0.05 room:0.5 room:0.1 room:0.05; do
  IFS=: read sn rho <<< "$spec"
  case $sn in
    truck) MD=$TR;; room) MD=$RM;;
  esac
  gpu=$(( i % 2 == 0 ? 1 : 3 ))
  bash $EV out/rt_${sn}_${rho}_ours.ply $MD rt_${sn}_${rho}_ours $gpu > eval/rt_${sn}_${rho}_ours.log 2>&1 &
  i=$((i+1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
echo "=== results ==="
for spec in truck:0.5 truck:0.1 truck:0.05 room:0.5 room:0.1 room:0.05; do
  IFS=: read sn rho <<< "$spec"
  echo "$sn rho=$rho: $(grep -o 'PSNR": [0-9.]*' eval/rt_${sn}_${rho}_ours.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/rt_${sn}_${rho}_ours.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/rt_${sn}_${rho}_ours.log | tail -1)"
done
echo RATIOS_DONE
