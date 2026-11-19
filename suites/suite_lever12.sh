#!/usr/bin/env bash
# Sweep the two untested levers on top of the final config
#   final = clip 0.05 + p2c1 + cover8
#   lever 1: --o2-grid (5/7/9/11/13) and --o2-sigma-cut (1.5/2.0/2.5/3.0)
#   lever 2: --sh-lam (0.003/0.01/0.03/0.1) and --sh-min-diff (0/0.1/0.2/0.35)
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
BASE="--scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
 --cost-plane 2.0 --cost-cg 1.0 --opacity-mode O2cal --o2-batch 32768 \
 --o2-cams cover --o2-views 8 \
 --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05 \
 --o2-grid 9 --o2-sigma-cut 2.0"
printf 'scene,arm,e2e,o2,final\n' > timing/lever12.csv
run() {
  sn=$1; tag=$2; shift 2
  case $sn in
    truck)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;  SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)    MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;   SC=/mnt/home1/wyq/3dgs/data/room;;
    garden)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=lv_${sn}_${tag}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --o2-scene $SC $BASE "$@" --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s']
print('${sn},${tag},%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/lever12.csv
  echo "  run $T done"
}
# ---- phase A: truck, one variable at a time ------------------------------
run truck g5   --o2-grid 5
run truck g7   --o2-grid 7
run truck g11  --o2-grid 11
run truck g13  --o2-grid 13
run truck s15  --o2-sigma-cut 1.5
run truck s25  --o2-sigma-cut 2.5
run truck s30  --o2-sigma-cut 3.0
run truck l003 --sh-lam 0.003
run truck l03  --sh-lam 0.03
run truck l10  --sh-lam 0.1
run truck d00  --sh-min-diff 0.0
run truck d10  --sh-min-diff 0.1
run truck d35  --sh-min-diff 0.35
cat timing/lever12.csv
echo "=== evals (3-way parallel) ==="
i=0
for tag in g5 g7 g11 g13 s15 s25 s30 l003 l03 l10 d00 d10 d35; do
  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla
  gpu=$(( i % 3 ))
  [ $gpu -eq 2 ] && gpu=3
  bash $EV out/lv_truck_${tag}.ply $MD lv_truck_${tag} $gpu > eval/lv_truck_${tag}.log 2>&1 &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "=== truck results (baseline cover8 = 21.5544) ==="
for tag in g5 g7 g11 g13 s15 s25 s30 l003 l03 l10 d00 d10 d35; do
  echo "$tag: $(grep -o '"PSNR": [0-9.]*' eval/lv_truck_${tag}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/lv_truck_${tag}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/lv_truck_${tag}.log|tail -1)"
done
echo LEVER12_A_DONE
