#!/usr/bin/env bash
# Re-run the repo's headline A/B arms at the *recommended* SH config (clip 0.05).
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
SH="--sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05"
printf 'scene,arm,e2e,o2,final\n' > timing/main_c05.csv
run() {
  sn=$1; arm=$2; shift 2
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;  SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;   SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=m05_${sn}_${arm}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-scene $SC --o2-batch 32768 --o2-cams fast \
    --o2-views 8 $SH "$@" --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s']
print('${sn},${arm},%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/main_c05.csv
  echo "  run $T done"
}
for sn in truck room garden; do
  run $sn base_sn
  run $sn p2c1   --cost-plane 2.0 --cost-cg 1.0
  run $sn e4     --cost-plane 2.0 --cost-cg 1.0 --cost-e4 0.5
done
run truck rnd --cost-plane 2.0 --cost-cg 1.0 --cost-random 1
cat timing/main_c05.csv
echo "=== evals (2-way parallel) ==="
i=0
for spec in "truck:base_sn" "truck:p2c1" "truck:e4" "truck:rnd" \
            "room:base_sn" "room:p2c1" "room:e4" \
            "garden:base_sn" "garden:p2c1" "garden:e4"; do
  IFS=: read sn arm <<< "$spec"
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla;;
  esac
  gpu=$(( i % 2 == 0 ? 1 : 3 ))
  bash $EV out/m05_${sn}_${arm}.ply $MD m05_${sn}_${arm} $gpu > eval/m05_${sn}_${arm}.log 2>&1 &
  i=$((i+1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
echo "=== results ==="
for spec in "truck:base_sn" "truck:p2c1" "truck:e4" "truck:rnd" \
            "room:base_sn" "room:p2c1" "room:e4" \
            "garden:base_sn" "garden:p2c1" "garden:e4"; do
  IFS=: read sn arm <<< "$spec"
  echo "$sn $arm: $(grep -o '"PSNR": [0-9.]*' eval/m05_${sn}_${arm}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/m05_${sn}_${arm}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/m05_${sn}_${arm}.log|tail -1)"
done
echo MAIN_C05_DONE
