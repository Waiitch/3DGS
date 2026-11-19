#!/usr/bin/env bash
# Phase B: push sigma_cut lower on truck, then confirm the winners on room/garden.
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
run() {
  sn=$1; tag=$2; shift 2
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;  SC=/mnt/home1/wyq/3dgs/data/truck;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;   SC=/mnt/home1/wyq/3dgs/data/room;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla; SC=/mnt/home1/wyq/3dgs/data/garden;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=lv2_${sn}_${tag}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --o2-scene $SC $BASE "$@" --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  echo "  run $T done"
}
run truck s10  --o2-sigma-cut 1.0
run truck s125 --o2-sigma-cut 1.25
run truck s15g13 --o2-sigma-cut 1.5 --o2-grid 13
run room  s15  --o2-sigma-cut 1.5
run room  s15g13 --o2-sigma-cut 1.5 --o2-grid 13
run garden s15 --o2-sigma-cut 1.5
run garden s15g13 --o2-sigma-cut 1.5 --o2-grid 13
echo "=== evals (3-way) ==="
i=0
for spec in "truck:s10" "truck:s125" "truck:s15g13" "room:s15" "room:s15g13" "garden:s15" "garden:s15g13"; do
  IFS=: read sn tag <<< "$spec"
  case $sn in
    truck)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)   MD=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
    garden) MD=/mnt/home1/wyq/3dgs/mpl_las/out/garden_vanilla;;
  esac
  gpu=$(( i % 3 )); [ $gpu -eq 2 ] && gpu=3
  bash $EV out/lv2_${sn}_${tag}.ply $MD lv2_${sn}_${tag} $gpu > eval/lv2_${sn}_${tag}.log 2>&1 &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "=== phase B ==="
for spec in "truck:s10" "truck:s125" "truck:s15g13" "room:s15" "room:s15g13" "garden:s15" "garden:s15g13"; do
  IFS=: read sn tag <<< "$spec"
  echo "$sn $tag: $(grep -o '"PSNR": [0-9.]*' eval/lv2_${sn}_${tag}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/lv2_${sn}_${tag}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/lv2_${sn}_${tag}.log|tail -1)"
done
echo LEVER12_B_DONE
