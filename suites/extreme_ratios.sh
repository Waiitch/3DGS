#!/usr/bin/env bash
# 鏋佺鍘嬬缉鐜囦笅閲嶆祴褰撳勾"鏁堟灉澶皬琚渷鎺?鐨勯€夎竟/鍏遍潰鍑嗗垯
#   rho = 0.5 (鍙悎骞朵竴鍗? 鍜?0.01 (250:1)
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
set -o pipefail
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
SH="--sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05"
P2="--cost-plane 2.0 --cost-cg 1.0"
TR=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla; TRS=/mnt/home1/wyq/3dgs/data/truck
printf 'rho,arm,e2e,final\n' > timing/extreme.csv

run () {
  rho=$1; tag=$2; shift 2
  T=ex_${rho}_${tag}
  $PY -u run_o2.py --ply $TR/point_cloud/iteration_30000/point_cloud.ply -o out/${T}.ply \
    --ratio $rho --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --o2-scene $TRS --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
    --opacity-mode O2cal --o2-batch 32768 --o2-cams cover --o2-views 8 --o2-sigma-cut 1.25 \
    $SH "$@" --json results/${T}.json > results/${T}.log 2>&1
  rc=$?
  $PY -q -c "
import json
d=json.load(open('results/${T}.json'))
print('${rho},${tag},%.2f,%d'%(d['end_to_end_s'], d['final']))" >> timing/extreme.csv 2>/dev/null
  echo "  $T rc=$rc"
}
for rho in 0.5 0.01; do
  run $rho base   $P2
  run $rho plain
  run $rho n05    $P2 --cost-normal 0.5
  run $rho n10    $P2 --cost-normal 1.0
  run $rho ax10   $P2 --cost-axis 1.0
  run $rho sr10   $P2 --cost-sratio 1.0
  run $rho e4     $P2 --cost-e4 0.5
  run $rho kl     $P2 --scorer kl
  run $rho maha   $P2 --scorer maha_fast
done
echo "=== runs done ==="; cat timing/extreme.csv
echo "=== evals (3-way) ==="
i=0
for rho in 0.5 0.01; do
  for tag in base plain n05 n10 ax10 sr10 e4 kl maha; do
    gpu=$(( i % 3 )); [ $gpu -eq 2 ] && gpu=3
    bash $EV out/ex_${rho}_${tag}.ply $TR ex_${rho}_${tag} $gpu > eval/ex_${rho}_${tag}.log 2>&1 &
    i=$((i+1)); if [ $((i % 3)) -eq 0 ]; then wait; fi
  done
done
wait
echo "=== results ==="
for rho in 0.5 0.01; do
  for tag in base plain n05 n10 ax10 sr10 e4 kl maha; do
    echo "rho=$rho $tag: $(grep -o 'PSNR": [0-9.]*' eval/ex_${rho}_${tag}.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/ex_${rho}_${tag}.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/ex_${rho}_${tag}.log | tail -1)"
  done
done
echo EXTREME_DONE
