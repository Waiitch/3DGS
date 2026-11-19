#!/usr/bin/env bash
# (a) complete the lever sweep on bicycle
# (b) validate the final config on a 5th scene (playroom)
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
printf 'scene,arm,e2e,o2,final\n' > timing/bike5.csv
run() {
  sn=$1; tag=$2; shift 2
  case $sn in
    bicycle)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla;  SC=/mnt/home1/wyq/3dgs/data/bicycle;;
    playroom) MD=/mnt/home1/wyq/3dgs/mpl_las/out/playroom_vanilla; SC=/mnt/home1/wyq/3dgs/data/playroom;;
  esac
  IN=$MD/point_cloud/iteration_30000/point_cloud.ply
  T=b5_${sn}_${tag}
  $PY -u run_o2.py --ply $IN -o out/${T}.ply --ratio 0.1 \
    --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
    --o2-scene $SC $BASE "$@" --json /tmp/${T}.json > /tmp/${T}.log 2>&1
  rc=$?
  $PY -c "
import json;d=json.load(open('/tmp/${T}.json'));s=d['stage_s']
print('${sn},${tag},%.3f,%.3f,%d'%(d['end_to_end_s'],s.get('o2',0),d['final']))" >> timing/bike5.csv
  echo "  run $T rc=$rc"
}
# (a) bicycle
run bicycle s10    --o2-sigma-cut 1.0
run bicycle g11    --o2-grid 11
run bicycle g13    --o2-grid 13
run bicycle s125g13 --o2-sigma-cut 1.25 --o2-grid 13
run bicycle l03    --sh-lam 0.03
run bicycle l10    --sh-lam 0.1
run bicycle d00    --sh-min-diff 0.0
run bicycle d10    --sh-min-diff 0.1
run bicycle d35    --sh-min-diff 0.35
# (b) 5th scene: base vs final
run playroom base  --o2-sigma-cut 2.0
run playroom final --o2-sigma-cut 1.25
cat timing/bike5.csv
echo "=== evals (3-way) ==="
i=0
for spec in "bicycle:s10" "bicycle:g11" "bicycle:g13" "bicycle:s125g13" \
            "bicycle:l03" "bicycle:l10" "bicycle:d00" "bicycle:d10" "bicycle:d35" \
            "playroom:base" "playroom:final"; do
  IFS=: read sn tag <<< "$spec"
  case $sn in
    bicycle)  MD=/mnt/home1/wyq/3dgs/mpl_las/out/bicycle_vanilla;;
    playroom) MD=/mnt/home1/wyq/3dgs/mpl_las/out/playroom_vanilla;;
  esac
  gpu=$(( i % 3 )); [ $gpu -eq 2 ] && gpu=3
  bash $EV out/b5_${sn}_${tag}.ply $MD b5_${sn}_${tag} $gpu > eval/b5_${sn}_${tag}.log 2>&1 &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "=== results (bicycle baseline cover8/s2.0 = 21.0326) ==="
for spec in "bicycle:s10" "bicycle:g11" "bicycle:g13" "bicycle:s125g13" \
            "bicycle:l03" "bicycle:l10" "bicycle:d00" "bicycle:d10" "bicycle:d35" \
            "playroom:base" "playroom:final"; do
  IFS=: read sn tag <<< "$spec"
  echo "$sn $tag: $(grep -o '"PSNR": [0-9.]*' eval/b5_${sn}_${tag}.log|tail -1) $(grep -o '"SSIM": [0-9.]*' eval/b5_${sn}_${tag}.log|tail -1) $(grep -o '"LPIPS": [0-9.]*' eval/b5_${sn}_${tag}.log|tail -1)"
done
echo BIKE5_DONE
