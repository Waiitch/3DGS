#!/usr/bin/env bash
# part 1: SH-degree reduction so the size axis is comparable with the papers
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
TR=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla
RM=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla
for spec in "truck:0.1" "truck:0.05" "room:0.1" "room:0.05"; do
  IFS=: read sn rho <<< "$spec"
  src=out/rt_${sn}_${rho}_ours.ply
  for d in 2 0; do
    dst=out/rt_${sn}_${rho}_ours_sh${d}.ply
    $PY pack_sh.py --in $src --out $dst --sh-degree $d 2>&1 | tail -6
  done
done
echo "=== evals ==="
i=0
for spec in "truck:0.1:2" "truck:0.05:2" "room:0.1:2" "room:0.05:2" \
            "truck:0.1:0" "truck:0.05:0" "room:0.1:0" "room:0.05:0"; do
  IFS=: read sn rho d <<< "$spec"
  case $sn in truck) MD=$TR;; room) MD=$RM;; esac
  gpu=$(( i % 2 == 0 ? 1 : 3 ))
  bash $EV out/rt_${sn}_${rho}_ours_sh${d}.ply $MD rt_${sn}_${rho}_sh${d} $gpu \
      > eval/rt_${sn}_${rho}_sh${d}.log 2>&1 &
  i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
for spec in "truck:0.1:2" "truck:0.05:2" "room:0.1:2" "room:0.05:2" \
            "truck:0.1:0" "truck:0.05:0" "room:0.1:0" "room:0.05:0"; do
  IFS=: read sn rho d <<< "$spec"
  echo "$sn rho=$rho SH$d: $(grep -o 'PSNR": [0-9.]*' eval/rt_${sn}_${rho}_sh${d}.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/rt_${sn}_${rho}_sh${d}.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/rt_${sn}_${rho}_sh${d}.log | tail -1)"
done
echo SH_PACK_DONE
