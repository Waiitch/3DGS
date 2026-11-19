#!/usr/bin/env bash
# evaluate the SH-reduced PLYs: needs a cfg_args with sh_degree = 2 / 0
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
for sn in truck room; do
  SRC=/mnt/home1/wyq/3dgs/mpl_las/out/${sn}_vanilla
  for d in 2 0; do
    D=/mnt/home1/wyq/3dgs/mpl_las/out/${sn}_sh${d}
    mkdir -p $D
    sed "s/sh_degree=3/sh_degree=${d}/" $SRC/cfg_args > $D/cfg_args
    echo "$sn sh$d cfg: $(grep -o 'sh_degree=[0-9]' $D/cfg_args)"
  done
done
i=0
for spec in "truck:0.1:2" "truck:0.05:2" "room:0.1:2" "room:0.05:2" \
            "truck:0.1:0" "truck:0.05:0" "room:0.1:0" "room:0.05:0"; do
  IFS=: read sn rho d <<< "$spec"
  D=/mnt/home1/wyq/3dgs/mpl_las/out/${sn}_sh${d}
  gpu=$(( i % 2 == 0 ? 1 : 3 ))
  bash $EV out/rt_${sn}_${rho}_ours_sh${d}.ply $D rt_${sn}_${rho}_sh${d} $gpu \
      > eval/rt_${sn}_${rho}_sh${d}.log 2>&1 &
  i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
for spec in "truck:0.1:2" "truck:0.05:2" "room:0.1:2" "room:0.05:2" \
            "truck:0.1:0" "truck:0.05:0" "room:0.1:0" "room:0.05:0"; do
  IFS=: read sn rho d <<< "$spec"
  echo "$sn rho=$rho SH$d: $(grep -o 'PSNR": [0-9.]*' eval/rt_${sn}_${rho}_sh${d}.log | tail -1) $(grep -o 'SSIM": [0-9.]*' eval/rt_${sn}_${rho}_sh${d}.log | tail -1) $(grep -o 'LPIPS": [0-9.]*' eval/rt_${sn}_${rho}_sh${d}.log | tail -1)"
done
echo SH_EVAL_DONE
