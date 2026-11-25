#!/usr/bin/env bash
# Phase 3 / S2: dual-domain joint pool.
#   S2-f1: dual_domain with the SAME 6-axis headings as T0-C3 (pool-only ablation)
#   S2-f2: dual_domain + gravity 0,+-15 headings
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
D=/mnt/home1/wyq/3dgs/data
CSV=timing/av_v3_explore_raw.csv
CAMS=results/av/cams
mkdir -p "$CAMS" eval results/av
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' '; }
model_of() { case $1 in
  truck) echo truck_vanilla;; room) echo room_vanilla;; garden) echo garden_vanilla;;
  bicycle) echo bicycle_vanilla;; playroom) echo playroom_vanilla;;
  train) echo tandt_train_vanilla;; drjohnson) echo drjohnson_vanilla;; esac; }
COMMON="--ratio 0.10 --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1
        --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 --o2-batch 32768
        --o2-views 8 --o2-sigma-cut 1.25 --cost-plane 2.0 --cost-cg 1.0
        --opacity-mode O2cal --sh-clip 0.05"
C3="--av-core-aabb 0.80 --av-no-scale-gate --av-cover-target gauss --av-gauss-probe 20000"
EXPS="
S2f1|$C3 --av-pos-mode dual_domain
S2f2|$C3 --av-pos-mode dual_domain --av-dir-mode gravity --av-pitches 0,-15,15
"
SC=${1:-truck,room,garden,bicycle,playroom,train,drjohnson}
echo "$EXPS" | while IFS='|' read -r EXP FLAGS; do
  [ -z "$EXP" ] && continue
  H=$(printf '%s' "$FLAGS" | md5sum | cut -c1-8)
  for sn in ${SC//,/ }; do
    RID="v4_${EXP}_${sn}_s0_${H}"
    grep -q "^${RID}," "$CSV" 2>/dev/null && { echo "  [$RID] skip"; continue; }
    md=$(model_of $sn); GR=$(pick_gpu); T0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GR $PY -u run_o2.py \
      --ply $M/$md/point_cloud/iteration_30000/point_cloud.ply \
      -o out/${RID}.ply $COMMON --sh-mode s0 --o2-scene $D/$sn \
      --o2-cams auto_virt $FLAGS --o2-dump-cams $CAMS/${RID}.json \
      --json results/av/${RID}.json > results/av/${RID}.log 2>&1
    rc=$?; T1=$(date +%s)
    if [ $rc -ne 0 ]; then
      printf '%s,S2,%s,%s,s0,0.10,8,"%s",FAIL,FAIL,FAIL,,,,0,,%s\n' \
        "$RID" "$EXP" "$sn" "$FLAGS" "$((T1-T0))" >> "$CSV"
      echo "  [$RID] FAIL"; tail -3 results/av/${RID}.log; continue; fi
    bash $EV out/${RID}.ply $M/$md "$RID" $GR > eval/${RID}.log 2>&1
    p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    s=$(grep -oE '"SSIM": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    l=$(grep -oE '"LPIPS": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
    split=$(grep -oE 'split=[0-9]+/[0-9]+' results/av/${RID}.log | head -1)
    printf '%s,S2,%s,%s,s0,0.10,8,"%s",%s,%s,%s,,,,0,%s,%s\n' \
      "$RID" "$EXP" "$sn" "$FLAGS" "$p" "$s" "$l" "$CAMS/${RID}.json" "$((T1-T0))" >> "$CSV"
    echo "  [$RID] PSNR=$p $split"
  done
done
echo S2_DONE
