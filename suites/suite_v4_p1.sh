#!/usr/bin/env bash
# Phase 1 batch: P1-C (T0-C3 dc on 7 scenes) + P1-B (bicycle forced-outdoor control)
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

run_one() {  # track exp scene shmode flags
  local TRACK=$1 EXP=$2 SN=$3 SHM=$4 FLAGS=$5
  local RID="p1_${TRACK}_${EXP}_${SN}_${SHM}_$(printf '%s' "$FLAGS" | md5sum | cut -c1-8)"
  grep -q "^${RID}," "$CSV" 2>/dev/null && { echo "  [$RID] skip"; return; }
  local md=$(model_of $SN) GR=$(pick_gpu) T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GR $PY -u run_o2.py \
    --ply $M/$md/point_cloud/iteration_30000/point_cloud.ply \
    -o out/${RID}.ply $COMMON --sh-mode $SHM --o2-scene $D/$SN \
    --o2-cams auto_virt $FLAGS --o2-dump-cams $CAMS/${RID}.json \
    --json results/av/${RID}.json > results/av/${RID}.log 2>&1
  local rc=$? T1=$(date +%s)
  if [ $rc -ne 0 ]; then
    printf '%s,%s,%s,%s,%s,0.10,8,"%s",FAIL,FAIL,FAIL,,,,0,,%s\n' \
      "$RID" "$TRACK" "$EXP" "$SN" "$SHM" "$FLAGS" "$((T1-T0))" >> "$CSV"
    echo "  [$RID] FAIL"; tail -3 results/av/${RID}.log; return; fi
  bash $EV out/${RID}.ply $M/$md "$RID" $GR > eval/${RID}.log 2>&1
  local p=$(grep -oE '"PSNR": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
  local s=$(grep -oE '"SSIM": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
  local l=$(grep -oE '"LPIPS": [0-9.]+' eval/${RID}.log|tail -1|grep -oE '[0-9.]+')
  printf '%s,%s,%s,%s,%s,0.10,8,"%s",%s,%s,%s,,,,0,%s,%s\n' \
    "$RID" "$TRACK" "$EXP" "$SN" "$SHM" "$FLAGS" "$p" "$s" "$l" "$CAMS/${RID}.json" "$((T1-T0))" >> "$CSV"
  echo "  [$RID] PSNR=$p"
}

# --- P1-C: T0-C3 in dc mode on all 7 scenes ---
for sn in truck room garden bicycle playroom train drjohnson; do
  run_one P1C dc $sn dc "$C3"
done
# --- P1-B: bicycle natural vs forced outdoor ---
run_one P1B forced_outdoor bicycle s0 "$C3 --av-cav6 999999"
echo P1_DONE
