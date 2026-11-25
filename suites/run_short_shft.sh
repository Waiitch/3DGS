#!/usr/bin/env bash
# 10-second SH-only rendering-loss fine-tune on OUR compressed truck model.
# geometry frozen (script default) + opacity frozen (--opacity-lr 0) => SH only.
set -o pipefail
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
SRC=/mnt/home1/wyq/nanogs-work/o2gpu5x/out/dc3_M2c.ply     # our best truck model
SCENE=/mnt/home1/wyq/3dgs/data/truck
REF=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla          # for cfg_args / cameras
WORK=/mnt/home1/wyq/shortft/truck_m2c
rm -rf "$WORK"; mkdir -p "$WORK/point_cloud/iteration_30000"
cp "$SRC" "$WORK/point_cloud/iteration_30000/point_cloud.ply"
cp "$REF/cfg_args" "$WORK/" 2>/dev/null
cp "$REF/cameras.json" "$WORK/" 2>/dev/null
cp "$REF/input.ply" "$WORK/" 2>/dev/null
echo "=== staged $(basename $SRC) -> $WORK ==="

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
export CUDA_VISIBLE_DEVICES=$GPU

for IT in 1000 2000; do
  T0=$(date +%s)
  $PY -u /mnt/home1/wyq/ghap-work/finetune_appearance.py \
    -s "$SCENE" -m "$WORK" \
    --ft-iterations $IT --start-iteration 30000 \
    --out-iteration $((30000 + IT / 10)) \
    --opacity-lr 0 > /tmp/sft_${IT}.log 2>&1
  RC=$?; T1=$(date +%s)
  OUT=$WORK/point_cloud/iteration_$((30000 + IT / 10))/point_cloud.ply
  if [ $RC -ne 0 ] || [ ! -f "$OUT" ]; then
    echo "  [SH-ft $IT] FAIL"; tail -5 /tmp/sft_${IT}.log; continue
  fi
  bash $EV "$OUT" "$REF" "shft_${IT}" "$GPU" > /tmp/sft_ev_${IT}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' /tmp/sft_ev_${IT}.log|tail -1|grep -oE '[0-9.]+')
  ss=$(grep -oE '"SSIM": [0-9.]+' /tmp/sft_ev_${IT}.log|tail -1|grep -oE '[0-9.]+')
  lp=$(grep -oE '"LPIPS": [0-9.]+' /tmp/sft_ev_${IT}.log|tail -1|grep -oE '[0-9.]+')
  echo "  [SH-ft ${IT} iters] PSNR=$p SSIM=$ss LPIPS=$lp  wall=$((T1-T0))s"
done
echo SFT_DONE
