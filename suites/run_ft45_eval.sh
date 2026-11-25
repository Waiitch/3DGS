#!/usr/bin/env bash
# Verify the POST-fine-tune numbers by re-evaluating iteration_45000 ourselves.
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
G=/mnt/home1/wyq/ghap-work
CSV=timing/o2_matched_count.csv
for spec in "truck|NanoGS (post-ft)|$G/ft_truck/truck10_nanogs_bc" \
            "room|NanoGS (post-ft)|$G/ft_room/room10_nanogs_bc" \
            "truck|GHAP (post-ft)|$G/ft_truck/truck10_ghap" \
            "room|GHAP (post-ft)|$G/ft_room/room10_ghap" \
            "truck|Original (post-ft)|$G/ft_truck/truck10_orig"; do
  IFS='|' read -r SC METH D <<< "$spec"
  PLY="$D/point_cloud/iteration_45000/point_cloud.ply"
  [ -f "$PLY" ] || { echo "  [$SC/$METH] no 45k ply"; continue; }
  grep -q "^$SC,$METH," "$CSV" 2>/dev/null && { echo "  [$SC/$METH] cached"; continue; }
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  TAG=$(echo "${SC}_${METH}45" | tr -c 'A-Za-z0-9_\n' '_')
  bash $EV "$PLY" "$D" "$TAG" "$GPU" > /tmp/ft45_${TAG}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' /tmp/ft45_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  ss=$(grep -oE '"SSIM": [0-9.]+' /tmp/ft45_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  lp=$(grep -oE '"LPIPS": [0-9.]+' /tmp/ft45_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  [ -z "$p" ] && { echo "  [$SC/$METH] FAILED"; tail -2 /tmp/ft45_${TAG}.log; continue; }
  printf '%s,%s,%s,0,0,%s,%s,%s\n' "$SC" "$METH" "$D@45000" "$p" "$ss" "$lp" >> $CSV
  echo "  [$SC/$METH @45k] PSNR=$p SSIM=$ss LPIPS=$lp"
done
echo FT45_DONE
