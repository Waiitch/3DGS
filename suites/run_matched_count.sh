#!/usr/bin/env bash
# Matched-gaussian-count comparison at 10% (truck ~205.5k, room ~131.2k).
# Every number is produced by OUR eval script on OUR GT split.
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
G=/mnt/home1/wyq/ghap-work
CSV=timing/o2_matched_count.csv
[ -f "$CSV" ] || echo "scene,method,dir,gauss,ply_mb,psnr,ssim,lpips" > "$CSV"
SPECS="
truck|Original (prune 10%)|$G/ft_truck/truck10_orig
truck|Original + fine-tune|$G/ft_truck/truck10_orig_full
truck|NanoGS|$G/ft_truck/truck10_nanogs_bc
truck|NanoGS + fine-tune|$G/ft_truck/truck10_nanogs_bc_full
truck|GHAP|$G/ft_truck/truck10_ghap
truck|GHAP + fine-tune|$G/ft_truck/truck10_ghap_full
truck|O2cal (archived)|$G/ft_truck/truck10_o2cal
room|NanoGS|$G/ft_room/room10_nanogs_bc
room|NanoGS + fine-tune|$G/ft_room/room10_nanogs_bc_full
room|GHAP|$G/ft_room/room10_ghap
room|GHAP + fine-tune|$G/ft_room/room10_ghap_full
"
echo "$SPECS" | while IFS='|' read -r SC METH D; do
  [ -z "$SC" ] && continue
  PLY="$D/point_cloud/iteration_30000/point_cloud.ply"
  [ -f "$PLY" ] || { echo "  [$SC/$METH] no ply"; continue; }
  grep -q "^$SC,$METH," "$CSV" 2>/dev/null && { echo "  [$SC/$METH] cached"; continue; }
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  NG=$($PY -c "
f=open('$PLY','rb'); n=0
while True:
    l=f.readline()
    if not l: break
    s=l.decode('ascii','ignore').strip()
    if s.startswith('element vertex'): n=int(s.split()[-1])
    if s=='end_header': break
print(n)")
  MB=$($PY -c "print(f'{$(stat -c%s $PLY)/1048576:.1f}')")
  TAG=$(echo "${SC}_${METH}" | tr -c 'A-Za-z0-9_\n' '_')
  bash $EV "$PLY" "$D" "$TAG" "$GPU" > /tmp/mt_${TAG}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' /tmp/mt_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  ss=$(grep -oE '"SSIM": [0-9.]+' /tmp/mt_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  lp=$(grep -oE '"LPIPS": [0-9.]+' /tmp/mt_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  [ -z "$p" ] && { echo "  [$SC/$METH] eval FAILED"; tail -3 /tmp/mt_${TAG}.log; continue; }
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$SC" "$METH" "$D" "$NG" "$MB" "$p" "$ss" "$lp" >> "$CSV"
  echo "  [$SC/$METH] gauss=$NG PSNR=$p SSIM=$ss LPIPS=$lp"
done
echo MT_DONE
