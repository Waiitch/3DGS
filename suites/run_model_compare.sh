#!/usr/bin/env bash
# Evaluate every available truck/room model with OUR eval script + OUR GT, so the
# numbers are mutually comparable (the reported paper numbers use a different
# renderer/protocol and are NOT comparable to our table).
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
R=/mnt/home1/wyq/3dgs
CSV=timing/model_compare.csv
[ -f "$CSV" ] || echo "scene,method,dir,ply_mb,n_gauss,psnr,ssim,lpips" > "$CSV"
# scene | method label | model dir (relative to $R)
SPECS="
truck|Vanilla 3DGS (ours)|mpl_las/out/truck_vanilla
truck|Vanilla 3DGS (original)|original_out/truck
truck|LAS|out/truck_las_code
truck|LAS (paper cfg)|out/truck_las_paper
truck|MP-LAS|out/truck_mplas_fixed
truck|MP-LAS (spec)|out/truck_mplas_spec
truck|abl d0707+o2|out/truck3_abl_d0707_o2
truck|Pixel-GS|pixelgs_out/truck
truck|FreGS|fregs_out/truck_fra
room|Vanilla 3DGS (ours)|mpl_las/out/room_vanilla
room|LAS|out/room_las_code
room|abl d0707+o2|out/room_abl_d0707_o2
room|Vanilla 3DGS (v2)|out/room_vanilla2
"
echo "$SPECS" | while IFS='|' read -r SC METH REL; do
  [ -z "$SC" ] && continue
  D="$R/$REL"
  PLY="$D/point_cloud/iteration_30000/point_cloud.ply"
  [ -f "$PLY" ] || { echo "  [$SC/$METH] no ply"; continue; }
  TAG=$(echo "${SC}_${METH}" | tr -c 'A-Za-z0-9_\n' '_')
  if grep -q "^$SC,$METH," "$CSV" 2>/dev/null; then echo "  [$SC/$METH] cached"; continue; fi
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  MB=$($PY -c "print(f'{$(stat -c%s $PLY)/1048576:.0f}')")
  NG=$($PY -c "
import struct,sys
f=open('$PLY','rb')
n=0
while True:
    line=f.readline()
    if not line: break
    s=line.decode('ascii','ignore').strip()
    if s.startswith('element vertex'): n=int(s.split()[-1])
    if s=='end_header': break
print(n)")
  bash $EV "$PLY" "$D" "$TAG" "$GPU" > /tmp/mc_${TAG}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' /tmp/mc_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  ss=$(grep -oE '"SSIM": [0-9.]+' /tmp/mc_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  lp=$(grep -oE '"LPIPS": [0-9.]+' /tmp/mc_${TAG}.log|tail -1|grep -oE '[0-9.]+')
  [ -z "$p" ] && { echo "  [$SC/$METH] eval FAILED"; tail -2 /tmp/mc_${TAG}.log; continue; }
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$SC" "$METH" "$REL" "$MB" "$NG" "$p" "$ss" "$lp" >> "$CSV"
  echo "  [$SC/$METH] PSNR=$p gauss=$NG ply=${MB}MB"
done
echo MC_DONE
