#!/usr/bin/env bash
# Round 2: corrected paths under mpl_las/out, plus the 7k checkpoint of every
# model (the init state that prune+finetune pipelines start from).
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
R=/mnt/home1/wyq/3dgs
CSV=timing/model_compare.csv
SPECS="
truck|LAS|mpl_las/out/truck_las_code
truck|LAS (paper)|mpl_las/out/truck_las_paper
truck|MP-LAS|mpl_las/out/truck_mplas_fixed
truck|MP-LAS (spec)|mpl_las/out/truck_mplas_spec
truck|Abl d0707+o2|mpl_las/out/truck3_abl_d0707_o2
room|LAS|mpl_las/out/room_las_code
room|Abl d0707+o2|mpl_las/out/room_abl_d0707_o2
room|Vanilla (v2)|mpl_las/out/room_vanilla2
"
echo "$SPECS" | while IFS='|' read -r SC METH REL; do
  [ -z "$SC" ] && continue
  D="$R/$REL"
  for IT in 30000 7000; do
    PLY="$D/point_cloud/iteration_$IT/point_cloud.ply"
    [ -f "$PLY" ] || { [ "$IT" = "30000" ] && echo "  [$SC/$METH] no ply"; continue; }
    M2="$METH @${IT}"
    [ "$IT" = "7000" ] && M2="$METH (7k ckpt)"
    grep -q "^$SC,$M2," "$CSV" 2>/dev/null && { echo "  [$SC/$M2] cached"; continue; }
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
    MB=$($PY -c "print(f'{$(stat -c%s $PLY)/1048576:.0f}')")
    NG=$($PY -c "
f=open('$PLY','rb'); n=0
while True:
    line=f.readline()
    if not line: break
    s=line.decode('ascii','ignore').strip()
    if s.startswith('element vertex'): n=int(s.split()[-1])
    if s=='end_header': break
print(n)")
    TAG=$(echo "${SC}_${METH}_${IT}" | tr -c 'A-Za-z0-9_\n' '_')
    bash $EV "$PLY" "$D" "$TAG" "$GPU" > /tmp/mc2_${TAG}.log 2>&1
    p=$(grep -oE '"PSNR": [0-9.]+' /tmp/mc2_${TAG}.log|tail -1|grep -oE '[0-9.]+')
    ss=$(grep -oE '"SSIM": [0-9.]+' /tmp/mc2_${TAG}.log|tail -1|grep -oE '[0-9.]+')
    lp=$(grep -oE '"LPIPS": [0-9.]+' /tmp/mc2_${TAG}.log|tail -1|grep -oE '[0-9.]+')
    [ -z "$p" ] && { echo "  [$SC/$M2] eval FAILED"; tail -2 /tmp/mc2_${TAG}.log; continue; }
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$SC" "$M2" "$REL@$IT" "$MB" "$NG" "$p" "$ss" "$lp" >> "$CSV"
    echo "  [$SC/$M2] PSNR=$p gauss=$NG ply=${MB}MB"
  done
done
echo MC2_DONE
