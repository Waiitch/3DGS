#!/usr/bin/env bash
# Run the OFFICIAL NanoGS simplify() CLI on truck/room at r=0.1, time it, eval it.
cd /mnt/home1/wyq/nanogs-work/gpu5x/official || exit 1
export PYTHONPATH=/mnt/home1/wyq/nanogs-work/gpu5x/official:/mnt/home1/wyq/nanogs-work/kl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
M=/mnt/home1/wyq/3dgs/mpl_las/out
CSV=/mnt/home1/wyq/nanogs-work/o2gpu5x/timing/official_nanogs_compare.csv
[ -f "$CSV" ] || echo "scene,method,ratio,gauss,ply_mb,psnr,ssim,lpips,time_s" > "$CSV"

for sn in truck room; do
  IN=$M/${sn}_vanilla/point_cloud/iteration_30000/point_cloud.ply
  OUT=/tmp/official_${sn}_r01.ply
  echo "===== 官方 NanoGS: $sn r=0.1 ====="
  T0=$(date +%s.%N)
  $PY -u -m nanogs.simplification -i "$IN" -o "$OUT" -r 0.1 > /tmp/off_${sn}.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  SEC=$(python3 -c "print(f'{$T1-$T0:.1f}')")
  echo "  rc=$RC  wall=$SEC s"
  tail -3 /tmp/off_${sn}.log | cut -c1-120
  [ $RC -ne 0 ] && continue
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d " ")
  NG=$($PY -c "
f=open('$OUT','rb'); n=0
while True:
    l=f.readline()
    if not l: break
    s=l.decode('ascii','ignore').strip()
    if s.startswith('element vertex'): n=int(s.split()[-1])
    if s=='end_header': break
print(n)")
  MB=$($PY -c "print(f'{$(stat -c%s $OUT)/1048576:.1f}')")
  bash $EV "$OUT" "$M/${sn}_vanilla" "official_${sn}_r01" "$GPU" > /tmp/offeval_${sn}.log 2>&1
  p=$(grep -oE '"PSNR": [0-9.]+' /tmp/offeval_${sn}.log|tail -1|grep -oE '[0-9.]+')
  ss=$(grep -oE '"SSIM": [0-9.]+' /tmp/offeval_${sn}.log|tail -1|grep -oE '[0-9.]+')
  lp=$(grep -oE '"LPIPS": [0-9.]+' /tmp/offeval_${sn}.log|tail -1|grep -oE '[0-9.]+')
  printf '%s,Official NanoGS (no ft),0.10,%s,%s,%s,%s,%s,%s\n' "$sn" "$NG" "$MB" "$p" "$ss" "$lp" "$SEC" >> $CSV
  echo "  PSNR=$p SSIM=$ss LPIPS=$lp gauss=$NG wall=$SEC s"
done
echo OFFICIAL_DONE
