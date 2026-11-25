#!/usr/bin/env bash
cd /mnt/home1/wyq/ghap-work || exit 1
echo "===== 各目录是否有 iteration_45000（微调后）====="
for p in ft_truck/* ft_room/*; do
  [ -d "$p" ] || continue
  f45="$p/point_cloud/iteration_45000/point_cloud.ply"
  f30="$p/point_cloud/iteration_30000/point_cloud.ply"
  if [ -f "$f45" ]; then
    printf "%-40s 45000:%10s B\n" "$p" "$(stat -c%s "$f45")"
  elif [ -f "$f30" ]; then
    printf "%-40s (only 30000: %s B)\n" "$p" "$(stat -c%s "$f30")"
  fi
done
echo
echo "===== 所有微调耗时记录 ====="
grep -h "\[ft\] saved" logs/ft_*.log 2>/dev/null | sed 's/.*saved //' | sort -u | head -30
echo
echo "===== 迭代速度 ====="
grep -h "\[ft\] 15000/15000" logs/ft_*.log 2>/dev/null | sort -u | head -20
