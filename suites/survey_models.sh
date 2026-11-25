#!/usr/bin/env bash
# Robust survey: every trained truck/room model on the server, with PLY size,
# sh_degree and the method family it belongs to.
for root in /mnt/home1/wyq/3dgs/original_out /mnt/home1/wyq/3dgs/igs_out \
            /mnt/home1/wyq/3dgs/pixelgs_out /mnt/home1/wyq/3dgs/fregs_out \
            /mnt/home1/wyq/3dgs/mpl_las/out /mnt/home1/wyq/3dgs/out \
            /mnt/home1/wyq/3dgs/_compare3; do
  [ -d "$root" ] || continue
  for d in "$root"/*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    case "$n" in *truck*|*room*) ;; *) continue;; esac
    ply="$d/point_cloud/iteration_30000/point_cloud.ply"
    it=30000
    if [ ! -f "$ply" ]; then ply="$d/point_cloud/iteration_7000/point_cloud.ply"; it=7000; fi
    [ -f "$ply" ] || continue
    sz=$(stat -c%s "$ply" 2>/dev/null)
    sh=$(grep -o "sh_degree=[0-9]" "$d/cfg_args" 2>/dev/null | head -1)
    [ -z "$sh" ] && sh=$(grep -o "'sh_degree': [0-9]" "$d/cfg_args" 2>/dev/null | head -1)
    printf "%s|%s|%s|%s|%s\n" "$(basename $(dirname $d))" "$n" "$it" "$sz" "$sh"
  done
done | sort -u
