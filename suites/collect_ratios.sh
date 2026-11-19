#!/usr/bin/env bash
cd /mnt/home1/wyq/nanogs-work/o2gpu5x || exit 1
PY=/mnt/home1/wyq/nanogs-work/venv/bin/python
$PY - <<'PYEOF'
import glob, os, re, json

def n_gauss(p):
    with open(p, 'rb') as f:
        hdr = b''
        while True:
            line = f.readline()
            hdr += line
            if line.strip() == b'end_header' or not line or len(hdr) > 40000:
                break
        m = re.search(rb'element vertex (\d+)', hdr)
        return int(m.group(1)) if m else -1

def metrics(log):
    if not os.path.exists(log):
        return None
    t = open(log, errors='ignore').read()
    out = {}
    for k, pat in (("psnr", r'"PSNR":\s*([0-9.]+)'), ("ssim", r'"SSIM":\s*([0-9.]+)'),
                   ("lpips", r'"LPIPS":\s*([0-9.]+)')):
        m = re.findall(pat, t)
        out[k] = float(m[-1]) if m else None
    return out

rows = []
for log in sorted(glob.glob("eval/rt_*.log")):
    stem = os.path.basename(log)[:-4]          # rt_<scene>_<rho>_<arm>
    parts = stem.split("_")
    if len(parts) < 4:
        continue
    scene, rho, arm = parts[1], parts[2], parts[3]
    m = metrics(log)
    ply = "out/%s.ply" % stem
    ng = n_gauss(ply) if os.path.exists(ply) else -1
    mb = os.path.getsize(ply) / 1048576 if os.path.exists(ply) else 0
    rows.append((scene, rho, arm, ng, mb, m))

order = {"truck": 0, "room": 1}
rows.sort(key=lambda r: (order.get(r[0], 9), float(r[1]), r[2]))
print("%-6s %-5s %-5s %10s %9s %8s %8s %8s" %
      ("scene", "rho", "arm", "gaussians", "ply_MB", "PSNR", "SSIM", "LPIPS"))
for scene, rho, arm, ng, mb, m in rows:
    if not m or m["psnr"] is None:
        print("%-6s %-5s %-5s %10d %9.1f %8s" % (scene, rho, arm, ng, mb, "running"))
        continue
    print("%-6s %-5s %-5s %10d %9.1f %8.4f %8.5f %8.5f"
          % (scene, rho, arm, ng, mb, m["psnr"], m["ssim"], m["lpips"]))
PYEOF
