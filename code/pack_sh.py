"""Rewrite a 3DGS PLY with a lower SH degree, and report packed sizes.

The papers we compare against (LightGaussian / FlexGaussian / PUP / REFINE)
report the size of their *quantized* representation, while our pipeline writes
plain float32 SH-3 PLYs (248 B / gaussian).  This script produces PLYs at a
lower SH degree so the renders (and therefore PSNR) can be measured for that
representation, and prints the size of several packing formats so the size axis
is defined explicitly instead of being inferred.

Usage:
  python pack_sh.py --in out/a.ply --out out/a_sh2.ply --sh-degree 2
"""
from __future__ import annotations

import argparse
import numpy as np

SH3 = 45          # f_rest coefficients for degree 3
BYTES = {"fp32": 4, "fp16": 2}


def read_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            header.append(line.decode("ascii", "replace"))
            if line.strip() == b"end_header":
                break
        props, count = [], 0
        for line in header:
            t = line.split()
            if not t:
                continue
            if t[0] == "element" and t[1] == "vertex":
                count = int(t[2])
            elif t[0] == "property" and t[1] != "list":
                props.append((t[2], t[1]))
        dtype = [(n, "<f4" if t == "float" else "<u1") for n, t in props]
        data = np.fromfile(f, dtype=np.dtype(dtype), count=count)
    return header, props, data


def packed_bytes(n_props_by_kind, dtype):
    b = 0
    for name, cnt in n_props_by_kind.items():
        b += cnt * BYTES[dtype]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--sh-degree", type=int, default=2, choices=[0, 1, 2, 3])
    args = ap.parse_args()

    header, props, data = read_ply(args.src)
    n = len(data)
    names = [p[0] for p in props]
    keep_rest = 3 * ((args.sh_degree + 1) ** 2 - 1)   # 0 / 9 / 24 / 45
    rest = [f"f_rest_{i}" for i in range(45)]
    keep = [nm for nm in names if nm not in rest[:0]]
    keep = []
    for nm in names:
        if nm.startswith("f_rest_"):
            if int(nm.split("_")[-1]) < keep_rest:
                keep.append(nm)
        else:
            keep.append(nm)

    dt = [(nm, "<f4") for nm in keep]
    out = np.zeros(n, dtype=np.dtype(dt))
    for nm in keep:
        out[nm] = data[nm]

    # header in 3DGS order
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for nm in keep:
        lines.append(f"property float {nm}")
    lines.append("end_header")
    with open(args.dst, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        out.tofile(f)

    # ---- size report ----------------------------------------------------
    base = {"mu": 3, "sc": 3, "rot": 4, "op": 1, "dc": 3}
    def size(sh_rest, dt):
        d = dict(base); d["rest"] = sh_rest
        return sum(c * BYTES[dt] for c in d.values())
    sh3_rest = 45
    sh2_rest = 3 * 9 - 3          # 24
    sh0_rest = 0
    print(f"[pack] in={args.src}  N={n:,}  ->  out={args.dst}  SH degree {args.sh_degree}")
    print(f"[pack] written PLY: {np.dtype(dt).itemsize * n / 1048576:.2f} MB (fp32, SH{args.sh_degree})")
    print("[pack] bytes / gaussian and total size under different packings:")
    for label, mode in (("fp32 SH3", None), ("fp16 SH3", None),
                        ("fp16 SH2", None), ("fp16 SH0", None)):
        rest_c = {"fp32 SH3": sh3_rest, "fp16 SH3": sh3_rest,
                  "fp16 SH2": sh2_rest, "fp16 SH0": sh0_rest}[label]
        prec = "fp32" if label.startswith("fp32") else "fp16"
        b = size(rest_c, prec)
        print(f"   {label:9s}: {b:3d} B/gauss -> {b * n / 1048576:8.2f} MB")


if __name__ == "__main__":
    main()
