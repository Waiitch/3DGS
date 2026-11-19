# 3dgs — O2cal 视角采样 / 透明度重拟合压缩

3D Gaussian Splatting 的**后处理压缩**工具：在固定高斯预算下，把上百万高斯合并成目标数量，
并用「逐对视角采样 + 屏幕覆盖重拟合」把合并后的透明度校准回去。
**零训练**（不重新优化任何参数），单场景 PLY→PLY 端到端约 4–8 秒。

- 相机视角**逐对（per-merge-pair）自适应选择**：每个合并从全部相机里挑 8 个最能"看见"它的视角
- 透明度 `a` 由投影覆盖 patches 的最小二乘重拟合：`a = Σ G_m·T_p / Σ G_m²`
- DC 球谐校正（clip 0.05）修合并带来的颜色偏移
- 与"剪枝类"方法（REFINE / PUP 3D-GS 等）同口径对比见 `docs/REPORT-pics-e2e.md`

## 主要结果

同 base 模型、同评测脚本（官方 test split）、同 PLY 格式（SH3/fp32）、同"保留率 = 剩余高斯/源高斯"：

### 相对朴素合并（nanogs 基线，关掉 O2cal）

| ρ | truck | room |
|---:|---:|---:|
| 0.50 | +0.002 dB | +0.006 dB |
| 0.10 | **+1.862 dB** | **+1.976 dB** |
| 0.05 | **+1.864 dB** | **+1.857 dB** |

### 相对其它方法（零训练，即"微调前"）

| ρ | 场景 | 本方法 | REFINE | PUP 3D-GS（仅剪枝） |
|---:|---|---:|---:|---:|
| 0.10 | truck | **21.58** | 16.34 | 14.76 |
| 0.05 | truck | **20.56** | 14.35 | 12.95 |
| 0.01 | truck | **18.46** | 11.69 | — |
| 0.10 | room | **27.32** | 21.37 | — |
| 0.01 | room | **22.67** | 14.80 | — |

耗时（PLY→PLY，不含评测）：本方法 **4.2–7.8 s**；REFINE 0.7 s；PUP 约 40–60 s（+5k 步微调约 200 s）。

## 环境

- Python 3.10、PyTorch 2.0.1+cu117、CUDA 11.8
- 需要 nano-gs 官方实现（提供 `nanogs.utils.ply_utils` / `nanogs.utils.merge` / `nanogs.simplification`）：
  ```bash
  export PYTHONPATH=/path/to/nano-gs-official:/path/to/eval_scripts
  ```
- 评测脚本（`eval_simplified.sh`）来自官方 3DGS 仓库的 `render.py` + `metrics.py`

## 快速开始

```bash
python code/run_o2.py \
  --ply /path/to/model/point_cloud/iteration_30000/point_cloud.ply \
  -o out/compressed.ply --ratio 0.1 \
  --cost-device gpu --merge-device gpu --edges-device gpu --fast-io 1 \
  --scorer scale_norm --no-blas 1 --jacobi 1 --knn-tuned 1 \
  --cost-plane 2.0 --cost-cg 1.0 \
  --opacity-mode O2cal --o2-scene /path/to/colmap/scene --o2-batch 32768 \
  --o2-cams cover --o2-views 8 --o2-sigma-cut 1.25 \
  --sh-mode dc --sh-sum 1 --sh-min-diff 0.2 --sh-lam 0.01 --sh-clip 0.05 \
  --json out/run.json
```

关键参数（详见 `docs/REPORT-pics-e2e.md` §11）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ratio` | 0.1 | 保留高斯比例 |
| `--o2-cams` | fast | `cover` 全局覆盖选视角（推荐）、`pair` 逐对自适应 |
| `--o2-pick` | — | `guard`/`track`/`dir`/`sup`/`all` 等逐对策略 |
| `--o2-sigma-cut` | 2.0 | **1.25 在四个场景都更好且零成本** |
| `--o2-grid` | 9 | patch 采样网格（13 只多 +0.01~0.02 dB，时间 +78%） |
| `--sh-clip` | 0.30 | **0.05 是正解**（0.30 会掉 0.2–0.8 dB） |
| `--cost-plane` / `--cost-cg` | 0 / 0 | 2.0 / 1.0 = p2c1 选边，+0.08 dB |
| `--cost-axis` | 0 | ρ≤0.02 时开 1.0（+0.05~+0.20 dB） |

## 文件结构

```
code/
  run_o2.py        主流程：knn → 边代价 → 贪心配对 → 合并 → O2cal 重拟合 → SH 校正
  pics.py          PICS 内核（逐对相机 screen/选择、patch 统计、透明度重拟合）
  o2gpu.py         单相机集版本的 O2cal 内核（参考实现）
  shfix.py         DC 球谐校正
  gpulib.py        GPU 基础算子（knn/贪心/合并）
  fastio.py        快速 PLY 读写
  localdensity.py  局部密度（ρ 归一化）
  cams_fast.py     相机加载与 cover 选视角
  cams_track.py    COLMAP points3D 轨迹重叠矩阵
  pack_sh.py       SH 降阶 + 打包尺寸分析
  o2eval*.py       离线评估（E_alpha / held-out / pair-split / 预算 / 全局方法）
  export_cams.py / verify_pool.py   辅助工具
suites/            可复现实验的 shell 驱动（见 docs 报告各节的"复现"）
docs/              实验报告
results/           各实验的 CSV（PSNR/SSIM/LPIPS/耗时）
```

## 复现

```bash
bash suites/suite_pics.sh        # 相机臂端到端
bash suites/suite_time3.sh       # 干净计时
bash suites/suite_clip05.sh      # clip=0.05 下的相机臂
bash suites/suite_main_c05.sh    # 主表复核（p2c1/E4/随机）
bash suites/suite_lever12.sh     # grid/sigma/lam/min_diff 扫描
bash suites/ratios_3points.sh    # 压缩率 0.5/0.1/0.05
bash suites/baseline_refine.sh   # REFINE 零训练基线
bash suites/baseline_pup.sh      # PUP 3D-GS 基线
bash suites/extreme_ratios.sh    # 极端压缩率 0.5/0.01 下的选边准则
```

## 已知事项

- **方向**：ρ≥0.5 时本方法与朴素合并无差别（+0.002 dB）；优势随压缩变狠而增大，ρ≤0.1 时 +1.86~1.98 dB。
- **`track` 选择曾有一个实现 bug**（冗余惩罚按元素掩码导致失效），已修复；旧端到端 track 数字作废，详见报告 §12.1。
- **SH 不能朴素截断**：SH3→SH2 会掉 0.7–1.4 dB，尺寸轴对比需要蒸馏而不是截断（报告 §16.5）。
- 评测务必在干净环境里跑：若 `PYTHONPATH` 里残留 PUP 的 `diff_gaussian_rasterization`，
  标准渲染器会报 `unexpected keyword argument 'antialiasing'`，指标全 NaN。

## 许可

研究代码，未附带上游 3DGS / nano-gs 的许可。使用前请遵守相应上游仓库的许可条款。
