#!/usr/bin/env bash
# REFINE 后期微调计时：把 REFINE 的输出放到 iteration_30000 路径下（脚本要求），
# 用 --first_iter 30001 从 30000 之后开始，5000 步到 35000（与 PUP 的 refine 同 schedule）。
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
source /mnt/home1/wyq/3dgs/miniconda3/etc/profile.d/conda.sh
conda activate pup
export CUDA_HOME=/usr/local/cuda-11.8; export PATH=$CUDA_HOME/bin:$PATH
B=/mnt/home1/wyq/gaussian-splatting-pup/submodules
export PYTHONPATH=/mnt/home1/wyq/shims:$B/simple-knn/build/lib.linux-x86_64-cpython-310:$B/diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-310:$B/rasterization_and_pup_fisher/build/lib.linux-x86_64-cpython-310:$B/compress-diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-310
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
cd /mnt/home1/wyq/gaussian-splatting-pup || exit 1
R=/mnt/home1/wyq/nanogs-work/o2gpu5x/refine_out
printf 'tag,gaussians,iterations,seconds,sec_per_1k,psnr,ssim,lpips\n' > /mnt/home1/wyq/nanogs-work/o2gpu5x/timing/refine_ft.csv
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh

ft () {  # tag source_ply
  tag=$1; SRC=$2
  M=./experiments/$tag
  rm -rf $M; mkdir -p $M/point_cloud/iteration_30000
  cp -f /mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla/cameras.json $M/cameras.json 2>/dev/null
  cp -f /mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla/cfg_args    $M/cfg_args 2>/dev/null
  cp -f "$SRC" $M/point_cloud/iteration_30000/point_cloud.ply
  n=$(head -c 2000 "$SRC" | tr -d '\0' | grep -m1 'element vertex' | awk '{print $3}')
  echo "=== fine-tune $tag  N=$n  5000 it ==="
  t0=$(date +%s)
  python prune_finetune.py -s /mnt/home1/wyq/3dgs/data/truck -m $M --eval \
    --prune_iterations 999999 --prune_percent 0.0 \
    --start_pointcloud $M/point_cloud/iteration_30000/point_cloud.ply \
    --first_iter 30001 \
    --iterations 35000 --position_lr_max_steps 35000 --save_iterations 35000 \
    --checkpoint_iterations 0 --test_iterations 0 --port 6080 \
    > logs_ft_${tag}.log 2>&1
  rc=$?
  t1=$(date +%s); sec=$((t1 - t0))
  its=$(grep -oE "[0-9.]+it/s" logs_ft_${tag}.log | tail -1)
  echo "  rc=$rc  ${sec}s（含启动）  最后速率=$its"
  f=$(find $M/point_cloud -name point_cloud.ply | tail -1)
  bash $EV "$f" /mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla "$tag" 1 \
      > /mnt/home1/wyq/nanogs-work/o2gpu5x/eval/$tag.log 2>&1
  p=$(grep -o 'PSNR": [0-9.]*' /mnt/home1/wyq/nanogs-work/o2gpu5x/eval/$tag.log | tail -1 | sed 's/.*: //')
  s=$(grep -o 'SSIM": [0-9.]*' /mnt/home1/wyq/nanogs-work/o2gpu5x/eval/$tag.log | tail -1 | sed 's/.*: //')
  l=$(grep -o 'LPIPS": [0-9.]*' /mnt/home1/wyq/nanogs-work/o2gpu5x/eval/$tag.log | tail -1 | sed 's/.*: //')
  echo "  $tag: PSNR=$p SSIM=$s LPIPS=$l"
  echo "$tag,$n,5000,$sec,$(python -c "print('%.2f'%(${sec}/5.0))"),$p,$s,$l" \
      >> /mnt/home1/wyq/nanogs-work/o2gpu5x/timing/refine_ft.csv
}
ft rft_truck_k10 $R/refine_truck_k10/point_cloud.ply
ft rft_truck_k50 $R/refine_truck_k50/point_cloud.ply
echo FT_DONE
