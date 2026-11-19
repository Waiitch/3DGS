#!/usr/bin/env bash
# PUP without the co-adaptation step (prune at 30001, save at 30002): isolates
# the pruning criterion from the 5000-iteration refinement.
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
source /mnt/home1/wyq/3dgs/miniconda3/etc/profile.d/conda.sh
conda activate pup
export CUDA_HOME=/usr/local/cuda-11.8; export PATH=$CUDA_HOME/bin:$PATH
B=/mnt/home1/wyq/gaussian-splatting-pup/submodules
export PYTHONPATH=/mnt/home1/wyq/shims:$B/simple-knn/build/lib.linux-x86_64-cpython-310:$B/diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-310:$B/rasterization_and_pup_fisher/build/lib.linux-x86_64-cpython-310:$B/compress-diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-310
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
cd /mnt/home1/wyq/gaussian-splatting-pup || exit 1
run () {  # scene prune tag
  sn=$1; pc=$2; tag=$3
  case $sn in
    truck) DATA=/mnt/home1/wyq/3dgs/data/truck; ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)  DATA=/mnt/home1/wyq/3dgs/data/room;  ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
  esac
  M=./experiments/${tag}
  mkdir -p $M/point_cloud/iteration_30000
  cp -f $ORIG/cameras.json $M/cameras.json 2>/dev/null
  cp -f $ORIG/cfg_args    $M/cfg_args 2>/dev/null
  echo "=== PUP-norefine $sn prune=$pc ==="
  python prune_finetune.py -s $DATA -m $M --eval \
    --prune_percent $pc --prune_type fisher \
    --start_pointcloud $ORIG/point_cloud/iteration_30000/point_cloud.ply \
    --iterations 30002 --position_lr_max_steps 30002 --save_iterations 30002 \
    --checkpoint_iterations 0 --test_iterations 0 \
    --fisher_resolution 4 --port 6072 --prune_iterations 30001 \
    > logs_pupnr_${tag}.log 2>&1
  echo "  rc=$? -> $(ls $M/point_cloud/ 2>/dev/null | tr '\n' ' ')"
}
run truck 0.5  pupnr_truck_k50
run truck 0.9  pupnr_truck_k10
run truck 0.95 pupnr_truck_k05
echo "=== eval ==="
EV=/mnt/home1/wyq/nanogs-work/kl/eval_simplified.sh
P=/mnt/home1/wyq/gaussian-splatting-pup/experiments
for t in pupnr_truck_k50 pupnr_truck_k10 pupnr_truck_k05; do
  f=$P/$t/point_cloud/iteration_30002/point_cloud.ply
  [ -f "$f" ] || f=$(find $P/$t/point_cloud -name point_cloud.ply | tail -1)
  bash $EV "$f" /mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla "$t" 3 > /mnt/home1/wyq/nanogs-work/o2gpu5x/eval/$t.log 2>&1
  echo "$t: $(grep -o 'PSNR": [0-9.]*' /mnt/home1/wyq/nanogs-work/o2gpu5x/eval/$t.log | tail -1) $(grep -o 'SSIM": [0-9.]*' /mnt/home1/wyq/nanogs-work/o2gpu5x/eval/$t.log | tail -1)"
done
echo PUPNR_DONE
