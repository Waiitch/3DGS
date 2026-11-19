#!/usr/bin/env bash
# PUP 3D-GS baseline: prune + 5000-iteration co-adaptation at three keep ratios
export TMPDIR=/mnt/home1/wyq/tmp TMP=/mnt/home1/wyq/tmp TEMP=/mnt/home1/wyq/tmp
source /mnt/home1/wyq/3dgs/miniconda3/etc/profile.d/conda.sh
conda activate pup
export CUDA_HOME=/usr/local/cuda-11.8; export PATH=$CUDA_HOME/bin:$PATH
B=/mnt/home1/wyq/gaussian-splatting-pup/submodules
export PYTHONPATH=/mnt/home1/wyq/shims:$B/simple-knn/build/lib.linux-x86_64-cpython-310:$B/diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-310:$B/rasterization_and_pup_fisher/build/lib.linux-x86_64-cpython-310:$B/compress-diff-gaussian-rasterization/build/lib.linux-x86_64-cpython-310
cd /mnt/home1/wyq/gaussian-splatting-pup || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

run_pup () {   # scene prune_percent tag
  sn=$1; pc=$2; tag=$3
  case $sn in
    truck) DATA=/mnt/home1/wyq/3dgs/data/truck;  ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/truck_vanilla;;
    room)  DATA=/mnt/home1/wyq/3dgs/data/room;   ORIG=/mnt/home1/wyq/3dgs/mpl_las/out/room_vanilla;;
  esac
  M=./experiments/${tag}
  mkdir -p $M/point_cloud/iteration_30000
  cp -f $ORIG/cameras.json $M/cameras.json 2>/dev/null
  cp -f $ORIG/cfg_args    $M/cfg_args 2>/dev/null
  echo "=== PUP $sn prune=$pc -> $M ==="
  python prune_finetune.py -s $DATA -m $M --eval \
    --prune_percent $pc --prune_type fisher \
    --start_pointcloud $ORIG/point_cloud/iteration_30000/point_cloud.ply \
    --iterations 35000 --position_lr_max_steps 35000 --save_iterations 35000 \
    --checkpoint_iterations 0 --test_iterations 0 \
    --fisher_resolution 4 --port 6071 --prune_iterations 30001 \
    > logs_pup_${tag}.log 2>&1
  echo "  rc=$? -> $(ls $M/point_cloud/ 2>/dev/null | tr '\n' ' ')"
}
run_pup truck 0.5  pup_truck_k50
run_pup truck 0.9  pup_truck_k10
run_pup truck 0.95 pup_truck_k05
run_pup room  0.5  pup_room_k50
run_pup room  0.9  pup_room_k10
run_pup room  0.95 pup_room_k05
echo "=== 各模型高斯数 ==="
for t in pup_truck_k50 pup_truck_k10 pup_truck_k05 pup_room_k50 pup_room_k10 pup_room_k05; do
  f=$(find ./experiments/$t/point_cloud -name point_cloud.ply 2>/dev/null | tail -1)
  if [ -n "$f" ]; then echo "$t: $(ls -l $f | awk '{print $5}') bytes  $(grep -a -m1 'element vertex' $f)"; else echo "$t: MISSING"; fi
done
echo PUP_ALL_DONE
