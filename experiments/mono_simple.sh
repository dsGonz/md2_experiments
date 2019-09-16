# Our standard mono model
CUDA_VISIBLE_DEVICES=0 python3 ../train.py --model_name kitti_test_run\
  --data_path /mnt/data0-nfs/shared-datasets/kitti_data/ \
  --log_dir ../logs --split eigen_zhou --png

