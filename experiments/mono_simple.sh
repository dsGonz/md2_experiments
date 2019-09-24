# Our standard mono model
EXP_NUM="scale_0"
CUDA_VISIBLE_DEVICES=0 python3 ../train.py --model_name $EXP_NUM\
  --data_path /mnt/data0-nfs/shared-datasets/kitti_data/ \
  --log_dir ../logs --split eigen_zhou --png \
  --scales 0

