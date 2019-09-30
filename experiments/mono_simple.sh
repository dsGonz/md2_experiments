# Our standard mono model
EXP_NUM="ms_ssim_1"
CUDA_VISIBLE_DEVICES=1 python3 ../train.py --model_name $EXP_NUM\
  --data_path /mnt/data0-nfs/shared-datasets/kitti_data/ \
  --log_dir ../logs --split eigen_zhou --png \
  --scales 0 --ms_ssim
