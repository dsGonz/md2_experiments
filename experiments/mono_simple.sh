# Our standard mono model
EXP_NUM="ms_ssim_1"
SPLIT="benchmark"
DEVICE_NO="0"
CUDA_VISIBLE_DEVICES=$DEVICE_NO python3 ../train.py --model_name $EXP_NUM\
  --data_path /mnt/data0-nfs/shared-datasets/kitti_data/ \
  --log_dir ../logs --split $SPLIT --png \
  #--scales 0 --ms_ssim
