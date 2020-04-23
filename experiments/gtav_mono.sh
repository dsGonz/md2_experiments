# Our standard mono model
EXP_NUM="gtav_openroad_static_0"
SPLIT="gtav_openroad_static"
DPATH="/mnt/data0-nfs/shared-datasets/gtav_data/static_data"
DEVICE_NO=0
CUDA_VISIBLE_DEVICES=$DEVICE_NO python3 ../train.py --model_name $EXP_NUM\
  --data_path $DPATH \
  --log_dir ../logs --split $SPLIT --png \
  --dataset gtav 
  #--min_depth 0.1 --max_depth 120.0
  #--learning_rate 0.25e-4
  #--batch_size 6
