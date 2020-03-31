# Our standard mono model
EXP_NUM="gtav_open_road_2"
SPLIT="gtav_openroad"
DPATH="/mnt/data0-nfs/shared-datasets/gtav_data/"
DEVICE_NO=0
CUDA_VISIBLE_DEVICES=$DEVICE_NO python3 ../train.py --model_name $EXP_NUM\
  --data_path $DPATH \
  --log_dir ../logs --split $SPLIT --png \
  --dataset gtav --height 192 --width 480 \
  #--min_depth 0.1 --max_depth 120.0
  #--learning_rate 0.25e-4
  #--batch_size 6
