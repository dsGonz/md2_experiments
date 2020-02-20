# Our standard mono model
EXP_NUM="presil_1"
SPLIT="presil_dynamic"
DPATH="/mnt/data0-nfs/shared-datasets/PreSIL/"
DEVICE_NO=0
CUDA_VISIBLE_DEVICES=$DEVICE_NO python3 ../train.py --model_name $EXP_NUM\
  --data_path $DPATH \
  --log_dir ../logs --split $SPLIT --png \
  --dataset gtav --height 256 --width 480 \
  --learning_rate 0.5e-4
  #--min_depth 0.1 --max_depth 100.0
