# Our standard mono model
EXP_NUM="gtavkitti_0"
SPLIT="gtavkitti"
DPATH="/mnt/data0-nfs/shared-datasets"
DEVICE_NO=0
CUDA_VISIBLE_DEVICES=$DEVICE_NO python3 ../train.py --model_name $EXP_NUM\
  --data_path $DPATH \
  --log_dir ../logs --split $SPLIT --png \
  --dataset gtavkitti
  #--min_depth 0.1 --max_depth 120.0
  #--learning_rate 0.25e-4
  #--batch_size 6
