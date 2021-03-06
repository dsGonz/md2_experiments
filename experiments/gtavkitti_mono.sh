# Our standard mono model
EXP_NUM="gtavkitti"
SPLIT="gtavkitti50"
DPATH="/mnt/data0-nfs/shared-datasets"
DEVICE_NO=0
CUDA_VISIBLE_DEVICES=$DEVICE_NO python3 ../train.py --model_name $EXP_NUM\
  --data_path $DPATH \
  --log_dir ../logs --split $SPLIT --png \
  --dataset gtavkitti \
  --num_workers 4 \
  #--learning_rate 1e-5
