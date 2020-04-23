# Our standard mono model
EXP_NUM="pt_gtav_kitti50"
SPLIT="ez_50"
DPATH="/mnt/data0-nfs/shared-datasets/kitti_data"
DEVICE_NO=0
CUDA_VISIBLE_DEVICES=$DEVICE_NO python3 ../train.py --model_name $EXP_NUM\
  --data_path $DPATH \
  --log_dir ../logs --split $SPLIT --png \
  --dataset kitti \
  --num_workers 4 \
  --load_weights_folder ../logs/gtav_openroad_0/models/weights_19
  #--learning_rate 1e-5
