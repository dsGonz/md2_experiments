MODEL=kitti_test_run
CUDA_VISIBLE_DEVICES=1 python3 ../evaluate_depth.py --eval_mono \
                        --load_weights_folder ../logs/$MODEL/models/weights_19 \
                        --split eigen_zhou --data_path /mnt/data0-nfs/shared-datasets/kitti_data \
                        --png
