#!/bin/bash
clear


if [ $1 == "102" ]; then
    CUDA_VISIBLE_DEVICES=$2 py_gxdai main.py \
        --phase $3 \
        --batch_size 5 \
        --lamb 20 \
        --sketch_train_list ./sketch_train102.txt \
        --sketch_test_list ./sketch_test102.txt \
        --shape_list ./shape102.txt \
        --ckpt_dir ./checkpoint \
        --logdir ./logs \
        --lossType "weightedContrastiveLoss"
elif [ $1 == "101" ]; then
    CUDA_VISIBLE_DEVICES=$2 py_gxdai main.py --phase $3 --batch_size 1 --sketch_train_list ./sketch_train102.txt --sketch_test_list ./sketch_test102.txt --shape_list ./shape102.txt
fi

