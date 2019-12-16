#!/bin/bash

# location of the hdf5 file used for training/testing
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_train.hdf5
TEST_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5
# Directory with web viewable movies
DISPLAY_DIR=/nrs/branson/kwaki/data/hantman_mp4
# Directory with movies for train/test
VIDEO_DIR=/nrs/branson/kwaki/data/hantman_pruned

# output directory
OUT_DIR=/nrs/branson/kwaki/outputs/test2

python threaded_hungarian_mouse.py \
    --train_file $TRAIN_FILE \
    --test_file  $TEST_FILE\
    --display_dir  $DISPLAY_DIR \
    --video_dir $VIDEO_DIR \
    --out_dir $OUT_DIR \
    --hantman_arch "bidirconcat" \
    --feat_keys "hoghof" \
    --mini_batch 5 \
    --total_epochs 400 \
    --cuda_device 0 \
    --reweight \
    --save_iterations 50 \
    --update_iterations 50 \
    --lstm_num_layers 2 \
    --use_pool \
    --learning_rate 1e-05 \
    --lstm_hidden_dim 256 \
    --normalize \
    --anneal_type "exp_step" \
    --loss "hungarian" \
    --hantman_perframe_weight 0.99 \
    --perframe_decay 0.9 \
    --perframe_decay_step 5 \
    --perframe_stop 0.5 \
    --hantman_tp 4.0 \
    --hantman_fp 1.0 \
    --hantman_fn 2.0 

