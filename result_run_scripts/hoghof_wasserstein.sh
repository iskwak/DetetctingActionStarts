#!/bin/bash

# location of the hdf5 file used for training/testing
TRAIN_FILE=data/hdf5/split_M134_train.hdf5
TEST_FILE=data/hdf5/split_M134_test.hdf5
# Directory with web viewable movies
DISPLAY_DIR=data/display_videos
# Directory with movies for train/test
# VIDEO_DIR=/nrs/branson/kwaki/data/hantman_pruned

# output directory
OUT_DIR=outputs/wasserstein

python threaded_hungarian_mouse.py \
    --train_file $TRAIN_FILE \
    --test_file  $TEST_FILE\
    --display_dir  $DISPLAY_DIR \
    --out_dir $OUT_DIR \
    --hantman_arch "bidirconcat" \
    --feat_keys "hoghof" \
    --mini_batch 10 \
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
    --loss "wasserstein" \
    --hantman_perframe_weight 0.99 \
    --perframe_decay 0.9 \
    --perframe_decay_step 1 \
    --perframe_stop 0.5

