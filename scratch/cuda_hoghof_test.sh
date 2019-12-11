#!/bin/bash

# python threaded_hungarian_mouse.py \
#     --train_file /nrs/branson/kwaki/data/20180729_base_hantman/hantman_train.hdf5\
#     --test_file /nrs/branson/kwaki/data/20180729_base_hantman/hantman_test.hdf5\
#     --val_file /nrs/branson/kwaki/data/20180729_base_hantman/hantman_valid.hdf5\
#     --out_dir /nrs/branson/kwaki/outputs/20180729_test3\
#     --arch bidirconcat\
#     --feat_keys hog_side,hog_front,hof_side,hof_front\
#     --learning_rate 0.0001\
#     --lstm_hidden_dim 100\
#     --cuda_device 0\
#     --hantman_mini_batch=10\
#     --hantman_perframeloss=WEIGHTED_MSE\
#     --hantman_seq_length=1500\
#     --total_epochs=1000\
#     --hantman_perframe_weight=100.0\
#     --hantman_struct_weight=1.0\
#     --hantman_tp=10.0\
#     --hantman_fp=0.25\
#     --hantman_fn=20.0\
#     --reweight\
#     --normalize\
#     --display_dir /nrs/branson/kwaki/data/hantman_mp4\
#     --video_dir /nrs/branson/kwaki/data/hantman_pruned

# python threaded_hungarian_mouse.py \
#     --train_file /nrs/branson/kwaki/data/20180729_base_hantman/hantman_train.hdf5\
#     --test_file /nrs/branson/kwaki/data/20180729_base_hantman/hantman_test.hdf5\
#     --val_file /nrs/branson/kwaki/data/20180729_base_hantman/hantman_valid.hdf5\
#     --out_dir /nrs/branson/kwaki/outputs/20180729_test4\
#     --arch bidirconcat\
#     --feat_keys hog_side,hog_front,hof_side,hof_front,positions\
#     --learning_rate 0.0001\
#     --lstm_hidden_dim 100\
#     --cuda_device 2\
#     --hantman_mini_batch=10\
#     --hantman_perframeloss=WEIGHTED_MSE\
#     --hantman_seq_length=1500\
#     --total_epochs=1000\
#     --hantman_perframe_weight=100.0\
#     --hantman_struct_weight=1.0\
#     --hantman_tp=10.0\
#     --hantman_fp=0.25\
#     --hantman_fn=20.0\
#     --reweight\
#     --normalize\
#     --display_dir /nrs/branson/kwaki/data/hantman_mp4\
#     --video_dir /nrs/branson/kwaki/data/hantman_pruned


python threaded_hungarian_mouse.py \
    --train_file /nrs/branson/kwaki/data/20180410_all_hoghof/all_mouse_multi_day2_train.hdf5\
    --test_file /nrs/branson/kwaki/data/20180410_all_hoghof/all_mouse_multi_day2_test.hdf5\
    --val_file /nrs/branson/kwaki/data/20180410_all_hoghof/all_mouse_multi_day2_valid.hdf5\
    --out_dir /nrs/branson/kwaki/outputs/20180729_test5\
    --arch bidirconcat\
    --feat_keys hog_side,hog_front,hof_side,hof_front,positions\
    --learning_rate 0.0001\
    --lstm_hidden_dim 100\
    --cuda_device 2\
    --hantman_mini_batch=10\
    --hantman_perframeloss=WEIGHTED_MSE\
    --hantman_seq_length=1500\
    --total_epochs=1000\
    --hantman_perframe_weight=100.0\
    --hantman_struct_weight=1.0\
    --hantman_tp=10.0\
    --hantman_fp=0.25\
    --hantman_fn=20.0\
    --reweight\
    --normalize\
    --display_dir /nrs/branson/kwaki/data/hantman_mp4\
    --video_dir /nrs/branson/kwaki/data/hantman_pruned
