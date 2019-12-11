#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
# . /opt/venv/bin/activate
source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ../
python dataparallel_test.py --hantman_mini_batch=20 --learning_rate=0.001 --hantman_perframeloss=WEIGHTED_MSE --hantman_seq_length=1500 --lstm_input_dim=4096 --total_epochs=200 --hantman_perframe_weight=100.0 --hantman_struct_weight=1.0 --hantman_tp=10.0 --hantman_fp=0.25 --hantman_fn=20.0 --feat_keys paw_side_norm,paw_front_norm,img_side_norm,img_front_norm --lstm_hidden_dim 128 --arch concat --train_file /nrs/branson/kwaki/data/20170827_vgg/one_mouse_multi_day_train.hdf5 --test_file /nrs/branson/kwaki/data/20170827_vgg/one_mouse_multi_day_test.hdf5 --out_dir /nrs/branson/kwaki/outputs/20171227_cluster_multi --cuda_device 0
