#!/bin/bash
. /opt/venv/bin/activate
cd /groups/branson/bransonlab/kwaki/for_goran/QuackNN
python /groups/branson/bransonlab/kwaki/for_goran/QuackNN/hungarianmouse.py --hantman_mini_batch=5 --learning_rate=0.001 --hantman_perframeloss=WEIGHTED_MSE --hantman_seq_length=1500 --lstm_input_dim=4096 --total_epochs=200 --hantman_perframe_weight=100.0 --hantman_struct_weight=1.0 --hantman_tp=10.0 --hantman_fp=0.25 --hantman_fn=20.0 --feat_keys img_side_norm,img_front_norm --lstm_hidden_dim 128 --arch concat --train_file /nrs/branson/kwaki/data/20170827_vgg/one_mouse_multi_day_train.hdf5 --test_file /nrs/branson/kwaki/data/20170827_vgg/one_mouse_multi_day_test.hdf5 --out_dir /home/cericg/out_dir --cuda_device 0
