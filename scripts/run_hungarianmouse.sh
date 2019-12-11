#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
# . /opt/venv/bin/activate
source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ../


nvidia-smi -l > /nrs/branson/kwaki/outputs/2dconvpretrain/20180320_test/nvidia-smi &
smi_pid=$!

python hungarianmouse.py --train_file /nrs/branson/kwaki/data/20180319_2dconv/one_mouse_multi_day_train.hdf5 --test_file /nrs/branson/kwaki/data/20180319_2dconv/one_mouse_multi_day_test.hdf5 --image_dir /localhome/kwaki/frames --out_dir /nrs/branson/kwaki/outputs/2dconvpretrain/20180320_test --cuda_device 0 --hantman_mini_batch=10 --learning_rate=0.0001 --hantman_perframeloss=WEIGHTED_MSE --hantman_seq_length=1500 --lstm_input_dim=4096 --total_epochs=500 --hantman_perframe_weight=100.0 --hantman_struct_weight=1.0 --hantman_tp=10.0 --hantman_fp=0.25 --hantman_fn=20.0 --reweight --feat_keys conv2d --lstm_hidden_dim 256 --arch bidirconcat --normalize 

kill $smi_pid