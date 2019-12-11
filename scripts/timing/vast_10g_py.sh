#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd /groups/branson/bransonlab/kwaki/timing/QuackNN

nvidia-smi -l > /nrs/branson/kwaki/outputs/timing/vast_10g-nvidia-smi &
smi_pid=$!

# vast 10g script. Should look like the 100g script (with different output location).
# vast is on /nrs/branson-v space. This data is stored in
# /nrs/branson-v/kwaki/hantman...
# command is setup to be as close to my actual training as possible
python threaded_hungarian_mouse.py \
    --train_file /nrs/branson-v/kwaki/hantman/hantman_split_M134_train.hdf5\
    --test_file /nrs/branson-v/kwaki/hantman/hantman_split_M134_test.hdf5\
    --valid_file /nrs/branson-v/kwaki/hantman/hantman_split_M134_valid.hdf5\
    --out_dir /nrs/branson/kwaki/outputs/timing/vast_10g\
    --learning_rate 0.001 --mini_batch 10 --total_epochs 25\
    --display_dir /nrs/branson/kwaki/data/hantman_mp4\
    --video_dir /nrs/branson/kwaki/data/hantman_pruned\
    --max_workers 2 --cuda_device 0 --reweight --save_iterations 10\
    --update_iterations 10 --hantman_arch bidirconcat --lstm_hidden_dim 256\
    --lstm_num_layers 2 --feat_keys rgb_i3d_view1_fc,rgb_i3d_view2_fc\
    --use_pool --loss weighted_mse --normalize 

kill $smi_pid
