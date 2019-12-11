#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ../


nvidia-smi -l > /nrs/branson/kwaki/outputs/20180623_3dconvforward/20180623-0.0001_0_output/proc_info/nvidia-smi &
smi_pid=$!

python hantman_3dconv_eval.py --train_file /nrs/branson/kwaki/data/20180605_base_hantman/hantman_train.hdf5 --test_file /nrs/branson/kwaki/data/20180605_base_hantman/hantman_test.hdf5 --valid_file /nrs/branson/kwaki/data/20180605_base_hantman/hantman_valid.hdf5 --out_dir /nrs/branson/kwaki/outputs/20180623_3dconvforward/20180623-0.0001_0_output --learning_rate 0.0001 --hantman_mini_batch=4 --total_epochs=200 --frames "-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10" --display_dir /nrs/branson/kwaki/data/hantman_mp4 --video_dir /nrs/branson/kwaki/data/hantman_pruned --cuda_device 0 --reweight --model /nrs/branson/kwaki/outputs/20180623_3dconvforward/20180623-0.0001_0_output/networks/162027/network.pt

kill $smi_pid