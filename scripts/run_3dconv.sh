#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ../


nvidia-smi -l > /nrs/branson/kwaki/outputs/20180619_3dforward_test/nvidia-smi &
smi_pid=$!

python hantman_3dconv.py --hantman_mini_batch=4 --learning_rate=0.0001 --total_epochs=200 --train_file /nrs/branson/kwaki/data/20180605_base_hantman/hantman_train.hdf5  --test_file /nrs/branson/kwaki/data/20180605_base_hantman/hantman_test.hdf5 --valid_file /nrs/branson/kwaki/data/20180605_base_hantman/hantman_valid.hdf5   --out_dir /nrs/branson/kwaki/outputs/20180619_3dforward_test --display_dir /nrs/branson/kwaki/data/hantman_mp4 --video_dir /nrs/branson/kwaki/data/hantman_pruned --cuda_device 0

kill $smi_pid