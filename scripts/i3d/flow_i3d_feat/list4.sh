#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ~/checkouts/kinetics-i3d

nvidia-smi -l > /nrs/branson/kwaki/data/i3d/flow-list4-nvidia-smi &
smi_pid=$!
python evaluate_flow.py\
    --filelist /nrs/branson/kwaki/data/lists/feature_gen/list4.txt\
    --flow_dir /nrs/branson/kwaki/data/hantman_flow\
    --output_dir /nrs/branson/kwaki/data/20180729_base_hantman/exps/flow_i3d\
    --gpus 1 --batch_size 50 --window_size 64 --window_start -31
kill $smi_pid
