#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ~/checkouts/QuackNN

python -m i3d.train_i3d_rgb_hantman --window_size 64 --window_start -31 --eval_type rgb --train_file $1 --out_dir $2 --display_dir /nrs/branson/kwaki/data/hantman_mp4/ --video_dir $3 --hantman_mini_batch 6 --learning_rate 0.0001 --total_epochs 100 --use_pool --max_workers 5
