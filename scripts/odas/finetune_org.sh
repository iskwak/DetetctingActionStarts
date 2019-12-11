#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ~/checkouts/QuackNN

python -m i3d.train_i3d_c3d_hantman --window_size 128 --window_start -63 --eval_type rgb --train_file $1 --out_dir $2 --display_dir /nrs/branson/kwaki/data/hantman_mp4/ --video_dir $3 --hantman_mini_batch 4 --learning_rate 0.0001 --total_epochs 100 --use_pool --max_workers 5
