

python -m scratch.i3d.create_train_data --window_size 64 --window_start -31 --eval_type flow --train_file /nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_train.hdf5 --out_dir /nrs/branson/kwaki/outputs/i3d_ff/train_data --display_dir /nrs/branson/kwaki/data/hantman_mp4/ --video_dir /nrs/branson/kwaki/data/hantman_flow/front --hantman_mini_batch 10 --frame 1 --learning_rate 0.01 --total_epochs 1 --nouse_pool

