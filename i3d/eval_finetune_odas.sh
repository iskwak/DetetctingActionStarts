
OUTDIR=/nrs/branson/kwaki/data/test/finetune_flow/
FEATDIR=/nrs/branson/kwaki/data/test/finetune_flow/feats2
LOGOUT=/nrs/branson/kwaki/data/test/finetune_flow/
LISTDIR=/nrs/branson/kwaki/data/lists/feature_gen/
BASEKEY="finetune_i3d_flow_side"
# MODEL=/nrs/branson/kwaki/outputs/tests/rgb_test/networks/epoch_0000.ckpt
# MODEL=/nrs/branson/kwaki/outputs/i3d_ff/adam2/rgb/M147/side/networks/beep/epoch_0020.ckpt
MODEL=/nrs/branson/kwaki/outputs/odas_unfrozen/rgb/M134/side/networks/epoch_0030.ckpt
# MODEL=/nrs/branson/kwaki/outputs/tests/train/networks/epoch_0000.ckpt

python -m i3d.eval_finetune_odas --filelist ${LISTDIR}debug_list.txt --gpus 1 --batch_size 6 --window_size 64 --window_start -63 --movie_dir "/nrs/branson/kwaki/data/videos/hantman_rgb/" --out_dir ${OUTDIR} --feat_key ${BASEKEY} --eval_type rgb --logfile ${LOGOUT}$1_debug.txt --model ${MODEL} --type side --feat_dir ${FEATDIR}

