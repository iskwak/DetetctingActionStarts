
OUTDIR=/nrs/branson/kwaki/data/features/canned_i3d/flow
LOGOUT=/nrs/branson/kwaki/data/lists/feature_gen/20190330_feats/
LISTDIR=/nrs/branson/kwaki/data/lists/feature_gen/
BASEKEY="canned_i3d_flow"


python -m i3d.evaluate_hantman_check --filelist ${LISTDIR}$1.txt --gpus 1 --batch_size 10 --window_size 64 --window_start -31 --movie_dir "/nrs/branson/kwaki/data/videos/hantman_flow/" --out_dir ${OUTDIR} --feat_key ${BASEKEY} --eval_type flow --logfile ${LOGOUT}$1_debug.txt
