
OUTDIR=/groups/branson/bransonlab/kwaki/data/thumos14/test/
LOGOUT=/groups/branson/bransonlab/kwaki/data/thumos14/test/
LISTFILE=/groups/branson/bransonlab/kwaki/data/thumos14/lists/debug.txt
MOVIEDIR=/groups/branson/bransonlab/kwaki/data/thumos14/flow_videos/
BASEKEY="canned_i3d_rgb"
# BASEKEY="canned_i3d_flow"

# python -m i3d.evaluate_thumos --filelist ${LISTFILE} --gpus 1 --batch_size 20 --window_size 64 --window_start -31 --movie_dir ${MOVIEDIR} --out_dir ${OUTDIR} --feat_key ${BASEKEY} --eval_type rgb --logfile ${LOGOUT}debug.txt

# python -m i3d.evaluate_thumos --filelist ${LISTFILE} --gpus 1 --batch_size 20 --window_size 64 --window_start -31 --movie_dir ${MOVIEDIR} --out_dir ${OUTDIR} --feat_key ${BASEKEY} --eval_type flow --logfile ${LOGOUT}debug.txt

python -m i3d.evaluate_thumos --filelist ${LISTFILE} --gpus 1 --batch_size 20 --window_size 16 --window_start -15 --movie_dir ${MOVIEDIR} --out_dir ${OUTDIR} --feat_key ${BASEKEY} --eval_type flow --logfile ${LOGOUT}debug.txt
