# test M173
# OUTDIR=/nrs/branson/kwaki/data/test/
# LOGOUT=/nrs/branson/kwaki/data/lists/feature_gen/20190402_test/
# LISTDIR=/nrs/branson/kwaki/data/lists/feature_gen/
# MOVIEDIR="/nrs/branson/kwaki/data/videos/hantman_flow/front"
# BASEKEY="M173_i3d_flow_front"
# 
# 
# python -m i3d.process_i3d_network\
#     --filelist ${LISTDIR}debug_list.txt --batch_size 10\
#     --window_size 64 --window_start -31 --movie_dir ${MOVIEDIR}\
#     --out_dir ${OUTDIR} --feat_key ${BASEKEY} --eval_type flow\
#     --model /nrs/branson/kwaki/outputs/i3d_ff/flow/M173/front/networks/epoch_0080.ckpt

OUTDIR=/nrs/branson/kwaki/data/test/rgb_test
BASEKEY="M173_i3d_rgb_front"
LISTDIR=/nrs/branson/kwaki/data/lists/feature_gen/
MOVIEDIR="/nrs/branson/kwaki/data/videos/hantman_rgb/front"
FEATURE_DIR=/nrs/branson/kwaki/data/test/rgb_test/features


python -m i3d.process_i3d_network\
    --filelist ${LISTDIR}debug_list.txt --batch_size 10\
    --window_size 64 --window_start -31 --movie_dir ${MOVIEDIR}\
    --out_dir ${OUTDIR} --feat_key ${BASEKEY} --eval_type rgb\
    --model /nrs/branson/kwaki/outputs/i3d_ff/rgb/M173/front/networks/epoch_0080.ckpt\
    --feature_dir ${FEATURE_DIR}
