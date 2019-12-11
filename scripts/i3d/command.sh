
LOGOUT=/nrs/branson/kwaki/data/lists/feature_gen/20190330_feats/
SINGSCRIPT=~/checkouts/QuackNN/scripts/i3d/run_singularity.sh
# BASELIST=/nrs/branson/kwaki/data/lists/feature_gen/list10_
BASELIST=/nrs/branson/kwaki/data/lists/feature_gen/list

# # queue up the rgb jobs
# OUTDIR=/nrs/branson/kwaki/data/features/canned_i3d/rgb
# BASEKEY="canned_i3d_rgb"
# MOVIEDIR=/nrs/branson/kwaki/data/videos/hantman_split/
# LOGFILE=${LOGOUT}/rgb_debug_
# for i in {1..10}
# do
#   bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"\
#     -q gpu_tesla -o ${LOGOUT}/logs/rgb_$i.log\
#     "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} rgb ${LOGFILE}$i.txt"
# done


# # queue up the flow jobs
# OUTDIR=/nrs/branson/kwaki/data/features/canned_i3d/flow
# BASEKEY="canned_i3d_flow"
# MOVIEDIR=/nrs/branson/kwaki/data/videos/hantman_flow/
# LOGFILE=${LOGOUT}/flow_debug_
# # for i in {1..10}
# for i in {1..4}
# do
#   bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"\
#     -q gpu_tesla -m f12u31 -o ${LOGOUT}/logs/flow_$i.log\
#     "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} flow ${LOGFILE}$i.txt"
# done

# OUTDIR=/nrs/branson/kwaki/data/features/canned_i3d/flow
# BASEKEY="canned_i3d_flow"
# MOVIEDIR=/scratch/kwaki/flow/
# # MOVIEDIR=/nrs/branson/kwaki/data/videos/hantman_flow/
# LOGFILE=${LOGOUT}/flow_debug_
# SINGSCRIPT=~/checkouts/QuackNN/scripts/i3d/rev_singularity.sh
# BASELIST=/nrs/branson/kwaki/data/lists/feature_gen/rev_list
# # for i in {1..10}
# for i in {3..3}
# do
#   bsub -n 2 -gpu "num=1"\
#     -q gpu_tesla -m f12u31 -o ${LOGOUT}/logs/rev_list$i.log\
#     "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} flow ${LOGFILE}$i.txt"
# done
# 
#     # -q gpu_rtx -m f14u10 -o ${LOGOUT}/logs/rev_list$i.log\


# queue up the flow jobs
OUTDIR=/nrs/branson/kwaki/data/features/canned_i3d/flow
BASEKEY="canned_i3d_flow"
SINGSCRIPT=~/checkouts/QuackNN/scripts/i3d/rev_singularity.sh
MOVIEDIR=/scratch/kwaki/flow/
LOGFILE=${LOGOUT}/rest/rest_flow_debug_
BASELIST=/nrs/branson/kwaki/data/lists/feature_gen/splits/rem_split_
for i in {0..14}
do
  bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"\
    -q gpu_tesla -o ${LOGOUT}/logs/rest_flow_$i.log\
    "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} flow ${LOGFILE}$i.txt"
done

