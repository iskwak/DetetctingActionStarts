# FRONT_DIR=/nrs/branson/kwaki/data/videos/hantman_split/front
# FRONT_DIR=/scratch/kwaki/rgb/front
# SIDE_DIR=/scratch/kwaki/rgb/side
FRONT_DIR=/sharedscratch/branson/kwaki/hantman/front
SIDE_DIR=/sharedscratch/branson/kwaki/hantman/side

# M134
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M134/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M134/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"


# M147
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M147_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M147/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M147/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"


# M173
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M173_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M173/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M173/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"



# M174
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M174_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M174/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/odas_unfrozen_reweight/rgb/M174/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_tesla -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/odas/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"

