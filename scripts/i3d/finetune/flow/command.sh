FRONT_DIR=/scratch/kwaki/flow/front 
SIDE_DIR=/scratch/kwaki/flow/side

# M134
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M134/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u05 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M134/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u05 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"


# M147
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M147_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M147/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u10 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M147/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u10 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"


# M173
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M173_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M173/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u15 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M173/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u15 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"



# M174
TRAIN_FILE=/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M174_train.hdf5

# front
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M174/front/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u20 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${FRONT_DIR}"

# side
OUT_DIR=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M174/side/
mkdir -p ${OUT_DIR}
bsub -n 5 -gpu "num=1" -q gpu_rtx -m f14u20 -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/i3d/finetune/flow/run_singularity_call.sh ${TRAIN_FILE} ${OUT_DIR} ${SIDE_DIR}"
