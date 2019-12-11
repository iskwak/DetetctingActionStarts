# side
OUT_DIR=/nrs/branson/kwaki/test/memtest/
mkdir -p ${OUT_DIR}
bsub -n 3 -gpu "num=1" -q gpu_rtx -o ${OUT_DIR}output.log "~/checkouts/QuackNN/scripts/memtest/run_singularity_call.sh"
