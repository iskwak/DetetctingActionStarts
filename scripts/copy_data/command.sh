bsub -n 1 -gpu "num=1" -q gpu_any -o /nrs/branson/kwaki/outputs/odas/copylog.log ~/checkouts/QuackNN/scripts/copy_data/singularity.sh

