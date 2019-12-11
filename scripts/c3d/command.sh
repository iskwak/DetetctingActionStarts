# bsub -n 2 -gpu "num=4:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/c3d/train.log ~/checkouts/QuackNN/scripts/c3d/run_all.sh
# bsub -n 2 -gpu "num=2:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/c3d/train2.log ~/checkouts/QuackNN/scripts/c3d/run_all.sh
bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/c3d/valid.log ~/checkouts/QuackNN/scripts/c3d/run_all.sh
