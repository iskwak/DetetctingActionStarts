bsub -n 2 -gpu "num=2:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/c3d/test.log ~/checkouts/QuackNN/scripts/c3d/test_call.sh
