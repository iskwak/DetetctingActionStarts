bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/i3d/flow-list1.log ~/checkouts/QuackNN/scripts/i3d/flow_i3d_feat/run_list1.sh
bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/i3d/flow-list2.log ~/checkouts/QuackNN/scripts/i3d/flow_i3d_feat/run_list2.sh
bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/i3d/flow-list3.log ~/checkouts/QuackNN/scripts/i3d/flow_i3d_feat/run_list3.sh
bsub -n 2 -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" -q gpu_tesla -o /nrs/branson/kwaki/data/i3d/flow-list4.log ~/checkouts/QuackNN/scripts/i3d/flow_i3d_feat/run_list4.sh
