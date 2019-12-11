# 10g nrs qumulo
bsub -n 2 -P scicompsys -gpu "num=1" -m f13u10 -q gpu_gtx -J qumulo_10g -o /nrs/branson/kwaki/outputs/timing/qumulo_10g.log /groups/branson/bransonlab/kwaki/timing/qumulo_10g_sing.sh
# 10g nrs vast
bsub -n 2 -P scicompsys -gpu "num=1" -m f13u10 -q gpu_gtx -J vast_10g -o /nrs/branson/kwaki/outputs/timing/vast_10g.log /groups/branson/bransonlab/kwaki/timing/vast_10g_sing.sh
 
# 100g nrs qumulo
bsub -n 2 -P scicompsys -gpu "num=1" -m f13u15 -q gpu_gtx -J qumulo_100g -o /nrs/branson/kwaki/outputs/timing/qumulo_100g.log /groups/branson/bransonlab/kwaki/timing/qumulo_100g_sing.sh
# 100g nrs vast
bsub -n 2 -P scicompsys -gpu "num=1" -m f13u15 -q gpu_gtx -J vast_100g -o /nrs/branson/kwaki/outputs/timing/vast_100g.log /groups/branson/bransonlab/kwaki/timing/vast_100g_sing.sh

# either scratch
bsub -n 2 -P scicompsys -gpu "num=1" -q gpu_gtx -J scratch -o /nrs/branson/kwaki/outputs/timing/scratch.log /groups/branson/bransonlab/kwaki/timing/scratch_sing.sh
