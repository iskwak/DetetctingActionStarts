OUTDIR=/groups/branson/bransonlab/kwaki/data/thumos14/features/rgb_64_past
LOGOUT=/groups/branson/bransonlab/kwaki/data/thumos14/logs/rgb64_past/
LISTFILE=/groups/branson/bransonlab/kwaki/data/thumos14/lists/split_32/list_
# MOVIEDIR=/groups/branson/bransonlab/kwaki/data/thumos14/videos/
MOVIEDIR=/sharedscratch/branson/kwaki/thumos14

BASEKEY="canned_i3d_rgb_64_past"
SINGSCRIPT=~/checkouts/QuackNN/scripts/thumos/i3d/run_singularity.sh

# nodes to use:
# f15u30, f16u15, f16u20, f16u35

mkdir ${LOGOUT}
for i in {0..31}
do
	bsub -n 2 -gpu "num=1"\
		 -q gpu_any -o ${LOGOUT}/bsub_$i.log\
		 "${SINGSCRIPT} ${LISTFILE}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} rgb ${LOGOUT}rgb64_past_$i.txt 64 -63"
		 # "${SINGSCRIPT} ${LISTFILE}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} rgb ${LOGOUT}rgb16_past_$i.txt 16 -15"
		 # "${SINGSCRIPT} ${LISTFILE}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} rgb ${LOGOUT}rgb_$i.txt 64 -31"
done
