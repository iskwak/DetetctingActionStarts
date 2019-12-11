OUTDIR=/groups/branson/bransonlab/kwaki/data/thumos14/features/flow_16_past
LOGOUT=/groups/branson/bransonlab/kwaki/data/thumos14/logs/flow_16_past/
LISTFILE=/groups/branson/bransonlab/kwaki/data/thumos14/lists/split_32/list_
# MOVIEDIR=/groups/branson/bransonlab/kwaki/data/thumos14/videos/
MOVIEDIR=/sharedscratch/branson/kwaki/flow_thumos14

BASEKEY="canned_i3d_flow_16_past"
SINGSCRIPT=~/checkouts/QuackNN/scripts/thumos/i3d/run_singularity.sh

for i in {0..31}
do
	bsub -n 4 -gpu "num=1"\
		 -q gpu_rtx -o ${LOGOUT}/bsub_$i.log\
		 "${SINGSCRIPT} ${LISTFILE}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} flow ${LOGOUT}flow_$i.txt 16 -15"
		 # "${SINGSCRIPT} ${LISTFILE}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} flow ${LOGOUT}flow_redo_$i.txt  64 -31"
	# echo bsub -n 2 -gpu "num=1"\
	# 	 -q gpu_rtx -m e11u10 -o ${LOGOUT}/rgb_$i.log\
	# 	 "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR} ${BASEKEY} flow ${LOGFILE}$i.txt"
done
