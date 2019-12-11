# queue up the flow jobs
OUTDIR=/groups/branson/bransonlab/kwaki/data/thumos14/flow_videos
SINGSCRIPT=~/checkouts/QuackNN/scripts/i3d/compute_flow/run_flow1.sh
MOVIEDIR=/scratch/kwaki/thumos14/
LOGFOLDER=/groups/branson/bransonlab/kwaki/data/thumos14/lists/logs/
BASELIST=/groups/branson/bransonlab/kwaki/data/thumos14/lists/split_20/list_


for i in {0..19}
do
	bsub -n 1 -gpu "num=1"\
			 -q gpu_rtx -o ${LOGFOLDER}/gpu_flow_$i.log\
			 "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR} ${LOGFOLDER}/proc_files_$i.txt"
done

