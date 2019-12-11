
# LOGOUT=/nrs/branson/kwaki/data/lists/feature_gen/20190330_feats/
BASELIST=/nrs/branson/kwaki/data/lists/split_40/list_

# M173 flow Side
SINGSCRIPT=~/checkouts/QuackNN/scripts/i3d/compute_finetune/withlogits/rev_singularity.sh
BASEKEY="finetune_i3d_flow_side"

OUTDIR=/groups/branson/bransonlab/kwaki/data/features/finetune2_i3d/M173/flow/side_compute
MOVIEDIR=/scratch/kwaki/flow/
FEATUREDIR=/groups/branson/bransonlab/kwaki/data/features/finetune2_i3d/M173/flow/side
MODELFILE=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M173/side/networks/beep/epoch_0020.ckpt

for i in {0..39}
do
	mkdir -p ${OUTDIR}$i
	LOGFILE=${OUTDIR}$i/log.txt
	bsub -n 2 -gpu "num=1"\
		 -q gpu_rtx -o ${OUTDIR}$i/output.log\
		 "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR}$i ${BASEKEY} flow ${LOGFILE} ${MODELFILE} side ${FEATUREDIR}"
done


# M173 flow Front
SINGSCRIPT=~/checkouts/QuackNN/scripts/i3d/compute_finetune/withlogits/rev_singularity.sh
BASEKEY="finetune_i3d_flow_front"

OUTDIR=/groups/branson/bransonlab/kwaki/data/features/finetune2_i3d/M173/flow/front_compute
MOVIEDIR=/scratch/kwaki/flow/
FEATUREDIR=/groups/branson/bransonlab/kwaki/data/features/finetune2_i3d/M173/flow/front
MODELFILE=/nrs/branson/kwaki/outputs/i3d_ff/adam2/flow/M173/front/networks/beep/epoch_0020.ckpt

for i in {0..39}
do
	mkdir -p ${OUTDIR}$i
	LOGFILE=${OUTDIR}$i/log.txt
	bsub -n 2 -gpu "num=1"\
		 -q gpu_rtx -o ${OUTDIR}$i/output.log\
		 "${SINGSCRIPT} ${BASELIST}${i}.txt ${MOVIEDIR} ${OUTDIR}$i ${BASEKEY} flow ${LOGFILE} ${MODELFILE} front ${FEATUREDIR}"
done
