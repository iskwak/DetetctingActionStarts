export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch"

# run the script for singularity.
singularity exec --nv /nrs/branson/kwaki/simgs/branson_cuda10_sonnet.simg ./rev_compute_feat.sh $1 $2 $3 $4 $5 $6 $7

