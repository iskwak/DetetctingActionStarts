export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch,/sharedscratch/branson,/misc/local"

singularity exec --nv /nrs/branson/kwaki/simgs/branson_cuda10_sonnet.simg ./copy_script.sh
