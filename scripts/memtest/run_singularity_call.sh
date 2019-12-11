#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch"

# run the script for singularity.
singularity exec --nv /misc/local/singularity/branson_cuda10_27.simg \
    ./run_script.sh