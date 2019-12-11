#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab"

# run the script for singularity.
singularity exec --nv /nrs/branson/kwaki/simgs/base_python35.simg ./run_c3d.sh
