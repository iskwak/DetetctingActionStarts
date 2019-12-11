#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch/,/nrs/branson-v"

# run the script for singularity.
singularity exec --nv /groups/branson/bransonlab/kwaki/timing/branson_v3.simg ./qumulo_100g_py.sh
