#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab"

# run the script for singularity.
singularity exec --nv /nrs/branson/kwaki/simgs/dmsonnet.simg ./list4.sh
