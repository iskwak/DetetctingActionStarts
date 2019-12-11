#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch"

# run the script for singularity.
singularity exec --nv /nrs/branson/kwaki/simgs/dmsonnet.simg $1

