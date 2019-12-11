#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch"
cd ~/checkouts/QuackNN/scripts

singularity exec --nv /misc/local/singularity/branson_v3.simg ./run_singularity_hoghof_generation.sh
