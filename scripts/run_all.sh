#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch"
cd ~/checkouts/QuackNN/scripts

# run the script for singularity.
# singularity exec --nv /misc/local/singularity/branson.simg ./run_hungarianmouse.sh
# singularity exec --nv /misc/local/singularity/branson.simg ./run_hungarianmouse.sh
# singularity exec --nv /misc/local/singularity/branson.simg ./run_feat_create.sh
# singularity exec --nv /misc/local/singularity/branson_torchvision.simg ./run_feat_create.sh
singularity exec --nv /misc/local/singularity/branson_v3.simg ./run_3dconv.sh
# singularity exec --nv /misc/local/singularity/branson.simg ./run_3dconv.sh