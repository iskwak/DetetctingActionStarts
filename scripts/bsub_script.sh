#!/bin/bash
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch,/sharedscratch/branson/kwaki"
cd ~/checkouts/QuackNN/scripts

# run the script for singularity.
singularity exec --nv %image% %command%  
