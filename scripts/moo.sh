#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
source /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ~/checkouts/QuackNN/

echo "hi"
sshfs kwaki@bransonk-ws9:/localhome/kwaki/outputs ~/mounts/frames
