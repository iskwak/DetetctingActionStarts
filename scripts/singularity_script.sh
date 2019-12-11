#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
source /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ~/checkouts/QuackNN/

# nvidia-smi -l > %path%/nvidia-smi &
# smi_pid=$!

%python_command%

# kill $smi_pid
