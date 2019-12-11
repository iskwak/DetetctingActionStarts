#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ~/checkouts/hx173149-C3D-tensorflow

nvidia-smi -l > /nrs/branson/kwaki/data/c3d/test-nvidia-smi &
smi_pid=$!
python predict_c3d_test_hantman.py
kill $smi_pid
