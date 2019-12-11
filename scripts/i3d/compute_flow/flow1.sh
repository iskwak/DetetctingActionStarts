#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
# . /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ~/checkouts/gpu_flow/build

./compute_flow --file_list $1 --vid_path $2 --out_path $3 --gpuID 0 --log $4
