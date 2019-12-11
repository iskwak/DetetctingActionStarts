#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
# source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
cd ../


nvidia-smi -l > /nrs/branson/kwaki/data/20180319_2dconv2/nvidia-smi &
smi_pid=$!

python -m scratch.create_2dconv_features
# python 

kill $smi_pid