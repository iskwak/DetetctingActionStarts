#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate

cd ~/checkouts/rutuja_heffalump/drivers
make clean
make

nvidia-smi -l > /nrs/branson/kwaki/data/features/hantman_hoghof2/nvidia-smi &
smi_pid=$!

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/opencv/build/lib
# CUDA_VISIBLE_DEVICES="0" ./hantman_driver -w 352 -h 260 -C 20 -N 8 -S 3 -D 1 -O 3e-6 -c 20 -n 8 -b 5 -v 0 -e 0 -p 3 -P 2 -i /nrs/branson/kwaki/data/features/hantman_hoghof2
./hantman_driver -w 352 -h 260 -C 20 -N 8 -S 3 -D 1 -O 3e-6 -c 20 -n 8 -b 5 -v 0 -e 0 -p 3 -P 2 -i /nrs/branson/kwaki/data/features/hantman_hoghof2

kill $smi_pid