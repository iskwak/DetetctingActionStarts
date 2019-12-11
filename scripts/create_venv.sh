#!/bin/bash
# virtualenv -p python2.7 /groups/branson/bransonlab/kwaki/venvs/pytorch
source /groups/branson/bransonlab/kwaki/venvs/pytorch/bin/activate
pip install --upgrade pip
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
pip install torchvision 
pip install GitPython h5py sklearn scipy python-gflags
