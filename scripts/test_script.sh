#!/bin/bash
# expects to be run in <...>/QuackNN/scripts
. /opt/venv/bin/activate
cd ../models
python test_example.py --cuda_device=0
