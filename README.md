# DetetctingActionStarts
## Overview
This repository contains code to train and run the models described in: [Detecting the Starting Frame of Actions in Video](https://arxiv.org/abs/1906.03340).

## Requirements
* pytorch 1.0.1
* torchvision
* opencv-python
* h5py
* GitPython
* python-gflags

## Usage
Run scripts in results_scripts folder
```
./result_run_scripts/hoghof_matching.sh
```

## Dataset
[http://research.janelia.org/bransonlab/MouseReachData/](http://research.janelia.org/bransonlab/MouseReachData/)


## Usage Notes
Visualizations require Cross Origin Request 
* [http://testingfreak.com/how-to-fix-cross-origin-request-security-cors-error-in-firefox-chrome-and-ie/](Firefox)

## Citation
If you use this code, please cite the following paper.
```
@article{kwak2019detecting,
  title={Detecting the Starting Frame of Actions in Video},
  author={Kwak, Iljung S and Kriegman, David and Branson, Kristin},
  journal={arXiv preprint arXiv:1906.03340},
  year={2019}
}
```
