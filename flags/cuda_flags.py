"""GFlags to help with pytorch cuda settings."""
# import torch
import gflags

gflags.DEFINE_integer("cuda_device", 0, "Which CUDA device to use, -1 for cpu.")
