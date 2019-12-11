"""Based off of convert_hantman_data.py"""
from __future__ import print_function, division
import numpy
import argparse
import h5py
import scipy.io as sio
import helpers.paths as paths
# import helpers.git_helper as git_helper
import os
import time
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import PIL
import cv2
import cv
import sys

rng = numpy.random.RandomState(123)

# g_all_exp_dir = "/mnt/"
labels_dir = "/media/drive3/kwaki/data/mpiicooking2/labels"
videos_dir = "/media/drive3/kwaki/data/mpiicooking2/videos"
out_dir = "/media/drive3/kwaki/data/mpiicooking2/hdf5"
batch_size = 50


def preprocess_img(preproc, img):
    """Apply pytorch preprocessing."""
    pil_img = PIL.Image.fromarray(img)
    tensor_img = preproc(pil_img)

    return tensor_img


def process_file(vgg, preproc, filename):
    movie_filename = os.path.join(videos_dir, filename[:-3] + 'avi')
    print(movie_filename)
    cap = cv2.VideoCapture(movie_filename)
    if cap.isOpened() is False:
        import pdb; pdb.set_trace()

    num_frames = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
    i = 0
    j = 0
    retval = True
    features = numpy.zeros((num_frames, 1, 4096))
    while retval is True:
        # construct a batch
        batch = torch.zeros(batch_size, 3, 224, 224)
        for j in range(batch_size):
            retval, frame = cap.read()
            if retval is False:
                # print "move ended?"
                break
            frame = cv2.resize(frame, (224, 224))
            import pdb; pdb.set_trace()
            batch[j, :] = preprocess_img(preproc, frame)
        # print(j)
        start_idx = i
        end_idx = i + j + 1
        temp = vgg(Variable(batch).cuda()).data.cpu().numpy()
        try:
            features[start_idx:end_idx, 0, :] = temp[:j + 1, :]
        except:
            import pdb; pdb.set_trace()

    # create hdf5 with the features
    h5_name = os.path.join(out_dir, "exps", filename[:-3] + 'hdf5')
    with h5py.File(h5_name, "w") as h5_data:
        h5_data["vgg"] = features

    # import pdb; pdb.set_trace()

    cap.release()


def main():
    # first get file list. extension, and add the movie extension
    filenames = os.listdir(labels_dir)
    filenames.sort()

    # create the network
    vgg16 = models.vgg16(pretrained=True)
    # want the 4096 features, before the relu.
    vgg16.classifier = torch.nn.Sequential(
        *[vgg16.classifier[layer_i] for layer_i in range(2)])
    # put on the GPU for better compute speed
    vgg16.cuda()
    # put into eval mode
    vgg16.eval()

    # setup the preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preproc = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    for filename in filenames:
        tic = time.time()
        process_file(vgg16, preproc, filename)
        print(time.time() - tic)
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    # opts = setup_opts(opts)
    paths.create_dir(out_dir)
    paths.create_dir(os.path.join(out_dir, "exps"))
    paths.save_command2(out_dir, sys.argv)

    main()
