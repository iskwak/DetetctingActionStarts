"""Create 3d conv features."""
import sys
import os
import h5py
import numpy
# from helpers.RunningStats import RunningStats
import helpers.paths as paths
import helpers.git_helper as git_helper
from models.hantman_3dconv import Hantman3DConv
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
import cv2
import PIL
import time
import helpers.videosampler as videosampler
# import torch.nn.Parameter as Parameter

# input_name = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_test.hdf5"
# out_dir = "/media/drive2/kwaki/data/hantman_processed/20180212_3dconv"
# network_file = "/nrs/branson/kwaki/outputs/20180205_3dconv/networks/54530/network.pt"
network_file = "/nrs/branson/kwaki/outputs/20180428_feedforward3dconv/networks/136735/network.pt"
out_dir = "/nrs/branson/kwaki/data/20180508_3dconv_all"
input_name = "/nrs/branson/kwaki/data/20180410_all_hoghof/all_mouse_multi_day2_train.hdf5"
video_dir = "/nrs/branson/kwaki/data/hantman_pruned"


class Chopped3DConv(nn.Module):
    def __init__(self, state_dict):
        super(Chopped3DConv, self).__init__()
        self.convs = nn.Sequential(OrderedDict([
            # ('conv1', nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1)),
            ('conv1', nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)),
            ('bn1', nn.BatchNorm3d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)),
            ('bn2', nn.BatchNorm3d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ('conv3', nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)),
            ('bn3', nn.BatchNorm3d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),

            ('conv4', nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('bn4', nn.BatchNorm3d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
            # ('conv4', nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)),
            # ('bn4', nn.BatchNorm3d(512)),
            # ('relu4', nn.ReLU(inplace=True)),
            # ('maxpool4', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        ]))
        self.fcs = nn.Sequential(OrderedDict([
            # ('fc1', nn.Linear(2 * 200704, 4096)),
            ('fc1', nn.Linear(2 * 50176, 4096))
        ]))

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.cpu())
            # else:
            #     import pdb; pdb.set_trace()

    def forward(self, inputs):
        input1 = self.convs(inputs[0])
        input2 = self.convs(inputs[1])
        # import pdb; pdb.set_trace()

        input1 = input1.view(input1.size(0), -1)
        input2 = input1.view(input2.size(0), -1)
        both = torch.cat([input1, input2], dim=1)

        x = self.fcs(both)

        return x


# def process_exp(exp_name):
def init_network():
    # base_net = Hantman3DConv().cuda()
    tic = time.time()
    state_dict = torch.load(network_file)
    print(time.time() - tic)
    # base_net.load_state_dict(state_dict)

    # create the new network.
    network = Chopped3DConv(state_dict).cuda()
    network.eval()

    # setup the preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preproc = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return network, preproc


def preprocess_img(preproc, img):
    """Apply pytorch preprocessing."""
    # resize to fit into the network.
    frame = cv2.resize(img, (224, 224))

    pil = PIL.Image.fromarray(frame)
    tensor_img = preproc(pil)

    return tensor_img


# def _load_chunk(opts, inputs, frames):
#     feat1 = torch.zeros(1, 1, 10, 224, 224)
#     feat2 = torch.zeros(1, 1, 10, 224, 224)

#     if numpy.any(numpy.array(frames) < 0):
#         for j in range(len(frames)):
#             if frames[j] < 0:
#                 frames[j] = 0
#     if np.any(numpy.array(frames) >= inputs[0].size(0)):
#         for j in range(len(frames)):
#             if frames[j] >= inputs[0].size(0):
#                 frames[j] = inputs[0].size(0) - 1
#     idxs = range(0, 10)
#     for i, frame in zip(idxs, frames):
#         feat1[:, 0, i, :, :] = input
#         s[0][frame, 0, :, :]
#         feat2[:, 0, i, :, :] = inputs[1][frame, 0, :, :]
#     # import pdb; pdb.set_trace()
#     return feat1, feat2


def load_chunk(preproc, img_side, img_front, curr_idx, num_frames):
    frames = range(curr_idx - 5, curr_idx + 5)

    for i in range(len(frames)):
        if frames[i] < 0:
            frames[i] = 0
        if frames[i] >= num_frames:
            frames[i] = num_frames - 1

    feat1 = torch.zeros(1, 1, 10, 224, 224)
    feat2 = torch.zeros(1, 1, 10, 224, 224)
    idxs = range(0, 10)
    for i, frame in zip(idxs, frames):
        # print(frame)
        feat1[0, 0, i, :, :] = preprocess_img(preproc, img_side[frame])[0]
        feat2[0, 0, i, :, :] = preprocess_img(preproc, img_front[frame])[0]
    # import pdb; pdb.set_trace()
    return feat1, feat2


def load_chunk2(preproc, img_side, img_front, curr_idx, num_frames):
    frames = range(curr_idx - 5, curr_idx + 5)

    for i in range(len(frames)):
        if frames[i] < 0:
            frames[i] = 0
        if frames[i] >= num_frames:
            frames[i] = num_frames - 1

    feat1 = torch.zeros(1, 1, 10, 224, 224)
    feat2 = torch.zeros(1, 1, 10, 224, 224)
    idxs = range(0, 10)
    for i, frame in zip(idxs, frames):
        # print(frame)
        # import pdb; pdb.set_trace()
        feat1[0, 0, i, :, :] = img_side[frame, :, :]
        feat2[0, 0, i, :, :] = img_front[frame, :, :]
    # import pdb; pdb.set_trace()
    return feat1, feat2


def preprocess_all(preproc, img_side, img_front, num_frames):
    side = torch.zeros(num_frames, 224, 224)
    front = torch.zeros(num_frames, 224, 224)

    for i in range(num_frames):
        side[i, :, :] = preprocess_img(preproc, img_side[i])[0]
        front[i, :, :] = preprocess_img(preproc, img_front[i])[0]

    return side, front


def process_exp(network, preproc, exp_name, in_file, out_dir):
    exp = in_file["exps"][exp_name]

    img_side = exp["raw"]["img_side"].value
    img_front = exp["raw"]["img_front"].value
    labels = exp["raw"]["labels"].value

    # hack... need to get processed labels. question, is this still needed?
    label_filename = os.path.join(
        "/media/drive1/data/hantman_processed/20170827_vgg/exps",
        exp_name)
    with h5py.File(label_filename, "r") as label_data:
        proc_labels = label_data["labels"].value

    num_frames = proc_labels.shape[0]
    side, front = preprocess_all(preproc, img_side, img_front, num_frames)
    features = torch.zeros(num_frames, 4096)
    # tic = time.time()
    for i in range(num_frames):
        # tensor_side, tensor_front = load_chunk(
        #     preproc, img_side, img_front, i, num_frames)
        tensor_side, tensor_front = load_chunk2(
            preproc, side, front, i, num_frames
        )

        var_side = Variable(tensor_side.cuda(), requires_grad=True)
        var_front = Variable(tensor_front.cuda(), requires_grad=True)
        features[i, :] = network([var_side, var_front]).cpu().data
    # print("\t%f" % (time.time() - tic))

    # save to the new experiment file.
    out_name = os.path.join(out_dir, "exps", exp_name)
    with h5py.File(out_name, "w") as out_file:
        out_file["conv3d"] = features.numpy()
        out_file["date"] = exp["date"].value
        out_file["mouse"] = exp["mouse"].value
        out_file["org_labels"] = labels
        out_file["labels"] = proc_labels


def main():
    output_name = os.path.join(out_dir, "all_mouse_multi_day2_train.hdf5")
    network, preproc = init_network()

    rng = numpy.random.RandomState(123)
    with h5py.File(input_name, "r") as in_file:
        sampler = videosampler.HantmanVideoSampler(
            None, in_file, video_dir, 1, seq_len=-1,
            use_pool=False, gpu_id=2, channels=1)

        seen = []
        exp_names = in_file["exp_names"].value
        exp_names.sort()

        for i in range(len(exp_names)):
            moo = sampler.get_minibatch()
            seen.append(moo[-1][0])
        import pdb; pdb.set_trace()
        print("hi")
    #     with h5py.File(output_name, "w") as out_file:
    #         # copy over the general stats
    #         print("a")
    #         out_file["date"] = in_file["date"].value
    #         out_file["exp_names"] = in_file["exp_names"].value
    #         out_file["mice"] = in_file["mice"].value
    #         print("b")
    #         out_file.create_group("exps")
    #         for exp_name in out_file["exp_names"]:
    #             print(exp_name)
    #             tic = time.time()
    #             process_exp(network, preproc, exp_name, in_file, out_dir)
    #             print("\t%f" % (time.time() - tic))


if __name__ == "__main__":
    paths.create_dir(out_dir)
    paths.create_dir(os.path.join(out_dir, "exps"))
    paths.save_command2(out_dir, sys.argv)
    git_helper.log_git_status(
        os.path.join(out_dir, "00_git_status.txt"))

    main()
