"""Create 3d conv features."""
import sys
import os
import h5py
import numpy
# from helpers.RunningStats import RunningStats
import helpers.paths as paths
import helpers.git_helper as git_helper
import torchvision.models.resnet as resnet
# from models.hantman_3dconv import Hantman3DConv
# from models.hantman_feedforward import HantmanFeedForward
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import cv2
import PIL
import time
import torch.utils.model_zoo as model_zoo
# import torch.nn.Parameter as Parameter
import math

input_name = "/nrs/branson/kwaki/data/20170827_vgg/one_mouse_multi_day_train.hdf5"
out_dir = "/nrs/branson/kwaki/data/20180319_2dconv"
network_file = "/nrs/branson/kwaki/outputs/20180216_2dconv_threaded_0001/networks/21200/network.pt"
movie_dir = "/nrs/branson/kwaki/data/hantman_pruned/"


class Chopped2D(nn.Module):
    # based on Resnet18 code in models. Mainly need to change the fc layer
    # and the forward pass.
    def __init__(self, state_dict, pretrained=False):
        super(Chopped2D, self).__init__()
        self.make_base_model(state_dict, pretrained)

    def make_base_model(self, state_dict, pretrained):
        block = resnet.BasicBlock
        layers = [2, 2, 2, 2]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2 * 512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # fill up the parameters.
        # if pretrained is True:
        #     self.load_state_dict(
        #         model_zoo.load_url(resnet.model_urls['resnet18']))
        # after loading up the pretrained weights. Tweak the model.
        # Mainly... the final FC layer needs to be 2 times the width.
        self.fc = nn.Linear(2 * 512 * block.expansion, 2 * 512 * block.expansion)

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.cpu())
            # else:
            #     import pdb; pdb.set_trace()
        # self.relu1 = nn.ReLU(True)
        # self.dropout1 = nn.Dropout()
        # self.fc2 = nn.Linear(2 * 512 * block.expansion, 2 * 512 * block.expansion)
        # self.relu2 = nn.ReLU(True)
        # self.dropout2 = nn.Dropout()
        # self.fc_class = nn.Linear(2 * 512 * block.expansion, 6)
        # self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def process_image(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, inputs):
        input1 = self.process_image(inputs[0])
        input2 = self.process_image(inputs[1])
        both = torch.cat([input1, input2], dim=1)

        x = self.fc(both)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.fc_class(x)
        # x = self.sigmoid(x)

        return x


# def process_exp(exp_name):
def init_network():
    # base_net = Hantman3DConv().cuda()
    tic = time.time()
    state_dict = torch.load(network_file)
    print(time.time() - tic)
    # base_net.load_state_dict(state_dict)

    # create the new network.
    network = Chopped2D(state_dict).cuda()
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


def preprocess_all(preproc, img_side, img_front, num_frames):
    side = torch.zeros(num_frames, 3, 224, 224)
    front = torch.zeros(num_frames, 3, 224, 224)

    for i in range(num_frames):
        side[i, :, :, :] = preprocess_img(preproc, img_side[i])
        front[i, :, :, :] = preprocess_img(preproc, img_front[i])

    return side, front


def process_frame(preproc, frame):
    # split the frame in half
    half_width = frame.shape[1] / 2
    frame1 = frame[:, :half_width, :]
    frame2 = frame[:, half_width:, :]

    frame1 = cv2.resize(frame1, (224, 224))
    frame2 = cv2.resize(frame2, (224, 224))

    img_side = preprocess_img(preproc, frame1)
    img_front = preprocess_img(preproc, frame2)

    return img_side, img_front


def preprocess_video(preproc, exp_name, num_frames):
    full_path = os.path.join(movie_dir, exp_name)
    movie_filename = os.path.join(full_path, "movie_comb.avi")

    # load the movie
    cap = cv2.VideoCapture(movie_filename)
    all_side = torch.zeros(num_frames, 3, 224, 224)
    all_front = torch.zeros(num_frames, 3, 224, 224)
    frame_num = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret is True:
            img_side, img_front = process_frame(preproc, frame)
            all_side[i, :] = img_side
            all_front[i, :] = img_front
            # cv2.imshow("moo", frame)
            # cv2.waitKey(10)
            frame_num += 1
        else:
            break
    print("\t%d frames" % frame_num)

    cap.release()

    return all_side, all_front


def process_exp(network, preproc, exp_name, in_file, out_dir):
    exp = in_file["exps"][exp_name]
    labels = exp["org_labels"].value
    proc_labels = exp["labels"].value

    num_frames = proc_labels.shape[0]
    # num_frames = 501
    side, front = preprocess_video(preproc, exp_name, num_frames)

    # side, front = preprocess_all(preproc, img_side, img_front, num_frames)
    features = torch.zeros(num_frames, 1, 1024)
    # tic = time.time()
    batch_size = 200
    num_batch = int(num_frames / batch_size)
    batch_id = 0
    i2 = 0
    for batch_id in range(num_batch):
        i1 = batch_id * batch_size
        i2 = (batch_id + 1) * batch_size

        tensor_side = side[i1:i2, :].view(batch_size, 3, 224, 224)
        tensor_front = front[i1:i2, :].view(batch_size, 3, 224, 224)

        var_side = Variable(tensor_side.cuda(), requires_grad=True)
        var_front = Variable(tensor_front.cuda(), requires_grad=True)

        features[i1:i2, 0, :] = network([var_side, var_front]).cpu().data
    # get the final batch
    # i1 = (batch_id + 1) * batch_size
    i1 = i2
    if i2 < num_frames:
        # print("moo")
        i2 = num_frames
        size = num_frames - i1
        tensor_side = side[i1:i2, :].view(size, 3, 224, 224)
        tensor_front = front[i1:i2, :].view(size, 3, 224, 224)

        var_side = Variable(tensor_side.cuda(), requires_grad=True)
        var_front = Variable(tensor_front.cuda(), requires_grad=True)

        features[i1:i2, 0, :] = network([var_side, var_front]).cpu().data

    # import pdb; pdb.set_trace()
    # save to the new experiment file.
    out_name = os.path.join(out_dir, "exps", exp_name)
    with h5py.File(out_name, "w") as out_file:
        out_file["conv2d"] = features.numpy()
        out_file["date"] = exp["date"].value
        out_file["mouse"] = exp["mouse"].value
        out_file["org_labels"] = labels
        out_file["labels"] = proc_labels


def main():
    # import pdb; pdb.set_trace()
    output_name = os.path.join(out_dir, "one_mouse_multi_day_train.hdf5")
    network, preproc = init_network()

    with h5py.File(input_name, "r") as in_file:
        with h5py.File(output_name, "w") as out_file:
            # copy over the general stats
            print("a")
            out_file["date"] = in_file["date"].value
            out_file["exp_names"] = in_file["exp_names"].value
            out_file["mice"] = in_file["mice"].value
            print("b")
            out_file.create_group("exps")
            for exp_name in out_file["exp_names"]:
                print(exp_name)
                tic = time.time()
                process_exp(network, preproc, exp_name, in_file, out_dir)
                
                out_file["exps"][exp_name] = h5py.ExternalLink(
                    os.path.join("exps", exp_name), "/"
                )
                print("\t%f" % (time.time() - tic))


if __name__ == "__main__":
    paths.create_dir(out_dir)
    paths.create_dir(os.path.join(out_dir, "exps"))
    paths.save_command2(out_dir, sys.argv)
    git_helper.log_git_status(
        os.path.join(out_dir, "00_git_status.txt"))

    main()
