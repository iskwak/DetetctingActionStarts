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
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import PIL

rng = numpy.random.RandomState(123)

g_frame_offsets = [-20, -10, -5, 0, 5, 10, 20]
g_delta_offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
g_smooth_window = 10
g_smooth_std = 5
g_exp_dir = "/localhome/kwaki/data/hantman_pruned/"
g_all_exp_dir = "/mnt/"


def setup_opts(opts):
    """Setup default arguments for the arg parser.

    returns an opt dictionary with default values setup.
    """
    parser = argparse.ArgumentParser(description="Parse and convert Hantman"
                                     " lab mouse mat files into a more python "
                                     "friendly structure")
    parser.add_argument("-f", "--filename", type=str, required=True,
                        help="matlab file to parse")
    # parser.add_argument("-o", "--outname", required=True, type=str,
    #                     help="output picke file name")
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output directory for picke""d data")
    parser.add_argument("-e", "--exp_dir", type=str, required=True,
                        help="location of mat files")
    parser.add_argument("-a", "--all_exp", type=str, required=True,
                        help="location of hantman data in original format")
    # parser.add_argument
    # parser.add_argument("
    command_args = parser.parse_args()

    opts["filename"] = command_args.filename
    opts["out_dir"] = command_args.out_dir
    opts["exp_dir"] = command_args.exp_dir
    opts["all_exp"] = command_args.all_exp
    return opts


def find_org_exp(opts, h5exp, exp_name):
    """Find the original experiment location... This is slow..."""
    exps = h5exp.keys()
    if exp_name in exps:
        return h5exp[exp_name]["path"].value

    exp_path = paths.find_exp_dir(exp_name, opts["all_exp"])

    cur_exp = h5exp.create_group(exp_name)
    cur_exp["path"] = exp_path

    return exp_path


def create_network_features(opts, model, preproc, exp_name, trxdata, cap):
    """Create the pre-trained network features."""
    x1 = numpy.asarray(trxdata["x1"])
    y1 = numpy.asarray(trxdata["y1"])
    x2 = numpy.asarray(trxdata["x2"])
    y2 = numpy.asarray(trxdata["y2"])

    # create a video to help see what the network will be seeing.
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out_name = os.path.join(opts["exp_dir"], exp_name, "paw.avi")
    paw_writer = cv2.VideoWriter(out_name, fourcc, 20.0,
                                 (448, 224), isColor=True)
    out_name = os.path.join(opts["exp_dir"], exp_name, "full.avi")
    frame_writer = cv2.VideoWriter(out_name, fourcc, 20.0,
                                   (448, 224), isColor=True)
    # height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

    half_width = (int)(width / 2)

    cap_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    img_side = torch.FloatTensor(cap_frames, 4096)
    img_front = torch.FloatTensor(cap_frames, 4096)
    paw_side = torch.FloatTensor(cap_frames, 4096)
    paw_front = torch.FloatTensor(cap_frames, 4096)

    # for i in range(cap_frames):
    retval = True
    frame_i = 0
    while retval is True:
        batch_size = 100

        img_data_side = torch.FloatTensor(batch_size, 3, 224, 224)
        img_data_front = torch.FloatTensor(batch_size, 3, 224, 224)
        paw_data_side = torch.FloatTensor(batch_size, 3, 224, 224)
        paw_data_front = torch.FloatTensor(batch_size, 3, 224, 224)

        counter = 0
        for batch_i in range(batch_size):
            retval, frame = cap.read()
            if retval is False:
                # print "move ended?"
                break

            # split the frame in half
            frame1 = frame[:, :half_width, :]
            startx = numpy.floor(x1[frame_i]) - 35
            startx = max(startx, 0)
            startx = (int)(startx)

            starty = numpy.floor(y1[frame_i]) - 35
            starty = max(starty, 0)
            starty = (int)(starty)

            new_frame1 = frame[starty:starty+71, startx:startx+71, :]
            # if starty < 0:
            #     new_frame1 = frame[starty:starty+71, startx:startx+71, :]
            #     cv2.imshow("moo", new_frame1)
            #     cv2.waitKey()
            #     import pdb; pdb.set_trace()

            frame2 = frame[:, half_width:, :]
            startx = numpy.floor(x2[frame_i]) - 35
            startx = max(startx, 0)
            startx = (int)(startx)

            starty = numpy.floor(y2[frame_i]) - 35
            starty = max(starty, 0)
            starty = (int)(starty)
            new_frame2 = frame[starty:starty+71, startx:startx+71, :]

            # convert the frames to the right size.
            # This may warp the images. A potential room for improvement.
            # if frame_i == 2391:
            #     import pdb; pdb.set_trace()
            frame1 = cv2.resize(frame1, (224, 224))
            frame2 = cv2.resize(frame2, (224, 224))
            # if frame_i == 604:
            #     import pdb; pdb.set_trace()
            # print "\t%d: %d,%d" % (frame_i, new_frame1.shape[0], new_frame1.shape[1])
            new_frame1 = cv2.resize(new_frame1, (224, 224))
            new_frame2 = cv2.resize(new_frame2, (224, 224))

            # preprocess the data (create tensors) and store in the batch
            # array.
            img_data_side[batch_i, :, :, :] = preprocess_img(preproc, frame1)
            img_data_front[batch_i, :, :, :] = preprocess_img(preproc, frame2)
            paw_data_side[batch_i, :, :, :] = preprocess_img(
                preproc, new_frame1)
            paw_data_front[batch_i, :, :, :] = preprocess_img(
                preproc, new_frame2)

            new_frame = numpy.concatenate([new_frame1, new_frame2], axis=1)
            paw_writer.write(new_frame)
            new_frame = numpy.concatenate([frame1, frame2], axis=1)
            frame_writer.write(new_frame)
            frame_i = frame_i + 1
            counter = counter + 1

        # The start of new batch of data into the full feature array is a
        # multiple of batch_size. Just in case this is the last batch and
        # doesn't fill the full array, calculate a start and end idx.
        # ceiling(frame / batch_size) would be the expected number of batches
        # that have been filled.
        frame_start = (int)(
            (numpy.ceil(1.0 * frame_i / batch_size) - 1) * batch_size)
        # batch end is the true end of the batch (may not be the full
        # batch_size)
        batch_end = frame_i - frame_start

        # compute the features.
        img_side[frame_start:frame_start + batch_size, :] = apply_network(
            vgg16, batch_end, img_data_side
        )
        img_front[frame_start:frame_start + batch_size, :] = apply_network(
            vgg16, batch_end, img_data_front
        )

        paw_side[frame_start:frame_start + batch_size, :] = apply_network(
            vgg16, batch_end, paw_data_side
        )
        paw_front[frame_start:frame_start + batch_size, :] = apply_network(
            vgg16, batch_end, paw_data_front
        )
    paw_writer.release()
    frame_writer.release()
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    return img_side, img_front, paw_side, paw_front


def get_labels(num_frames, rawmat):
    """Get the labels out of the rawmat."""
    label_names = ["Lift_labl_t0sPos", "Handopen_labl_t0sPos",
                   "Grab_labl_t0sPos", "Sup_labl_t0sPos",
                   "Atmouth_labl_t0sPos", "Chew_labl_t0sPos"]

    # convert the label format into a vector
    labels = numpy.zeros((num_frames, len(label_names)), dtype=numpy.float32)
    for i in range(len(label_names)):
        t0s_str = label_names[i]
        t0s = rawmat[t0s_str]
        for k in range(t0s.size):
            labels[t0s[0][k], i] = 1

    return labels


def apply_network(model, batch_end, tensor_img):
    """Apply the network to an image (assumes a square image)."""
    # make this a gpu computation?
    features = model.forward(Variable(tensor_img[:batch_end, :, :]).cuda())
    return features.data


def preprocess_img(preproc, img):
    """Apply pytorch preprocessing."""
    pil_img = PIL.Image.fromarray(img)
    tensor_img = preproc(pil_img)

    return tensor_img


def create_relative_features(trxdata, trx, cap):
    """Create relative position features."""
    x1 = numpy.asarray(trxdata["x1"])
    y1 = numpy.asarray(trxdata["y1"])
    x2 = numpy.asarray(trxdata["x2"])
    y2 = numpy.asarray(trxdata["y2"])

    # for some reason there are 5 copies of the info in trx?
    # want to use, perch, food, mouth
    # ... don"t really understand nested structs in matfiles and python
    # these are associated with x1,y1
    food = trx["trx"][0][0]["arena"]["food"][0][0][0]
    mouth = trx["trx"][0][0]["arena"]["mouth"][0][0][0]
    perch = trx["trx"][0][0]["arena"]["perch"][0][0][0]
    # these are associated with x2,y2
    foodfront = trx["trx"][0][0]["arena"]["foodfront"][0][0][0]
    mouthfront = trx["trx"][0][0]["arena"]["mouthfront"][0][0][0]
    # no perchfront

    # the arena positions are not relative to the concatenated
    # features
    # height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    foodfront[0] = foodfront[0] + width / 2
    mouthfront[0] = mouthfront[0] + width / 2

    # # next create the relative position features
    num_feat = x1.shape[0]

    features = numpy.zeros((num_feat, 10), dtype=numpy.float32)

    for i in range(num_feat):
        # create pose features
        # first frame
        features[i, 0] = x1[i] - perch[0]
        features[i, 1] = y1[i] - perch[1]
        features[i, 2] = x1[i] - mouth[0]
        features[i, 3] = y1[i] - mouth[1]
        features[i, 4] = x1[i] - food[0]
        features[i, 5] = y1[i] - food[1]
        # second frame
        features[i, 6] = x2[i] - mouthfront[0]
        features[i, 7] = y2[i] - mouthfront[1]
        features[i, 8] = x2[i] - foodfront[0]
        features[i, 9] = y2[i] - foodfront[1]

    pos_feat = numpy.concatenate([x1, y1, x2, y2, features], axis=1)
    return pos_feat


def process_mat(opts, logfile, exp_name, h5exp, rawmat, trxdata, model):
    """Process the matfile."""
    print "\tSearching for the experiment..."
    tic = time.time()
    org_exp = find_org_exp(opts, h5exp, exp_name)
    print "\t%s" % org_exp
    print "\tTook %f seconds" % (time.time() - tic)

    print "\tLoading trx.mat..."
    tic = time.time()
    trxmat = sio.loadmat(os.path.join(org_exp, "trx.mat"))
    print "\tTook %f seconds" % (time.time() - tic)

    # print "\tLoading features.mat..."
    # tic = time.time()
    trxmat = sio.loadmat(os.path.join(org_exp, "trx.mat"))
    # print "\tTook %f seconds" % (time.time() - tic)

    num_frames = trxdata["x1"].size
    logfile.write(",%d" % num_frames)
    labels = get_labels(num_frames, rawmat)

    # # load the movie for processing
    cap = cv2.VideoCapture(os.path.join(org_exp, "movie_comb.avi"))
    cap_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if cap_frames != num_frames:
        print "frame lengths don't match?"
        import pdb; pdb.set_trace()

    # pos_features, perch, mouth, food, mouthfront, foodfront =\
    pos_feat = create_relative_features(trxdata, trxmat, cap)
    img_side, img_front, paw_side, paw_front = create_network_features(
        opts, model, preproc, exp_name, trxdata, cap
    )

    cap.release()
    return img_side, img_front, paw_side, paw_front, pos_feat,\
        labels, num_frames


def parse_hantman_mat(opts, h5file, matfile, model, preproc, logfile):
    """Parse hantman matifle."""
    # each index in this mat file is an experiment (video). For each video,
    # create feature and label matrix. For label information, check the
    # rawdata field. For the trajectory data, check trxdata.
    num_experiments = matfile["trxdata"].size
    exp_names = [exp[0]["exp"][0] for exp in matfile["rawdata"]]
    exp_idx = numpy.argsort(exp_names)

    # get the h5keys
    if "exp" not in h5file.keys():
        h5exp = h5file.create_group("exp")
    else:
        h5exp = h5file["exp"]
    # h5keys = h5file["exp"].keys()

    # sub_exp = range(num_experiments)
    count = 0
    for i in exp_idx:
        print "%d of %d" % (count, num_experiments)
        print "%d" % i
        exp_name = matfile["rawdata"][i][0]["exp"][0]
        print "\t%s" % exp_name
        logfile.write("%s" % exp_name)

        out_file = os.path.join(opts["out_dir"], "exps", exp_name)
        # if os.path.isfile(out_file):
        #     continue
        # if "M134" in exp_name or "M173" in exp_name or "M174" in exp_name:
        #     continue
        tic = time.time()
        img_side, img_front, paw_side, paw_front, pos_feat, labels, num_frames =\
            process_mat(opts, logfile, exp_name,
                        h5exp,
                        matfile["rawdata"][i],
                        matfile["trxdata"][0][i],
                        model)

        # load the JAABA classifier scores
        # scores, postproc = process_scores(opts, logfile, org_exp, num_frames)

        # write the data to h5 files (one for each experiment)
        with h5py.File(out_file, "w") as exp_file:
            exp_file["pos_features"] = pos_feat
            exp_file["img_front"] = img_side.numpy()
            exp_file["img_side"] = img_front.numpy()
            exp_file["paw_side"] = paw_side.numpy()
            exp_file["paw_front"] = paw_front.numpy()
            exp_file["labels"] = labels
            trialtype = matfile["rawdata"][i][0]["trialtype"][0]
            trialtype = trialtype.encode("ascii", "ignore")
            exp_file.attrs["trail_type"] = trialtype
        print "Processing took: %f seconds" % (time.time() - tic)
        logfile.write("\n")
        logfile.flush()
        count = count + 1
    # END for i in range(num_experiments):

    return


def create_opts():
    """Create an opts dictionary."""
    opts = dict()
    opts["filename"] = ""
    opts["out_dir"] = ""
    opts["exp_dir"] = g_exp_dir
    opts["all_exp"] = g_all_exp_dir
    return opts

if __name__ == "__main__":
    opts = create_opts()
    opts = setup_opts(opts)

    # create the output directory
    paths.create_dir(opts["out_dir"])
    paths.create_dir(os.path.join(opts["out_dir"], "exps"))
    paths.save_command(opts["out_dir"])

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

    # log the git information
    # git_helper.log_git_status(
    #     os.path.join(opts["out_dir"], "00_git_status.txt"))

    # try to load the locations of the original experiments.
    h5filename = os.path.join(opts["out_dir"], "00_exp_cache.hdf5")
    h5file = h5py.File(h5filename, "a")

    # load the mat file
    matfile = sio.loadmat(opts["filename"])

    logfilename = os.path.join(opts["out_dir"], "00_log.txt")
    with open(logfilename, "w") as log:
        parse_hantman_mat(opts, h5file, matfile, vgg16, preproc, log)

    h5file.close()
