import cv2
import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
# import gflags
import gflags
import sys
import time
import subprocess
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching
import scipy

gflags.DEFINE_string("input_dir", "", "input folder")
gflags.DEFINE_string("output_dir", "", "output folder")
gflags.DEFINE_string("experiment", "", "experiment")
# this version will parse the losses


def create_frame(frame, data, frame_idx, start_frame, end_frame):
    print_names = data["names"]
    loss_names = data["loss_names"]
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 4.5), dpi=100)
    canvas = FigureCanvas(fig)
    main_fig = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[6,1])

    # figure handle for the video frames
    ax = plt.Subplot(fig, main_fig[0])
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

    # frame_fig = gridspec.GridSpecFromSubplotSpec(
    #     1, 1, subplot_spec=main_fig[0], wspace=0.0, hspace=0.0)

    # ax = plt.Subplot(fig, frame_fig[0])
    # ax.imshow(frame)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.add_subplot(ax)

    # create the handles for the ethogram style plots
    inner = gridspec.GridSpecFromSubplotSpec(
        len(print_names), 1, subplot_spec=main_fig[1], wspace=0.0, hspace=0.0)

    label_colors = [
        'cyan', 'yellow', 'lime', 'red', 'magenta', 'lavenderblush'
        # 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'
    ]
    label_names = [
        "Lift", "Hand", "Grab", "Supinate", "At Mouth", "Chew"
    ]

    # ylabels = ["ground truth", "wasserstein"]
    bar_height = 0.01
    ylabels = print_names
    # data_mat = [data["gt"], data["pred"]]
    num_frames = data["gt"].shape[0]
    # for j in range(len(print_names)):
    for j in range(1):
        loss_key = loss_names[j]

        ax = plt.Subplot(fig, inner[j])
        # create the prediction bar
        for k in range(num_frames):
            if any(data[loss_key][k, :] > 0):
                idx = numpy.argmax(data[loss_key][k, :])
                # ax.plot([k, k], [0, 1], color=(0,1.0,0.0,1.0))
                # try:
                ax.plot([k, k], [0, bar_height], label_colors[idx])
                # except:
                #     import pdb; pdb.set_trace()

        # plot frame indicator
        ax.plot([start_frame + frame_idx, start_frame + frame_idx],
                [0, bar_height], 'snow')
                # [0, bar_height], '0.6')
        # ax.plot(data["lift"][:, j+1])
        # ax.set_ylabel(ylabels[j])
        ax.set_ylim([0, bar_height])
        # ax.set_xlim([0, num_frames])
        ax.set_xlim([start_frame, end_frame])
        ax.set_xticks([])
        # if j != len(print_names) - 1:
        #     ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    plt.tight_layout()
    # main_fig.update(wspace=0.05, hspace=0.05)
    canvas.draw()       # draw the canvas, cache the renderer
    s, (width, height) = canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    X = numpy.fromstring(s, numpy.uint8).reshape((height, width, 4))
    plt.close('all')

    # ... bgr ...
    bgr_frame = X[:, :, :3].copy() # copy and get rid of alpha chan
    bgr_frame[:, :, 0] = X[:, :, 2]
    bgr_frame[:, :, 2] = X[:, :, 0]

    # add the time to the frame.
    # import pdb; pdb.set_trace()
    font = cv2.FONT_HERSHEY_SIMPLEX
    seconds_count = (frame_idx / 500.0)
    # temp = numpy.floor(seconds_count / 10)
    temp = numpy.floor(frame_idx / 10)
    offset = 0
    while temp > 0:
        offset = offset + 1
        temp = numpy.floor(temp / 10)

    # xoffset = 870 - offset * 20
    # cv2.putText(bgr_frame, "%.2f" % seconds_count, (xoffset, 360), font, 1, (255, 255, 0), 2)
    xoffset = 930 - offset * 15
    # cv2.putText(bgr_frame, "%.3f" % seconds_count, (xoffset, 360), font, 0.75, (255, 255, 255), 1)
    cv2.putText(bgr_frame, "%d" % frame_idx, (xoffset, 360), font, 0.75, (255, 255, 255), 1)


    # add label texts?
    # cv2.putText(bgr_frame, "GT", (270, 290), font, 0.5, (255, 255, 255), 1)
    # cv2.putText(bgr_frame, "Wasserstein", (200, 310), font, 0.5, (255, 255, 255), 1)
    # cv2.putText(bgr_frame, "Matching", (220, 330), font, 0.5, (255, 255, 255), 1)
    # cv2.putText(bgr_frame, "MSE", (255, 350), font, 0.5, (255, 255, 255), 1)
    cv2.putText(bgr_frame, "GT", (270, 350), font, 0.5, (255, 255, 255), 1)

    # for each label, add the label names
    base_x = 310
    y_coords = [290, 310, 330, 350]
    y_coords = [350, 310, 330, 350]
    x_offsets = [
        310, 345, 395, 440, 515, 595
    ]
    label_bgr = [
        (255, 255, 0), (0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 255),
        (245, 240, 255)
    ]
    alpha = 0

    # for i in range(len(y_coords)):
    for i in range(1):
        loss_key = loss_names[i]
        for j in range(len(label_colors)):
            # for the loss, find the closest label idx from the current
            # frame.
            label_idxs = numpy.argwhere(data[loss_key][:, j]).flatten()
            min_dist = numpy.inf
            for k in range(len(label_idxs)):
                if numpy.abs((frame_idx + start_frame) - label_idxs[k]) < min_dist:
                    min_dist = numpy.abs((frame_idx + start_frame) - label_idxs[k])
            alpha = 1 - numpy.min([30, min_dist]) / 30

            color = list(label_bgr[j])
            for k in range(len(color)):
                color[k] = int(color[k] * alpha)

            color = tuple(color)
            # if min_dist < 30:
            #     import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            if alpha > 0:
                cv2.putText(
                    bgr_frame, label_names[j],
                    (x_offsets[j], y_coords[i]),
                    font, 0.5, color, 1)

    return bgr_frame


def process_csv(filename):
    values = []
    with open(filename, "r") as fid:
        # first line is header
        line = fid.readline().strip()
        line = fid.readline()
        while line:
            # csv, want columns 2 and 3.
            scores = line.split(',')[:3]
            scores = [float(scores[0]), float(scores[1]), float(scores[2])]
            values.append(scores)
            line = fid.readline()

    values = numpy.asarray(values)
    frame_thresh = 0.7
    gt_sup, gt_idx = post_processing.nonmax_suppress(
        values[:, 2], frame_thresh)
    predict_sup, predict_idx = post_processing.nonmax_suppress(
        values[:, 1], frame_thresh)

    values = numpy.stack([values[:, 0], predict_sup, gt_sup]).T

    return values


def load_results(base_path):
    labels = ["lift", "hand", "grab", "supinate", "mouth", "chew"]
    all_data = {}

    for i in range(len(labels)):
        filename = os.path.join(base_path, "predict_%s.csv" % labels[i])
        data = process_csv(filename)
        all_data[labels[i]] = data
        # (depending on how we decide to plot), might be easier to have all
        # the gt and predictions in two matricies
    num_frames = all_data["lift"].shape[0]
    all_gt = numpy.zeros((num_frames, 20))
    all_pred = numpy.zeros((num_frames, 20))

    for i in range(num_frames):
        for j in range(len(labels)):
            all_gt[i, j] = all_data[labels[j]][i, 2]
            all_pred[i, j] = all_data[labels[j]][i, 1]

    all_data["gt"] = all_gt
    all_data["pred"] = all_pred

    return all_data, num_frames


def setup_args(argv):
    flags = gflags.FLAGS
    flags(argv)

    return flags


def get_frame_range(data, losses):
    # get ranges
    rows, cols = data["gt"].shape

    points = numpy.argwhere(data["gt"])
    all_starts = [numpy.min(points[:, 0])]
    all_ends = [numpy.max(points[:, 0])]

    for i in range(len(losses)):
        loss_name = losses[i]["loss"]
        points = numpy.argwhere(data[loss_name])
        pred_start_frame = numpy.min(points[:, 0])
        pred_end_frame = numpy.max(points[:, 0])

        all_starts.append(pred_start_frame)
        all_ends.append(pred_end_frame)

    start_frame = numpy.min(all_starts) - 30
    end_frame = numpy.max(all_ends) + 30

    return start_frame, end_frame


def get_main_exp_folder(base_dir, params):
    exps = os.listdir(base_dir)
    for exp in exps:
        all_keys = numpy.zeros((len(params), 1))
        keys = list(params.keys())
        for i in range(len(keys)):
            key = keys[i]
            if params[key] in exp:
                all_keys[i] = 1
        if all_keys.sum() == len(params):
            return exp


def merge_data(all_data, losses, print_names):
    # gt's should be the same
    data = {
        'gt': all_data[0]['gt'],
        'names': print_names
    }
    loss_names = ["gt"]
    for i in range(len(losses)):
        loss_name = losses[i]["loss"]
        data[loss_name] = all_data[i]["pred"]
        loss_names.append(loss_name)
    data["loss_names"] = loss_names
    return data


def main(argv):
    flags = setup_args(argv)

    # for each loss, get the results
    losses = [
        {'loss': 'wasserstein'},
        {'loss': 'hungarian'},
        {'loss': 'mse'},
    ]
    print_names =["Ground Truth", "MSE", "Wasserstein", "Matching"]
    # print_names =["Ground Truth", "Wasserstein", "Matching", "MSE"]
    # print_names =["Ground Truth"]
    results = []
    for i in range(len(losses)):
        loss_dir = get_main_exp_folder(flags.input_dir, losses[i])

        exp_dir = os.path.join(
            flags.input_dir, loss_dir, "predictions", "test", flags.experiment
        )
        print(exp_dir)
        data, num_frames = load_results(exp_dir)
        results.append(data)

    data = merge_data(results, losses, print_names)
    start_frame, end_frame = get_frame_range(data, losses)

    cap = cv2.VideoCapture(
        os.path.join(exp_dir, "movie_comb.avi")
    )
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_delta = end_frame - start_frame

    # create output move
    basename = flags.experiment

    ret, frame = cap.read()
    print("%d %d" % (start_frame, end_frame))
    i = 0
    tic = time.time()
    base_img_out = os.path.join(
        flags.output_dir, basename
    )
    if not os.path.exists(base_img_out):
        os.makedirs(base_img_out)
    while ret and i < frame_delta:
        # plot_frame = create_frame(frame, data, i, start_frame, end_frame)
        bgr_frame = create_frame(frame, data, i, start_frame, end_frame)
        # writer.write(plot_frame[:, :, :3])
        # bgr ....
        # rgb_frame = plot_frame.copy()
        # rgb_frame[:, :, 0] = plot_frame[:, :, 2]
        # rgb_frame[:, :, 2] = plot_frame[:, :, 0]
        cv2.imwrite(
            os.path.join(base_img_out, '%05d.png' % i),
            bgr_frame
        )

        ret, frame = cap.read()
        i = i + 1
    print("process time: %f" % (time.time() - tic))
    cap.release()
    # writer.release()
    movie_out = os.path.join(flags.output_dir, "%s.mp4" % basename)
    command = "/usr/bin/ffmpeg -framerate 30 -pattern_type glob -i '%s/*.png' -c:v libx264 -preset veryslow %s" % (base_img_out, movie_out)
    print(command)
    return_code = subprocess.call(command, shell=True)


if __name__ == "__main__":
    main(sys.argv)
