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

gflags.DEFINE_string("input_dir", "", "input folder")
gflags.DEFINE_string("output_dir", "", "output folder")


def create_frame(frame, data, frame_idx, start_frame, end_frame):
    fig = plt.figure(figsize=(10, 10), dpi=100)
    canvas = FigureCanvas(fig)
    main_fig = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0)

    # figure handle for the video frames
    frame_fig = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=main_fig[0], wspace=0.0, hspace=0.0)

    ax = plt.Subplot(fig, frame_fig[0])
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

    # create the handles for the ethogram style plots
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=main_fig[1], wspace=0.0, hspace=0.0)

    label_colors = [
        'b-', 'g-', 'r-', 'c-', 'm-'
    ]

    ylabels = ["ground truth", "wasserstein"]
    data_mat = [data["gt"], data["pred"]]
    num_frames = data_mat[0].shape[0]
    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        # create the prediction bar
        for k in range(num_frames):
            if any(data_mat[j][k, :] > 0):
                idx = numpy.argmax(data_mat[j][k, :])
                # ax.plot([k, k], [0, 1], color=(0,1.0,0.0,1.0))
                ax.plot([k, k], [0, 1], label_colors[idx])
                # import pdb; pdb.set_trace()

        # plot frame indicator
        ax.plot([start_frame + frame_idx, start_frame + frame_idx], [0, 1], '0.6')
        # ax.plot(data["lift"][:, j+1])
        ax.set_ylabel(ylabels[j])
        ax.set_ylim([0, 1])
        # ax.set_xlim([0, num_frames])
        ax.set_xlim([start_frame, end_frame])
        if j != 1:
            ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    plt.tight_layout()
    canvas.draw()       # draw the canvas, cache the renderer
    s, (width, height) = canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    X = numpy.fromstring(s, numpy.uint8).reshape((height, width, 4))
    plt.close('all')
    return X


def process_csv(filename):
    values = []
    with open(filename, "r") as fid:
        # first line is header
        line = fid.readline().strip()
        line = fid.readline()
        while line:
            # print(line.strip())
            # csv, want columns 2 and 3.
            scores = line.split(',')[:3]
            scores = [float(scores[0]), float(scores[1]), float(scores[2])]
            values.append(scores)
            line = fid.readline()
    values = numpy.asarray(values)
    return values


def load_results(base_path):
    labels = ["lift", "hand", "grab", "supinate", "mouth", "chew"]
    all_data = {}

    for i in range(len(labels)):
        filename = os.path.join(base_path, "processed_%s.csv" % labels[i])
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


def get_frame_range(gt, pred):
    # get ranges
    rows, cols = gt.shape
    points = numpy.argwhere(gt)
    gt_start_frame = numpy.min(points[:, 0])
    gt_end_frame = numpy.max(points[:, 0])

    points = numpy.argwhere(pred)
    pred_start_frame = numpy.min(points[:, 0])
    pred_end_frame = numpy.max(points[:, 0])

    start_frame = numpy.min([gt_start_frame, pred_start_frame]) - 30
    end_frame = numpy.min([gt_end_frame, pred_end_frame]) + 30

    return start_frame, end_frame


def main(argv):
    flags = setup_args(argv)
    data, num_frames = load_results(
        os.path.join(flags.input_dir, './')
    )
    start_frame, end_frame = get_frame_range(data["gt"], data["pred"])
    
    cap = cv2.VideoCapture(
        os.path.join(flags.input_dir, "movie_comb.avi")
    )
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_delta = end_frame - start_frame
    
    # create output move
    basename = os.path.basename(flags.input_dir)
    movie_name = os.path.join(flags.output_dir, "%s.avi" % basename)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # writer = cv2.VideoWriter(
    #     movie_name, fourcc=fourcc, fps=fps, frameSize=(1000, 1000))

    ret, frame = cap.read()
    print(num_frames)
    i = 0
    tic = time.time()
    base_img_out = os.path.join(
        flags.output_dir, basename
    )
    if not os.path.exists(base_img_out):
        os.makedirs(base_img_out)
    while ret and i < frame_delta:
        plot_frame = create_frame(frame, data, i, start_frame, end_frame)
        # if i > 200:
        #     break
        # import pdb; pdb.set_trace()
        # writer.write(plot_frame[:, :, :3])
        cv2.imwrite(
            os.path.join(base_img_out, '%05d.png' % i),
            plot_frame
        )

        ret, frame = cap.read()
        i = i + 1
    print("process time: %f" % (time.time() - tic))
    cap.release()
    # writer.release()
    # use ffmpeg and system calls to make the movie
    movie_out = os.path.join(flags.output_dir, "%s.mp4" % basename)
    command = "/usr/bin/ffmpeg -framerate 30 -pattern_type glob -i '%s/*.png' -c:v libx264 -preset veryslow %s" % (base_img_out, movie_out)
    print(command)
    return_code = subprocess.call(command, shell=True) 


if __name__ == "__main__":
    main(sys.argv)
