"""Helper script to create hdf5 dataset for jigsaw."""
import os
import h5py
import helpers.paths as paths
import helpers.git_helper as git_helper
import cv2
import numpy
import re
import multiprocessing


g_label_dir = '/media/drive3/kwaki/data/jigsaw/reorg/labels'
g_video_dir = '/media/drive3/kwaki/data/jigsaw/reorg/videos'
g_out_dir = '/media/drive3/kwaki/data/jigsaw/20180619_jigsaw_base'

g_smooth_window = 19
g_smooth_std = 2


def create_opts():
    """Create an opts dictionary."""
    opts = dict()
    opts["out_dir"] = g_out_dir
    opts["label_dir"] = g_label_dir
    opts["video_dir"] = g_video_dir
    opts["smooth_std"] = g_smooth_std
    opts["smooth_window"] = g_smooth_window

    return opts


def create_clip(opts, video_file, cap, start_frame, end_frame, behav_id):
    """Given a video and idx, create a clip around the idx."""
    # 30 frames per second video
    basename = os.path.basename(os.path.splitext(video_file)[0])
    clip_dir = os.path.join(
        os.path.dirname(opts["label_dir"]),
        "clips1",
        basename
    )
    if not os.path.isdir(clip_dir):
        os.makedirs(clip_dir)

    # first create the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    retval, frame = cap.read()
    if retval is False:
        # weird...
        import pdb; pdb.set_trace()
    frame_name = os.path.join(
        clip_dir,
        basename + "_%d_G%d" % (start_frame, behav_id + 1) + ".jpg"
    )
    cv2.imwrite(frame_name, frame)

    # next create a little clip of the behavior.
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = numpy.max([0, start_frame - 30])
    end_idx = numpy.min([end_frame + 30, cap_frames])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    # create a writer
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    out_name = os.path.join(
        clip_dir,
        basename + "_%d_G%d" % (start_frame, behav_id + 1) + ".avi"
    )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print("%d %d" % (width, height))
    writer = cv2.VideoWriter(
        out_name, fourcc, 30.0, (width, height), isColor=True)
    for frame_i in range(start_idx, end_idx + 1):
        retval, frame = cap.read()
        writer.write(frame)
    writer.release()
    # if "Knot_Tying_F004_capture1" in video_file:
    #     import pdb; pdb.set_trace()


def load_movie(cap):
    # load the movie as uint8 array.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = numpy.zeros((cap_frames, height, width, 3), dtype='uint8')
    for i in range(cap_frames):
        ret, frame = cap.read()
        all_frames[i, :] = frame

    return all_frames


def process_label_file(opts, label_file, video_file):
    """Load the labels, then smooth them."""
    cap = cv2.VideoCapture(os.path.join(opts["video_dir"], video_file))
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != 640:
        print("%s: %d, %d" % (video_file, width, height))

    org_labels = numpy.zeros((cap_frames, 15))
    # labels = numpy.zeros((cap_frames, 1, 15))
    full_label = os.path.join(opts["label_dir"], label_file)
    with open(full_label, "r") as fp:
        # First get the label info.
        line = fp.readline()
        while line:
            # each line is of the form (space delimited).
            # <start frame> <end frame> <label id>
            data_line = line.split(" ")
            # start frame is 1? matlab style!
            start_frame = int(data_line[0]) - 1
            end_frame = int(data_line[1]) - 1
            try:
                behav_id = int(re.findall("(?<=G)\d+", data_line[2])[0]) - 1
            except:
                import pdb; pdb.set_trace()
            org_labels[start_frame, behav_id] = 1

            # if we want to create clips of the behaviors to get an idea of
            # what they look like.
            # create_clip(opts, video_file, cap, start_frame, end_frame, behav_id)

            line = fp.readline()

        # load the movie as uint8 array
        # all_frames = load_movie(cap)
    cap.release()

    # labels[:, 0, :] = org_labels
    # import pdb; pdb.set_trace()
    return org_labels  # , labels, all_frames


def process_file(blob):
    opts = blob[0]
    label_file = blob[1]
    video_file = blob[2]
    # print(label_file)
    # print(video_file)
    # if "Knot" not in label_files[i]:
    #     break
    # process the label file and create label info.
    # make sure video_files[i] == label_files[i]
    # remove the extension
    label_base = os.path.basename(os.path.splitext(label_file)[0])
    video_base = os.path.basename(os.path.splitext(video_file)[0])
    if label_base not in video_base:
        import pdb; pdb.set_trace()
        # search for label_base name?
    # org_labels, labels, all_frames =\
    org_labels =\
        process_label_file(opts, label_file, video_file)

    subject_id = label_base[-4]

    # for now use the label base name...
    # store video name information in the hdf5 file
    # exp_hdf_name = os.path.join(opts["out_dir"], "exps", video_base)
    exp_hdf_name = os.path.join(opts["out_dir"], "exps", label_base)
    with h5py.File(exp_hdf_name, "w") as exp_data:
        # import pdb; pdb.set_trace()
        # exp_data["org_labels"] = org_labels
        exp_data["labels"] = org_labels
        exp_data["subject"] = subject_id
        exp_data["video_name"] = video_file
        # exp_data.create_dataset(
        #     "frames", all_frames.shape,
        #     chunks=(20, all_frames.shape[1], all_frames.shape[2], all_frames.shape[3]),
        #     compression="lzf", dtype='uint8')
        # exp_data["frames"][:] = all_frames

    return (label_base, subject_id)


def create_hdfs(opts, out_data, exp_dir):
    """loop over labels, and create the hdf5 with labels."""
    label_files = os.listdir(opts["label_dir"])
    video_files = os.listdir(opts["video_dir"])

    # sort the label_files and video_files
    label_files.sort()
    video_files.sort()
    # start with just capture 1's.
    # video_files = video_files[::2]
    # several capture 1's are weird res's. Capture 2 only has 1 weird one.
    video_files = video_files[1::2]

    # multiprocessing to deal with the hdf5 file creation.
    blobs = [
        (opts, label_file, video_file)
        for label_file, video_file in zip(label_files, video_files)
        # if "Knot" in label_file # DEBUG!!!!
    ]
    # labels_subjects = []
    # for blob in blobs:
    #     label_base, subject_id = process_file(blob)
    #     labels_subjects.append((label_base, subject_id))
    pool = multiprocessing.Pool(processes=10)
    labels_subjects = pool.map(process_file, blobs)
    pool.close()
    pool.join()

    # loop over label files, create labels for hdfs
    exp_names = []
    subject_names = []
    # for i in range(len(label_files)):
    for i in range(len(labels_subjects)):
        label_base = labels_subjects[i][0]
        subject_id = labels_subjects[i][1]
        # create the external link
        out_data["exps"][label_base] = h5py.ExternalLink(
            os.path.join("exps", label_base), "/"
        )
        exp_names.append(label_base)
        subject_names.append(subject_id)

    subject_names = numpy.unique(subject_names)
    return exp_names, subject_names


def main():
    opts = create_opts()

    paths.create_dir(opts["out_dir"])
    paths.save_command(opts, opts["out_dir"])

    # log the git information
    git_helper.log_git_status(
        os.path.join(opts["out_dir"], "00_git_status.txt"))

    exp_dir = os.path.join(opts["out_dir"], "exps")
    paths.create_dir(exp_dir)

    # logname = os.path.join(opts["out_dir"], "00_log.txt")
    # skipname = os.path.join(opts["out_dir"], "00_skipped.txt")
    out_name = os.path.join(opts["out_dir"], "data.hdf5")
    # with open(logname, "w") as log:
    #     with open(skipname, "w") as skip_log:
    with h5py.File(out_name, "w") as out_data:
        # create the exps group for external links.
        out_data.create_group("exps")
        exp_names, subject_names = create_hdfs(opts, out_data, exp_dir)

        # # fill the rest of the fields
        out_data["subjects"] = subject_names
        # out_data["tasks"]
        out_data["exp_names"] = exp_names


if __name__ == "__main__":
    main()
