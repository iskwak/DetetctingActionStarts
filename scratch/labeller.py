"""Create a folder of frames for each hantman movie_comb.avi."""
import cv2
import os
from sklearn.externals import joblib
import numpy
import scipy.io as sio
import helpers.sequences_helper as sequences_helper
# import shutil
import helpers.paths as paths

rng = numpy.random.RandomState(456)
label_names = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]


def process_key(key, frame_num, labelled):
    """Process the key press."""
    frame_offset = numpy.nan
    # labelled = 0
    if key == 112:
        # toggle play/pause
        frame_offset = -1
    elif key == 2555904:
        # right key, frame_offset is 0 because the video will move on it's own.
        frame_offset = 0
    elif key == 2424832:
        # left key, frame_offset is -2 because the video will step forward one,
        # so jump back 2.
        frame_offset = -2
    elif key == 49:
        # '1' key, don't pause, just return the label.
        labelled.append([frame_num, 1])
    elif key == 50:
        # '1' key, don't pause, just return the label.
        labelled.append([frame_num, 2])
    elif key == 51:
        # '1' key, don't pause, just return the label.
        labelled.append([frame_num, 3])
    elif key == 52:
        # '1' key, don't pause, just return the label.
        labelled.append([frame_num, 4])
    elif key == 53:
        # '1' key, don't pause, just return the label.
        labelled.append([frame_num, 5])
    elif key == 54:
        # '1' key, don't pause, just return the label.
        labelled.append([frame_num, 6])
    elif key == 113:
        # 'q' key, quit viewing the video
        frame_offset = numpy.inf
    if key >= 49 and key <= 54:
        print "%f, %f, %d" % (frame_num, frame_offset, key)

    return frame_offset, labelled


def check_labels(frame_num, frame_idx, mat_data, labels):
    """If labels is not none, print the labeled behavior for the frame."""
    label_names = ['Lift_labl_t0sPos', 'Handopen_labl_t0sPos',
                   'Grab_labl_t0sPos', 'Sup_labl_t0sPos',
                   'Atmouth_labl_t0sPos', 'Chew_labl_t0sPos']

    # check the mat data first
    for label_name in label_names:
        t0s = mat_data[label_name][0]
        if frame_num in t0s:
            print "mat_data: %d, %s" %\
                (frame_num, label_name)

    frame_num = int(frame_num)
    # convert frame_num using frame_idx info
    new_idx = [idx for idx in range(len(frame_idx))
               if frame_idx[idx] == frame_num]

    if len(new_idx) == 0 and frame_num < labels.shape[0] + 50:
        return True
    if len(new_idx) > 0:
        new_idx = new_idx[0]
    else:
        new_idx = labels.shape[0]
    if new_idx >= labels.shape[0]:
        return False

    idx = numpy.argwhere(labels[new_idx, :])
    if idx.size > 0:
        print "\t%d, %d, %s" % (frame_num, idx[0] + 1, label_names[idx[0][0]])

    return True


def play_video(video_filename, frame_idx, mat_data, labels=None):
    """Play a video, and optionally display labels."""
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    base_rate = int(fps * 4)
    play_rate = base_rate
    have_warned = False

    ret, frame = cap.read()
    labelled = []
    still_labels = False
    while ret is True:
        cv2.imshow("frame", frame)
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # CAP_PROP_POS_FRAMES gives the next frame. so do -1 get current frame
        if mat_data is None:
            have_warned = True
        else:
            still_labels =\
                check_labels(frame_num - 1, frame_idx, mat_data, labels)
        if still_labels is False and have_warned is False:
            print "no more labels"
            have_warned = True

        key = cv2.waitKey(play_rate)
        # print key
        frame_offset, labelled = process_key(key, frame_num, labelled)
        # print numpy.isnan(frame_offset)
        if not numpy.isnan(frame_offset):
            # if frame_offset is nan, then a key was pressed
            if frame_offset == -1:
                # toggle play rate
                play_rate = base_rate if play_rate == 0 else 0
            elif numpy.isinf(frame_offset):
                break
            else:
                # always set the play rate to 0
                play_rate = 0
                # adjust the frame number
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + frame_offset)

        ret, frame = cap.read()

    cv2.destroyWindow("frame")
    cap.release()
    labelled = numpy.asarray(labelled)
    print labelled
    return labelled


def view_labels(exp_name, data, mat_data):
    """View the labels of a video."""
    base_exp = os.path.basename(exp_name)
    exp_names = data['exp']
    # find the experiment in the data
    idx = [i for i in range(len(exp_names)) if exp_names[i] == base_exp]

    labels = data['labels'][idx[0]]
    frame_idx = data['frame_idx'][idx[0]]

    # open the video and play frame by frame.
    video_filename = os.path.join(exp_name, "movie_comb.avi")
    labelled = play_video(video_filename, frame_idx, mat_data, labels)
    return labelled


def write_predictions(out_dir, exp_name, labels, label_idx,
                      label_names_simple, data):
    """Write predictions in the manner of the other outputs."""
    """View the labels of a video."""
    base_exp = os.path.basename(exp_name)
    exp_names = data['exp']
    # find the experiment in the data
    idx = [i for i in range(len(exp_names)) if exp_names[i] == base_exp]

    ground_labels = data['labels'][idx[0]]
    frame_idx = data['frame_idx'][idx[0]]

    num_rows = ground_labels.shape[0]

    out_file = os.path.join(out_dir, "predict_%s.csv" % label_names_simple)
    with open(out_file, "w") as file:
        # write the header
        file.write("frame,")
        file.write(label_names_simple + ",")
        file.write(label_names_simple + " ground truth,")
        file.write("image\n")

        for i in range(num_rows):
            idx = numpy.argwhere(labels[:, 0] == frame_idx[i])
            if idx.size == 0:
                predict = 0
            else:
                if labels[idx[0], 1] - 1 == label_idx:
                    predict = 1
                else:
                    predict = 0

            # outfile.write("%f" % array[i][0])
            file.write("%f," % frame_idx[i])
            file.write("%f," % predict)
            file.write("%f," % ground_labels[i][label_idx])
            file.write("frames/%05d.jpg" % frame_idx[i])
            # for j in range(num_cols):
            #     outfile.write(",%f" % predict[i][j])
            #     outfile.write(",%f" % ground_truth[i][j])
            file.write("\n")

    return

base_path = 'C:\Users\ikwak\Desktop\data\hantman'
# base_path = "D:\data\hantman\hantman_pruned"
# frame_out = 'C:\Users\ikwak\Desktop\data\hantman_frames'
out_dir = ("C:\Users\\ikwak\\Desktop\\checkouts\\github\\QuackNN\\"
           "quackaction\\figs\\labels")
data_filename = ("joblib/no_convolve_2/data.npy")

exp_dirs = os.listdir(base_path)
exp_dirs.sort()
# only want to watch M134


print "Loading data..."
data = joblib.load(data_filename)
print "done"
exp_dirs = data['exp']
exp_dirs = [exp[0] for exp in exp_dirs if "M134" in exp[0]]
# import pdb; pdb.set_trace()

matfilename = "C:\Users\ikwak\Desktop\data\hantman\ToneVsLaserData20150717.mat"
matfile = sio.loadmat(matfilename)

rand_idx = rng.permutation(len(exp_dirs))
train_idx = rand_idx[0:10]
test_idx = rand_idx[10:20]
# for i in train_idx:
#     print exp_dirs[i]
#     exp_folder = os.path.join(base_path, exp_dirs[i])
#     idx = [idx for idx in range(len(matfile['rawdata']))
#            if matfile['rawdata'][idx][0]['exp'][0] == exp_dirs[i]]
#     mat_data = matfile['rawdata'][idx][0]
#     view_labels(exp_folder, data, mat_data)

print "DONE TRAINING"
label_names_simple = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
for i in test_idx:
    print exp_dirs[i]
    exp_folder = os.path.join(base_path, exp_dirs[i])
    idx = [j for j in range(len(matfile['rawdata']))
           if matfile['rawdata'][j][0]['exp'][0] == exp_dirs[i]]
    mat_data = matfile['rawdata'][j][0]
    labels = view_labels(exp_folder, data, None)

    # write labels
    exp_out_dir = os.path.join(out_dir, exp_dirs[i])
    # import pdb; pdb.set_trace()
    paths.create_dir(exp_out_dir)
    sequences_helper.copy_templates(exp_out_dir)
    # for each label
    for j in range(len(label_names_simple)):
        write_predictions(exp_out_dir, exp_dirs[i], labels, j,
                          label_names_simple[j], data)
    import pdb; pdb.set_trace()