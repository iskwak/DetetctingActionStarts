import h5py
import os
import numpy
import cv2


# action end and full labels version....
def create_h5_file(base_dir, exp_dir, label_id, video_name, fps_dict, action_starts, action_ends):
    full_video = os.path.join(base_dir, "videos", "%s.mp4" % video_name)
    cap = cv2.VideoCapture(full_video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps = float(fps_dict[video_name])

    full_labels = numpy.zeros((num_frames, 20))
    end_labels = numpy.zeros((num_frames, 20))
    action_starts.sort()

    exp_filename = os.path.join(exp_dir, video_name)
    with h5py.File(exp_filename, "a") as h5:
        print(exp_filename)
        # h5["labels"] = labels
        if "end_labels" in h5.keys():
            end_labels = h5["end_labels"]
        if "full_labels" in h5.keys():
            full_labels = h5["full_labels"]

        for i in range(len(action_starts)):
            start_idx = int(action_starts[i] * fps)
            end_idx = int(action_ends[i] * fps)
            try:
                full_labels[start_idx:end_idx + 1, label_id] = 1
                end_labels[end_idx, label_id] = 1
            except:
                import pdb; pdb.set_trace()

        if "end_labels" in h5.keys():
            # end_labels = h5["end_labels"]
            del h5["end_labels"]
        if "full_labels" in h5.keys():
            # full_labels = h5["full_labels"]
            del h5["full_labels"]
        h5["end_labels"] = end_labels
        h5["full_labels"] = full_labels

        # add features
        if "canned_i3d_rgb_64_past" not in h5.keys():
            h5["canned_i3d_rgb_64_past"] = h5py.ExternalLink(
                os.path.join("rgb_64_past", video_name),
                "/canned_i3d_rgb_64_past"
            )


# # action start version....
# def create_h5_file(base_dir, exp_dir, label_id, video_name, fps_dict, action_starts):
#     full_video = os.path.join(base_dir, "videos", "%s.mp4" % video_name)
#     cap = cv2.VideoCapture(full_video)
#     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()
#
#     fps = float(fps_dict[video_name])
#
#     labels = numpy.zeros((num_frames, 20))
#     action_starts.sort()
#
#     exp_filename = os.path.join(exp_dir, video_name)
#     with h5py.File(exp_filename, "a") as h5:
#         # h5["labels"] = labels
#
#         # mod labels
#         # labels = h5["labels"]
#         # for i in range(len(action_starts)):
#         #     frame_idx = int(action_starts[i] * fps)
#         #     try:
#         #         labels[frame_idx, label_id] = 1
#         #     except:
#         #         import pdb; pdb.set_trace()
#         # del h5["labels"]
#         # h5["labels"] = labels
#         # if "video_name" not in h5.keys():
#         #     h5["video_name"] = "%s.mp4" % video_name
#
#         # add features
#         # h5["canned_i3d_rgb_64"] = h5py.ExternalLink(
#         #     os.path.join("canned_i3d_rgb_64", video_name),
#         #     "/canned_i3d_rgb_64"
#         # )


def get_videos(base_dir, exp_dir, label_id, label_annotation_file, fps, label_name):
    videos = []
    with open(label_annotation_file, 'r') as fid:

        action_starts = []
        action_ends = []
        label_line = fid.readline().strip()
        # initialize the current video name
        curr_video = label_line.split()[0]
        videos.append(curr_video)
        while label_line:
            # print(label_line)
            # video_name, start_frame = label_line.split()[:2]
            video_name, start_frame, end_frame = label_line.split()
            if video_name != curr_video:
                create_h5_file(base_dir, exp_dir, label_id, curr_video, fps, action_starts, action_ends)
                # if the video_name doesn't match the current one, setup the
                # h5 file
                videos.append(video_name)
                curr_video = video_name
                action_starts = []
                action_ends = []
            action_starts.append(float(start_frame))
            action_ends.append(float(end_frame))
            label_line = fid.readline().strip()
        # need to do the last video too
        create_h5_file(base_dir, exp_dir, label_id, video_name, fps, action_starts, action_ends)
    return videos


def create_meta(filename, label_names, video_names):
    # with h5py.File(filename, "w") as h5data:
    #     h5data["exp_names"] = numpy.string_(video_names)
    #     h5data["label_names"] = numpy.string_(label_names)

    #     exp_group = h5data.create_group("exps")
    #     # create the external links
    #     for i in range(len(video_names)):
    #         video_name = video_names[i]
    #         if video_name in exp_group.keys():
    #             print(video_name)
    #             continue
    #         try:
    #             exp_group[video_name] = h5py.ExternalLink(
    #                 os.path.join("exps", video_name),
    #                 "/"
    #             )
    #         except:
    #             import pdb; pdb.set_trace()
    return

def main():
    base_dir = "/groups/branson/bransonlab/kwaki/data/thumos14/"
    exp_dir = os.path.join(base_dir, "h5data", "exps")
    labels_fname = os.path.join(base_dir, "meta", "labels.txt")

    label_names = []
    with open(labels_fname, "r") as fid:
        name = fid.readline().strip()
        while name:
            label_names.append(name)
            name = fid.readline().strip()

    # get the fps's...
    fps_dict = {}
    with open(os.path.join(base_dir, "meta", "fps.txt"), "r") as fid:
        line = fid.readline().strip()
        while line:
            name, fps = line.split(', ')
            fps_dict[name] = float(fps)
            line = fid.readline().strip()

    # loop over the videos
    valid_videos = []
    test_videos = []
    for i in range(len(label_names)):
        # create the data
        label_annotation_file = os.path.join(
            base_dir, 'meta', 'valid', 'annotation', '%s_val.txt' % label_names[i])
        videos = get_videos(base_dir, exp_dir, i, label_annotation_file, fps_dict, label_names[i])
        valid_videos = valid_videos + videos

        # also do test files
        label_annotation_file = os.path.join(
            base_dir, 'meta', 'test', 'annotation', '%s_test.txt' % label_names[i])
        videos = get_videos(base_dir, exp_dir, i, label_annotation_file, fps_dict, label_names[i])
        test_videos = test_videos + videos

    # create the meta files for train/test
    # valid_filename = os.path.join(base_dir, "h5data", "train.hdf5")
    # create_meta(valid_filename, label_names, valid_videos)
    # test_filename = os.path.join(base_dir, "h5data", "test.hdf5")
    # create_meta(test_filename, label_names, test_videos)


if __name__ == "__main__":
    main()
