"""Helper script to analyze jigsaw data."""
import os
import h5py
import helpers.template_helper as template_helper
import re
import numpy
import shutil
import json

# goal of the script...
# create an "analysis" folder?
# stats, number of labels
# locations of the labels? maybe location of first label in the video
# and location lf the last label.
# do we need


# additionally, create the "gt only" video files. viewer
# maybe 2 versions? one with bouts and one without...
# do we need the ability to zoom into the thing? thats a pain to add...w


# stats:
# bout lengths for each behavior
# difference between start frames
# try to figure out the "cannonical" sequence for each task
label_translation = {
    "G1": "Reaching for needle with right hand",
    "G2": "Positioning needle",
    "G3": "Pushing needle through tissue",
    "G4": "Transferring needle from left to right",
    "G5": "Moving to center with needle in grip",
    "G6": "Pulling suture with left hand",
    "G7": "Pulling suture with right hand",
    "G8": "Orienting needle",
    "G9": "Using right hand to help tightne suture",
    "G10": "Loosening more suture",
    "G11": "Dropping suture at end and moving to end points",
    "G12": "Reaching for needle with left hand",
    "G13": "Making C loop around right hand",
    "G14": "Reaching for suture with right hand",
    "G15": "Pulling suture with both hands"
}


def get_bout_labels(exp_name):
    org_label_dir = "/media/drive3/kwaki/data/jigsaw/reorg/labels"
    bout_filename = os.path.join(org_label_dir, exp_name + ".txt")
    bouts = []
    seq = []
    with open(bout_filename, "r") as bout_data:
        for line in bout_data:
            line_split = re.split(" ", line.strip())
            # the first two elements are the start and end frame of the bout
            # the 3rd is the bout label and the last element is a spare space.
            bouts.append(
                (int(line_split[0]), int(line_split[1]), line_split[2])
            )
            seq.append(line_split[2])
    return bouts, seq


def create_html_file(proc_dir, csv_name, video_name):
    in_filename = os.path.join("templates/predict_movie_template.html")
    out_filename = os.path.join(proc_dir, csv_name[:-3] + "html")
    with open(in_filename, "r") as template:
        with open(out_filename, "w") as out_file:
            keyval_dict = {
                 "csv": '"' + csv_name + '"',
                 "movie": '"' + video_name + '"',
                 "fps": 30
            }
            for line in template:
                new_line = template_helper.parse_line(line, keyval_dict)
                out_file.write(new_line)


def copy_templates(proc_dir, video_name):
    shutil.copy("templates/require.js", proc_dir)
    shutil.copy("templates/movie_viewer.js", proc_dir)
    # next create the html files
    create_html_file(proc_dir, "bout.csv", video_name)
    create_html_file(proc_dir, "start.csv", video_name)


def get_label_dists(bouts):
    dists = []
    for i in range(1, len(bouts)):
        dists.append(
            bouts[i][0] - bouts[i - 1][0]
        )

    return dists


def process_exp(out_dir, data, exp_name):
    exp = data["exps"][exp_name]
    video_dir = "/media/drive3/kwaki/data/jigsaw/reorg/mp4_videos"
    # org_label_dir = "/media/drive3/kwaki/data/jigsaw/reorg/labels"
    # setup the output space
    proc_dir = os.path.join(out_dir, exp_name)
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    # get the task
    exp_info = re.split("_", exp_name)
    # special case suturing...
    if "Suturing" in exp_info[0]:
        task = exp_info[0]
    else:
        task = exp_info[0] + "_" + exp_info[1]

    # symlink the movie
    video_name = exp["video_name"].value
    in_video = os.path.join(video_dir, video_name)
    out_video = os.path.join(proc_dir, video_name)
    if os.path.lexists(out_video):
        os.remove(out_video)
    os.symlink(in_video, out_video)

    # get bout information
    bouts, label_seq = get_bout_labels(exp_name)

    # create two label csv files. One with bouts and one with just the start
    # points.
    num_frames = exp["labels"].shape[0]
    bout_csv_name = os.path.join(proc_dir, "bout.csv")
    with open(bout_csv_name, "w") as bout_fd:
        # first the header
        # unique labels as the header
        unique_labels = numpy.unique(label_seq).tolist()
        bout_fd.write("frame")
        for label_i in unique_labels:
            bout_fd.write(",%s" % label_i)
        bout_fd.write("\n")
        # next the data.
        for i in range(num_frames):
            bout_fd.write("%d" % i)

            # the way the labels are stored, seems easier to figure which
            # behaviors are active, and then write
            label_vec = numpy.zeros((len(unique_labels)))
            for j in range(len(bouts)):
                if bouts[j][0] <= i + 1 and bouts[j][1] >= i + 1:
                    # get the index into the unique labels
                    idx = unique_labels.index(bouts[j][2])
                    # use this to update the label_vec
                    label_vec[idx] = 1
            # write the label_vec to the csv
            for score in label_vec:
                bout_fd.write(",%d" % score)
            bout_fd.write("\n")

    # create the start only csv
    start_csv_name = os.path.join(proc_dir, "start.csv")
    labels = exp["org_labels"].value
    with open(start_csv_name, "w") as start_fd:
        # same thing, write the header
        unique_labels = numpy.unique(label_seq).tolist()
        start_fd.write("frame")
        for label_i in unique_labels:
            start_fd.write(",%s" % label_i)
        start_fd.write("\n")
        # next the data.
        # for the data, first figure out which frame indicies to care about.
        idx_list = []
        for label_i in unique_labels:
            label_id = int(label_i[1:])
            # label_id = unique_labels.index(label_i)
            idx_list.append(label_id)

        # actually write the data to disk
        for i in range(num_frames):
            start_fd.write("%d" % i)
            for idx_i in idx_list:
                start_fd.write(",%d" % labels[i, idx_i - 1])
            start_fd.write("\n")

    # copy over templates
    copy_templates(proc_dir, video_name)

    # get distances between labels
    dists = get_label_dists(bouts)

    # create a dictionary of stats for this experiment.
    bout_sizes = [bout[1] - bout[0] for bout in bouts]
    task_info = {
        "exp_name": exp_name,
        "task": task,
        "bouts": bouts,
        "sequence": label_seq,
        "num_bouts": len(bouts),
        "num_frames": num_frames,
        "bout_sizes": bout_sizes,
        "label_dists": dists
    }

    return task_info


def main():
    base_dir = "/media/drive3/kwaki/data/jigsaw/"
    data_file = os.path.join(base_dir, "20180424_jigsaw_base/data.hdf5")
    out_dir = os.path.join(base_dir, "analysis")
    plot_base_dir = os.path.join(out_dir, "plots")

    if not os.path.exists(plot_base_dir):
        os.makedirs(plot_base_dir)

    # setup stats dictionary
    # labels = ["G%d" % i for i in range(1, 16)]
    # stats_list = []
    # for i in range(len(labels)):
    #     stats_dict{
    #         "label": labels[i],
    #         "num_bouts": 0,
    #         "ave_bout": 0,
    #     }
    #     stats_list.append(
    #         stats_dict
    #     )
    # # the stats_list is a per label stat thing.
    # # need to figure out the "average" sequence and the distance between
    # # labels for the tasks.
    # tasks = ["Knot_Tying", "Needle_Passing", "Suturing"]
    # task_stats = []
    # for i in range(len(tasks)):
    #     task_dict{
    #         "task": tasks[i],
    #         "ave_labels": 0,
    #         "ave_bout_size": 0,
    #         "exp_names": [],
    #         "all_labels": [],
    #         "total_labels": 0
    #         "total_bout_lengths": 0
    #     }

    tasks = {
        "Knot_Tying": [],
        "Needle_Passing": [],
        "Suturing": []
    }
    # load up the hdf5 file.
    base_hdf5_name = "/media/drive3/kwaki/data/jigsaw/20180424_jigsaw_base/data.hdf5"
    with h5py.File(base_hdf5_name, "r") as data:
        exp_names = data["exp_names"].value
        exp_names.sort()

        for exp_name in exp_names:
            task_info = process_exp(plot_base_dir, data, exp_name)
            tasks[task_info["task"]].append(
                task_info
            )
            # if task_info["task"] == "Knot_Tying":
            #     import pdb; pdb.set_trace()
            # # break
    # write the task data to disk
    json_name = os.path.join(out_dir, "info.json")
    with open(json_name, "w") as json_fd:
        json.dump(task_info, json_fd)

    # import pdb; pdb.set_trace()
    print("hi")

    all_diffs = []
    task_diffs = {
        "Knot_Tying": [],
        "Needle_Passing": [],
        "Suturing": []
    }

    all_seqs = []
    task_seqs = {
        "Knot_Tying": [],
        "Needle_Passing": [],
        "Suturing": []
    }
    for task in tasks:
        task_infos = tasks[task]
        for i in range(len(task_infos)):
            task_diffs[task] = task_diffs[task] + task_infos[i]["label_dists"]
            all_diffs = all_diffs + task_infos[i]["label_dists"]
            task_seqs[task] = task_seqs[task] + task_infos[i]["sequence"]
            all_seqs = all_seqs + task_infos[i]["sequence"]

    label_names = numpy.unique(all_seqs).tolist()
    for label_name in label_names:
        count = all_seqs.count(label_name)
        print("%s: %d" % (label_name, count))
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
