import os
import h5py
import shutil


def get_videos(label_annotation_file):
    videos = []
    with open(label_annotation_file, 'r') as fid:
        label_line = fid.readline().strip()
        while label_line:
            # print(label_line)
            video_name = label_line.split()[0]
            if video_name not in videos:
                videos.append(video_name)
            label_line = fid.readline().strip()
    return videos
            

def copy_videos(videos, type):
    base_out = '/groups/branson/bransonlab/kwaki/data/thumos14/videos/'
    if type == 'valid':
        input_dir = '/groups/branson/bransonlab/kwaki/data/thumos14/validation'
    else:
        input_dir = '/groups/branson/bransonlab/kwaki/data/thumos14/TH14_test_set_mp4'
    
    for video_name in videos:
        in_file = os.path.join(input_dir, "%s.mp4" % video_name)
        out_file = os.path.join(base_out, "%s.mp4" % video_name)
        # print("%s %s" % (in_file, out_file))
        shutil.copyfile(in_file, out_file)

def main():
    base_dir = '/groups/branson/bransonlab/kwaki/data/thumos14/'
    label_file = os.path.join(base_dir, 'meta', 'labels.txt')
    exp_dir = os.path.join(base_dir, 'thumos14', 'exps')

    label_names = []
    with open(label_file, 'r') as fid:
        label_name = fid.readline()
        while label_name:
            label_name = label_name.strip()
            # print(label_name)
            label_names.append(label_name)
            label_name = fid.readline()

    # copy test videos
    # for each file load the label file and get the labels.
    # also move files around
    for i in range(len(label_names)):
        # create the data
        label_annotation_file = os.path.join(
            base_dir, 'meta', 'valid', 'annotation', '%s_val.txt' % label_names[i])

        print(label_names[i])
        videos = get_videos(label_annotation_file)

        copy_videos(videos, 'valid')

    # next do the test folder.
    # copy test videos
    # for each file load the label file and get the labels.
    # also move files around
    for i in range(len(label_names)):
        # create the data
        label_annotation_file = os.path.join(
            base_dir, 'meta', 'test', 'annotation', '%s_test.txt' % label_names[i])

        print(label_names[i])
        videos = get_videos(label_annotation_file)
        copy_videos(videos, 'test')

    
if __name__ == "__main__":
    main()
