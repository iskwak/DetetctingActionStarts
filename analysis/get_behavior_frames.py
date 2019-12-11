import os
import h5py
import cv2
import numpy


def get_write_frame(cap, frame_idx, type, view, label, out_dir):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if type == "flow":
        frame = (frame.astype('float') - 127) * 4 + 127
    cv2.imwrite(
        os.path.join(out_dir, "%s_%s_%s.png" % (label, type, view)), frame)


def main():
    h5_name = "/nrs/branson/kwaki/data/20180729_base_hantman/exps/M134_20150504_v033"

    front_rgb = "/nrs/branson/kwaki/data/videos/hantman_rgb/front/M134_20150504_v033.avi"
    side_rgb = "/nrs/branson/kwaki/data/videos/hantman_rgb/side/M134_20150504_v033.avi"
    front_rgb_cap = cv2.VideoCapture(front_rgb)
    side_rgb_cap = cv2.VideoCapture(side_rgb)

    front_flow = "/nrs/branson/kwaki/data/videos/hantman_flow/front/M134_20150504_v033.avi"
    side_flow = "/nrs/branson/kwaki/data/videos/hantman_flow/side/M134_20150504_v033.avi"
    front_flow_cap = cv2.VideoCapture(front_flow)
    side_flow_cap = cv2.VideoCapture(side_flow)

    out_dir = "/nrs/branson/kwaki/outputs/analysis/frames"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    label_names = [
        "lift", "hand", "grab", "sup", "mouth", "chew"
    ]
    all_caps = [
        front_rgb_cap, side_rgb_cap, front_flow_cap, side_flow_cap
    ]
    types = [
        "rgb", "rgb", "flow", "flow"
    ]
    views = [
        "front", "side", "front", "side"
    ]
    with h5py.File(h5_name, "r") as h5_data:
        # open up the video for this behavior

        # for each label, figure out where the label occurs.
        for i in range(len(label_names)):
            label_name = label_names[i]
            frame_idx = numpy.argwhere(h5_data["labels"][:, i] == 1)[0][0]
            for j in range(len(all_caps)):
                get_write_frame(
                    all_caps[j], frame_idx,
                    types[j], views[j], label_name, out_dir)

        print("hi")

    front_rgb_cap.release()
    side_rgb_cap.release()
    front_flow_cap.release()
    side_flow_cap.release()
    

if __name__ == "__main__":
    main()
