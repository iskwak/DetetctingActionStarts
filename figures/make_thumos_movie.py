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


def create_frame(frame, labels):
    return

def main(argv):
    base_dir = '/groups/branson/bransonlab/kwaki/data/thumos14/videos'
    video_name = 'video_validation_0000266'
    # video_name = 'video_test_0001495'
    if 'video_validation_0000266' in video_name:
        labels = [
            ('Baseball Pitch', 72.8),
            ('Frisbee Catch', 9.6),
            ('Frisbee Catch', 12.4),
            ('Frisbee Catch', 22.0),
            ('Frisbee Catch', 40.7),
            ('Frisbee Catch', 54.2),
            ('Frisbee Catch', 58.8),
            ('Frisbee Catch', 63.2),
            ('Frisbee Catch', 67.8),
            ('Frisbee Catch', 90.5),
            ('Frisbee Catch', 92.7),
            ('Frisbee Catch', 127.2),
            ('Frisbee Catch', 129.9),
            ('Frisbee Catch', 137.9),
        ]
    else:
        # else video name is video_test_0001495
        labels = [
            ('Long Jump', 0.0),
            ('Long Jump', 22.8),
            ('Long Jump', 31.7),
            ('Long Jump', 43.2),
            ('Long Jump', 61.6),
            ('Long Jump', 69.2),
            ('Long Jump', 98.3),
            ('Long Jump', 116.0),
            ('Long Jump', 123.1),
            ('Long Jump', 144.8),
            ('Long Jump', 159.9),
            ('Long Jump', 170.5),
            ('Long Jump', 181.8),
            ('Long Jump', 201.6),
            ('Long Jump', 212.5),
            ('Long Jump', 244.7),
            ('Long Jump', 264.0),
            ('Long Jump', 274.5),
            ('Long Jump', 298.8),
            ('Long Jump', 318.6),
            ('Long Jump', 328.8),
            ('Long Jump', 343.2),
            ('Long Jump', 361.7),
            ('Long Jump', 368.8),
            ('Long Jump', 389.5),
            ('Long Jump', 407.6),
            ('Long Jump', 418.4),
            ('Long Jump', 436.0),
            ('Long Jump', 455.2),
            ('Long Jump', 464.5),
            ('Long Jump', 487.5),
            ('Long Jump', 506.7),
            ('Long Jump', 514.4),
        ]

    cap = cv2.VideoCapture(
        os.path.join(base_dir, '%s.mp4' % video_name)
    )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # create output move
    base_out = '/groups/branson/bransonlab/kwaki/forkristin/thumos examples'
    movie_name = os.path.join(base_out, "%s.avi" % video_name)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(
        movie_name, fourcc=fourcc, fps=fps, frameSize=(width, height)
    )

    ret, frame = cap.read()
    print(num_frames)
    i = 0
    tic = time.time()

    if 'video_validation_0000266' in video_name:
        font = cv2.FONT_HERSHEY_SIMPLEX,
        corner = (200, 150)
        fontScale = 0.5
        fontColor = (255, 0, 255)
        lineType = 1
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX,
        corner = (200, 150)
        fontScale = 0.5
        fontColor = (255, 0, 255)
        lineType = 1
    while ret:
        # see if anything is close
        diff = 0
        text = ''
        found = False
        min_dist = 61
        offset = 60
        for j in range(len(labels)):
            if numpy.abs(labels[j][1] * 30 - i) < offset and\
                numpy.abs(labels[j][1] * 30 - i) < min_dist:
                text = labels[j][0]
                alpha = numpy.abs(labels[j][1] * 30 - i) / offset
                min_dist = numpy.abs(labels[j][1] * 30 - i)
                found = True

        if found == True:
            overlay = frame.copy()
            cv2.putText(
                overlay, text,
                corner,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, # font scale
                fontColor,
                lineType
            )
            cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)
        writer.write(frame)
        ret, frame = cap.read()
        i = i + 1

    print(i)
    print("process time: %f" % (time.time() - tic))
    cap.release()
    writer.release()


if __name__ == "__main__":
    main(sys.argv)
