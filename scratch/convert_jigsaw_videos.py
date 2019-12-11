import os
import subprocess
# import cv2
# import numpy

org_dir = "/media/drive3/kwaki/data/jigsaw/reorg/videos"
out_dir = "/media/drive3/kwaki/data/jigsaw/reorg/mp4_videos2"


def convert_video(in_file, out_file):
    # going through opencv didn't seem to work... going to just use a system
    # call.
    # cap = cv2.VideoCapture(in_file)
    # num_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    # width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

    # fourcc = cv2.cv.CV_FOURCC(*'X264')
    # writer = cv2.VideoWriter(
    #     out_file, fourcc, fps, (width, height), isColor=True
    # )

    # for frame_i in range(num_frames):
    #     retval, img = cap.read()
    #     writer.write(img)

    # cap = cv2.VideoCapture(in_file)
    # fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # cap.release()
    # create the command
    command = "/opt/ffmpeg/bin/ffmpeg -y -i %s -c:v libx264 -f mp4 -acodec aac %s -r 30.000030"
    # print(command % (in_file, out_file))

    subprocess.call(
        command % (in_file, out_file), shell=True
    )


def main():
    video_names = os.listdir(org_dir)
    video_names.sort()

    for video_name in video_names:
        in_file = os.path.join(org_dir, video_name)
        out_file = os.path.join(out_dir, video_name)
        convert_video(in_file, out_file)


if __name__ == "__main__":
    main()
