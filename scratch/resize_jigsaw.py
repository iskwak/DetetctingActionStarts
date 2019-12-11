import os
# import subprocess
import cv2
# import numpy

org_dir = "/media/drive3/kwaki/data/jigsaw/reorg/videos"
out_dir = "/media/drive3/kwaki/data/jigsaw/reorg/resized"


def resize_video(in_file, out_file):
    # going through opencv didn't seem to work... going to just use a system
    # call.
    cap = cv2.VideoCapture(in_file)
    num_frames1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(
        out_file, fourcc, fps, (640, 480), isColor=True
    )

    for frame_i in range(num_frames1):
        retval, img = cap.read()
        # resize the image
        if width < 640:
            img = cv2.resize(img, (640, 480),
                             interpolation=cv2.INTER_CUBIC)
        writer.write(img)
    cap.release()
    writer.release()
    # print("%s: %d, %d, %d, %f" %
    #       (in_file, num_frames, width, height, fps))
    # a = fourcc & 255
    # b = (fourcc >> 8) & 255
    # c = (fourcc >> 16) & 255
    # d = (fourcc >> 24) & 255
    # print("%c%c%c%c" % (a, b, c, d))

    # ffmpeg directly
    # command = (
    #     "LD_LIBRARY_PATH=/opt/ffmpeg/lib:$LD_LIBRARY_PATH "
    #     "/opt/ffmpeg/bin/ffmpeg -i %s "
    #     "-vf scale=640:-1 -c:a copy -r %f %s"
    # )

    # if width < 640:
    #     print(command % (in_file, fps, out_file))
    #     subprocess.call(
    #         command % (in_file, fps, out_file), shell=True
    #     )
    #     # get stats of the new video?
    # cap = cv2.VideoCapture(out_file)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # num_frames2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # for i in range(num_frames2):
    #     retval, img = cap.read()
    #     # cv2.imshow('moo', img)
    #     # cv2.waitKey(int(fps))

    # print("\t%d, %d, %f, %d, %d" %
    #       (width, height, fps, num_frames1, num_frames2))


def main():
    video_names = os.listdir(org_dir)
    video_names.sort()

    for video_name in video_names:
        print(video_name)
        in_file = os.path.join(org_dir, video_name)
        out_file = os.path.join(out_dir, video_name)
        resize_video(in_file, out_file)


if __name__ == "__main__":
    main()
