import os
import cv2


def get_fps(video_name):
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def main():
    video_dir = "/groups/branson/bransonlab/kwaki/data/thumos14/videos"
    videos = os.listdir(video_dir)

    with open("/groups/branson/bransonlab/kwaki/data/thumos14/meta/fps.txt", "w") as fid:
        videos.sort()
        for i in range(len(videos)):
            fps = get_fps(os.path.join(video_dir, videos[i]))
            exp_name = videos[i].split('.')[0]
            fid.write("%s, %f\n" % (exp_name, fps))
    
if __name__ == "__main__":
    main()
