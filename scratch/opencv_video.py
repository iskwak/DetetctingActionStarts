import cv2
# import numpy as np

filename = '/media/drive1/data/temp/M134_20141204_v002/movie_comb.avi'
vidcap = cv2.VideoCapture(filename)

# get framerate
fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
print fps

while vidcap.isOpened():
    ret, frame = vidcap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(int(1.0 / fps * 1000))

# vidcap.release()

# vidcap.set(cv2.CAP_PROP_POS_MSEC,20000)      # just cue to 20 sec. position
# success,image = vidcap.read()
# if success:
#     cv2.imwrite("frame20sec.jpg", image)     # save frame as JPEG file
#     cv2.imshow("20sec",image)
#     cv2.waitKey()
