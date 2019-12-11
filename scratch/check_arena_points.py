"""Check the difference between the arena points..."""
import scipy.io as sio
import cv2
import pickle
import numpy

trxfile = '/media/drive1/data/hantman_pruned/M134_20141203_v002/trx.mat'
videofile = \
    '/media/drive1/data/hantman_pruned/M134_20141203_v002/movie_comb.avi'
# trxfile = 'C:\Users\ikwak\Desktop\hantman\M134_20141203_v002/trx.mat'
# videofile = \
#     'C:\Users\ikwak\Desktop\hantman\M134_20141203_v002/movie_comb.avi'

# get xy coords
matfile = '/localhome/kwaki/data/hantman/mats/ToneVsLaserData20150717.mat'
# matfile = 'ToneVsLaserData20150717.mat'
picklefile = '/localhome/kwaki/data/hantman/test.pkl'

with open(picklefile, 'rb') as infile:
    data = pickle.load(infile)
# import pdb; pdb.set_trace()
cap = cv2.VideoCapture(videofile)
trx = sio.loadmat(trxfile)
mat = sio.loadmat(matfile)

x1 = mat['trxdata'][0][1]['x1']
y1 = mat['trxdata'][0][1]['y1']
x2 = mat['trxdata'][0][1]['x2']
y2 = mat['trxdata'][0][1]['y2']
# import pdb; pdb.set_trace()

food = trx['trx'][0][1]['arena']['food'][0][0][0]
foodfront = trx['trx'][0][1]['arena']['foodfront'][0][0][0]
mouth = trx['trx'][0][1]['arena']['mouth'][0][0][0]
mouthfront = trx['trx'][0][1]['arena']['mouthfront'][0][0][0]
perch = trx['trx'][0][1]['arena']['perch'][0][0][0]
print "food"
print food
print "foodfront"
print foodfront

# foodfront = foodfront + d
# draw the points, food will be blue, foodfront wil be red
# opencv is bgr...

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
foodfront[0] = foodfront[0] + width/2
mouthfront[0] = mouthfront[0] + width/2

# ... test video writing too
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                      (int(width), int(height)), isColor=True)

# create a cropped movie?
crop_start = data['crops'][1]['crops'][0] + data['crops'][1]['idx'][0]
idx = 0
while True:
    ret, frame = cap.read()
    if ret is False:
        break

    # print frame.shape[1]/2
    # print foodfront
    # print foodfront
    cv2.circle(frame, (int(food[0]), int(food[1])), 5, (255, 0, 0))
    cv2.circle(frame, (int(foodfront[0]), int(foodfront[1])), 5, (0, 0, 255))
    cv2.circle(frame, (int(mouth[0]), int(mouth[1])), 5, (255, 0, 0))
    cv2.circle(frame, (int(mouthfront[0]), int(mouthfront[1])), 5, (0, 0, 255))
    cv2.circle(frame, (int(perch[0]), int(perch[1])), 5, (255, 0, 0))

    # draw the paw pos
    pos1 = x1[idx][0] - 2
    pos2 = y1[idx][0] - 2
    cv2.rectangle(frame,
                  (int(pos1), int(pos2)),
                  (int(pos1 + 4), int(pos2 + 4)),
                  (255, 0, 0))

    pos1 = x2[idx][0] - 2
    pos2 = y2[idx][0] - 2
    cv2.rectangle(frame,
                  (int(pos1), int(pos2)),
                  (int(pos1 + 4), int(pos2 + 4)),
                  (0, 0, 255))

    # cv2.imshow('frame', frame)
    # import pdb; pdb.set_trace()
    cur_idx = idx - crop_start
    if cur_idx >= 0 and cur_idx < data['labels'][1].shape[0]:
        out.write(frame)
        # moo = cv2.waitKey(1)
        # if moo & 0xFF == ord('q'):
        #     break

    idx = idx + 1

cap.release()
cv2.destroyAllWindows()
out.release()

cap = cv2.VideoCapture('output.avi')
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print length
# create a cropped movie?
crop_start = data['crops'][1]['crops'][0] + data['crops'][1]['idx'][0]
idx = 0
labels = data['labels'][1]
is_step = False
while True:
    ret, frame = cap.read()
    if ret is False:
        break

    cv2.imshow('frame', frame)
    if numpy.any(labels[idx, :]):
        is_step = True
        print labels[idx]
        print data['label_names'][numpy.where(labels[idx, :])[0][0]]

    if is_step is True:
        moo = cv2.waitKey()
        if moo & 0xFF == 's':
            is_step = True
        else:
            is_step = False
    else:
        moo = cv2.waitKey(1)
        if moo & 0xFF == ord('s'):
            is_step = True
    idx = idx + 1
cap.release()
