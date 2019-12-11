"""Check the difference between the arena points..."""
# import scipy.io as sio
import cv2
# import pickle
import numpy
from sklearn.externals import joblib

# matfile = 'ToneVsLaserData20150717.mat'
# picklefile = '/localhome/kwaki/data/hantman/test.pkl'
# with open(picklefile, 'rb') as infile:
#     data = pickle.load(infile)
infile = '/localhome/kwaki/data/hantman/joblib/test.npy'
data = joblib.load(infile)

# create a cropped movie?
crop_start = data['crops'][1]['crops'][0] + data['crops'][1]['idx'][0]
idx = 0
labels = data['labels'][1]
is_step = False

# for each movie, loop over all the frames, and grab the features at each
# behavior
num_exp = len(data['features'])
lift = {'features': None}
hand = {'features': None}
grab = {'features': None}
sup = {'features': None}
mouth = {'features': None}
chew = {'features': None}
mean_images = [lift, hand, grab, sup, mouth, chew]
label_counts = numpy.zeros((6,))
label_names = data['label_names']
for i in range(num_exp):
    # print i
    # for each label, find the frames for that label
    for j in range(len(label_names)):
        # print "\t%d" % j
        idx = numpy.where(data['labels'][i][:, j] > .999)
        for k in range(len(idx[0])):
            # print "\t\t%d" % k
            row_feat = data['features'][i][idx[0][k], :]
            # cat_image = row_feat.reshape((70, 140))
            # cv2.imshow('frame', cat_image)
            # cv2.waitKey()
            if mean_images[j]['features'] is None:
                mean_images[j]['features'] = row_feat
            else:
                mean_images[j]['features'] += row_feat
            label_counts[j] = label_counts[j] + 1

for i in range(len(label_names)):
    mean_images[i]['features'] = mean_images[i]['features'] / label_counts[i]
    cat_image = mean_images[i]['features'][0:9800].reshape((70, 140))
    cv2.imwrite(label_names[i] + '.png', cat_image * 255)
    print label_names[i]
    print mean_images[i]['features'][9800:]
