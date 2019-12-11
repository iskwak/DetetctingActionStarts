"""Convert matlab mouse data into a pickle."""
import numpy
import h5py
from sklearn.decomposition import IncrementalPCA
# import shutil
import time
from helpers.RunningStats import RunningStats
from sklearn.externals import joblib

fname = "/media/drive1/data/hantman_processed/20170827_vgg/data.hdf5"
out_name = "/media/drive1/data/hantman_processed/20170827_vgg/stats/stats_small.pkl"

ipca_side = IncrementalPCA(n_components=2000, batch_size=400)
ipca_front = IncrementalPCA(n_components=2000, batch_size=400)
load = False
if load is True:
    stats_data = joblib.load(out_name)

    ipca_side = stats_data["ipca_side"]
    ipca_front = stats_data["ipca_front"]

    img_side_stats = stats_data["img_side_stats"]
    img_front_stats = stats_data["img_front_stats"]
    sampled = stats_data["sampled"]
else:
    # first get the mean and variance
    with h5py.File(fname, "r") as h5_data:
        exps = h5_data["exps"].keys()
        sampled = numpy.random.permutation(exps)
        sampled = sampled[0:10]
        img_side_stats = RunningStats(4096)
        img_front_stats = RunningStats(4096)

        all_hoghof = []
        all_pos = []
        for exp in sampled:
            print exp
            tic = time.time()
            img_side = h5_data["exps"][exp]["img_side"].value
            img_front = h5_data["exps"][exp]["img_front"].value

            img_side_stats.add_data(
                img_side.reshape((img_side.shape[0], img_side.shape[2])))
            img_front_stats.add_data(
                img_front.reshape((img_front.shape[0], img_front.shape[2])))

            print "\t%f" % (time.time() - tic)

    img_side_std = numpy.sqrt(img_side_stats.var) + numpy.finfo("float32").eps
    img_front_std = numpy.sqrt(img_front_stats.var) + numpy.finfo("float32").eps
    import pdb; pdb.set_trace()
    # run the IPCA algorithm
    with h5py.File(fname, "r") as h5_data:
        exps = h5_data["exps"].keys()

        temp_side = []
        temp_front = []
        tic = time.time()
        total_tic = time.time()
        for exp in exps:
            print exp
            img_side = h5_data["exps"][exp]["img_side"].value
            img_front = h5_data["exps"][exp]["img_front"].value

            img_side = img_side.reshape(
                (img_side.shape[0], img_side.shape[2]))
            img_front = img_front.reshape(
                (img_front.shape[0], img_front.shape[2]))

            temp_side.append((img_side - img_side_stats.mean) / img_side_std)
            temp_front.append((img_front- img_front_stats.mean) / img_front_std)

            # num rows should be the same in side and front
            num_rows = sum([data.shape[0] for data in temp_side])
            if num_rows > 2000:
                # once the number of rows is greater than the number of
                # components run the ipca algo
                img_side = numpy.concatenate(temp_side)
                img_front = numpy.concatenate(temp_front)

                print img_side.shape
                # ipca_side.partial_fit(img_side)
                # ipca_front.partial_fit(img_front)
                print "\t%f" % (time.time() - tic)
                tic = time.time()
                # reset the temp_data list
                temp_data = []
    print "total tic: %f" % (time.time() - total_tic)
    # save the data to disk... ipca computation is super slow.
    stats_data = {
        "ipca_side": ipca_side,
        "ipca_front": ipca_front,
        "img_side_stats": img_side_stats,
        "sampled": sampled
    }

    # cum_sum_side = numpy.cumsum(ipca_side.explained_variance_ratio_)
    # cum_sum_front = numpy.cumsum(ipca_front.explained_variance_ratio_)

    joblib.dump(stats_data, out_name)


