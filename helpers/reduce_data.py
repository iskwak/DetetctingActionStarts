"""Convert matlab mouse data into a pickle."""
import numpy
import h5py
from sklearn.decomposition import IncrementalPCA
# import shutil
import time
from helpers.RunningStats import RunningStats
from sklearn.externals import joblib

fname = "/media/drive1/data/hantman_processed/20170711/data.hdf5"
out_name = "/media/drive1/data/hantman_processed/20170711/stats/stats_small.pkl"

ipca = IncrementalPCA(n_components=1000, batch_size=400)
load = False
if load is True:
    stats_data = joblib.load(out_name)

    ipca = stats_data["ipca"]
    hoghof_stats = stats_data["hoghof_stats"]
    pos_stats = stats_data["pos_stats"]
    sampled = stats_data["sampled"]
else:
    # first get the mean and variance
    with h5py.File(fname, "r") as h5_data:
        exps = list(h5_data["exps"].keys())
        sampled = numpy.random.permutation(exps)
        sampled = sampled[0:10]
        hoghof_stats = RunningStats(8000)
        pos_stats = RunningStats(10)

        all_hoghof = []
        all_pos = []
        for exp in sampled:
            print(exp)
            tic = time.time()
            hoghof = h5_data["exps"][exp]["hoghof"].value
            pos_features = h5_data["exps"][exp]["pos_features"].value

            hoghof_stats.add_data(hoghof)
            pos_stats.add_data(pos_features)

            # all_hoghof.append(hoghof)
            # all_pos.append(pos_features)

            print("\t%f" % (time.time() - tic))

    hoghof_std = numpy.sqrt(hoghof_stats.var) + numpy.finfo("float32").eps
    pos_std = numpy.sqrt(pos_stats.var)

    # run the IPCA algorithm
    with h5py.File(fname, "r") as h5_data:
        exps = list(h5_data["exps"].keys())

        temp_data = []
        tic = time.time()
        total_tic = time.time()
        for exp in sampled:
            print(exp)
            hoghof = h5_data["exps"][exp]["hoghof"].value

            temp_data.append((hoghof - hoghof_stats.mean) / hoghof_std)
            num_rows = sum([data.shape[0] for data in temp_data])
            if num_rows > 1000:
                # once the number of rows is greater than the number of
                # components run the ipca algo
                hoghof = numpy.concatenate(temp_data)
                print(hoghof.shape)
                ipca.partial_fit(hoghof)
                print("\t%f" % (time.time() - tic))
                tic = time.time()
                # reset the temp_data list
                temp_data = []
    print("total tic: %f" % (time.time() - total_tic))
    # save the data to disk... ipca computation is super slow.
    stats_data = {
        "ipca": ipca,
        "hoghof_stats": hoghof_stats,
        "pos_stats": pos_stats,
        "sampled": sampled
    }
    import pdb; pdb.set_trace()
    joblib.dump(stats_data, out_name)

# get the variance of the pca'ed data on the original data.
hoghof_pca_stats = RunningStats(500)
with h5py.File(fname, "r") as h5_data:
    exps = list(h5_data["exps"].keys())
    exps.sort()

    reduced_sum = numpy.zeros((510,), dtype="float32")
    count = 0
    # for exp in exps:
    for exp in sampled:
        print(exp)
        tic = time.time()
        hoghof = h5_data["exps"][exp]["hoghof"]
        hoghof = (hoghof - hoghof_stats.mean) / hoghof_std
        hoghof = numpy.dot(hoghof,
                           ipca.components_[:500, :].T)

        hoghof_pca_stats.add_data(hoghof)
        numpy.tile()
        # pos_features = h5_data["exps"][exp]["pos_features"]
        # pos_features = (pos_features - pos_stats.mean) / pos_std

        # reduced = reduced.reshape((reduced.shape[0], 1, reduced.shape[1]))
        # h5_data["exps"][exp]["reduced"] = reduced

        print("\t%f" % (time.time() - tic))
import pdb; pdb.set_trace()
#
# reduced_mean = reduced_sum / count
#
# with h5py.File(fname, "a") as h5_data:
#     exps = h5_data["exps"].keys()
#     exps.sort()
#
#     reduced_diff = numpy.zeros(reduced_mean.shape, dtype="float32")
#     count = 0
#     # for exp in exps:
#     for exp in sampled:
#         print exp
#         tic = time.time()
#         hoghof = h5_data["exps"][exp]["hoghof"]
#         hoghof = (hoghof - hoghof_mean) / hoghof_std
#         hoghof = numpy.dot(hoghof,
#                            ipca1.components_[:500, :].T)
#
#         pos_features = h5_data["exps"][exp]["pos_features"]
#         pos_features = (pos_features - pos_mean) / pos_std
#         pos_features = numpy.dot(pos_features,
#                                  ipca2.components_[:10, :].T)
#
#         # import pdb; pdb.set_trace()
#         reduced = numpy.concatenate(
#             [hoghof, pos_features], axis=1)
#         reduced_diff += numpy.square(reduced - reduced_mean).sum(axis=0)
#
#         count += reduced.shape[0]
#         # reduced = reduced.reshape((reduced.shape[0], 1, reduced.shape[1]))
#         # h5_data["exps"][exp]["reduced"] = reduced
#
#         print "\t%f" % (time.time() - tic)
#
# reduced_var = numpy.sqrt(reduced_diff / count)
# import pdb; pdb.set_trace()
#
# maxs = []
# mins = []
# with h5py.File(fname, "a") as h5_data:
#     exps = h5_data["exps"].keys()
#     exps.sort()
#
#     count = 0
#     # for exp in exps:
#     for exp in sampled:
#         print exp
#         tic = time.time()
#         hoghof = h5_data["exps"][exp]["hoghof"]
#         hoghof = (hoghof - hoghof_mean) / hoghof_std
#         hoghof = numpy.dot(hoghof,
#                            ipca1.components_[:500, :].T)
#
#         pos_features = h5_data["exps"][exp]["pos_features"]
#         pos_features = (pos_features - pos_mean) / pos_std
#         pos_features = numpy.dot(pos_features,
#                                  ipca2.components_[:10, :].T)
#
#         # import pdb; pdb.set_trace()
#         reduced = numpy.concatenate(
#             [hoghof, pos_features], axis=1)
#         reduced = reduced / reduced_var.max()
#         maxs.append(reduced.max())
#         mins.append(reduced.min())
#
#         count += reduced.shape[0]
#         # reduced = reduced.reshape((reduced.shape[0], 1, reduced.shape[1]))
#         # h5_data["exps"][exp]["reduced"] = reduced
#
#         print "\t%f" % (time.time() - tic)
# import pdb; pdb.set_trace()
