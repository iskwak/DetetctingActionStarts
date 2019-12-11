import h5py
import numpy

base_string = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_%s_test.hdf5'
mice = ['M134', 'M147', 'M173', 'M174']

for i in mice:
    fname = base_string % i

    with h5py.File(fname, "r") as hfile:
        dates = numpy.unique(hfile["date"])
        print("%s: %d" % (i, len(dates)))
        for j in range(len(dates)):
            num_exps = numpy.sum(hfile["date"][()] == dates[j])
            print("\t%s: %d" % (dates[j], num_exps))
        print("\t%d" % len(hfile["date"]))

        print("")
