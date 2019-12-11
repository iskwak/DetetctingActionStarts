import os
import numpy


def write_csv(name, headers, data):
    with open(name, "w") as fid:
        fid.write("%s" % headers[0])
        for i in range(1, len(headers)):
            fid.write(",%s" % headers[i])
        fid.write("\n")

        r, c = data.shape
        for i in range(r):
            fid.write("%f" % data[i, 0])
            for j in range(1, c):
                fid.write(",%f" % data[i, j])
            fid.write("\n")


def bin_center(edges):
    centers = []
    for i in range(1, len(edges)):
        centers.append((edges[i] - edges[i - 1]) / 2.0 + edges[i - 1])

    return numpy.array(centers)


def create_histogram(mse, hung, wass, label):
    # histo = numpy.histogram(all_lift)
    temp = list(range(-9, 10, 1))
    # temp = list(range(-10, 11, 1))
    # temp = list(range(-52, 62, 5))
    # temp[0] = -400
    # temp[-1] = 400
    histo_mse = numpy.histogram(mse[label], temp)
    histo_hung = numpy.histogram(hung[label], temp)
    histo_wass = numpy.histogram(wass[label], temp)

    centers = bin_center(histo_mse[1])
    histo2 = numpy.stack([
        centers,
        histo_mse[0],
        histo_hung[0],
        histo_wass[0]
    ]).T

    out_dir = "/nrs/branson/kwaki/outputs/analysis/histogram/fps/%s" % label
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    write_csv(out_dir + "/data.csv",
              ["bins", "MSE", "Matching", "Wasserstein"],
              histo2)



def main():
    mse = numpy.load("/nrs/branson/kwaki/outputs/analysis/histogram/fps/weighted.npy")
    mse = mse.item()
    hung = numpy.load("/nrs/branson/kwaki/outputs/analysis/histogram/fps/hungarian.npy")
    hung = hung.item()
    wass = numpy.load("/nrs/branson/kwaki/outputs/analysis/histogram/fps/wasserstein.npy")
    wass = wass.item()

    label_names = [
        "lift", "hand", "grab", "supinate", "mouth", "chew"]
    for label in label_names:
        create_histogram(
            mse, hung, wass, label)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
