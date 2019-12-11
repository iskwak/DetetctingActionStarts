import h5py


def main():
    fname = '/media/drive3/data.mat'
    with h5py.File(fname, 'r') as h5_data:
        keys = h5_data["data"].keys()
        for key in keys:
            print(key)


if __name__ == "__main__":
    main()
