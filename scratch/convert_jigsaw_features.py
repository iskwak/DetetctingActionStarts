"""Helper script to convert JIGSAW mat file features to hdf5"""
import h5py
import scipy.io
import os


def get_unique_mats(mat_dir):
    rel_paths = []
    mat_names = []
    # for each directory, add new mat files
    split_dirs = os.listdir(mat_dir)
    split_dirs.sort()
    for split in split_dirs:
        full_path = os.path.join(mat_dir, split)
        new_mats = os.listdir(full_path)
        # for each new mat file in the folder, only add if the base
        # file name isn't in the mat_names list.
        for new_mat in new_mats:
            if new_mat not in mat_names:
                mat_names.append(new_mat)
                rel_paths.append(
                    os.path.join(split, new_mat)
                )
    return rel_paths
                

def main():
    base_dir = '/groups/branson/bransonlab/kwaki/data/SpatialJIGSAWFeatures'
    mat_dir = os.path.join(base_dir, 'SpatialCNN')
    hdf_dir = os.path.join(base_dir, 'HDF5')

    # SpatialJIGSAWFeatures folder has 8 split folders, each with mat
    # files. However it seems that the mat files can be duplicates.
    # So get a list of unique mat files first.
    mat_files = get_unique_mats(mat_dir)
    mat_files.sort()
    print(mat_files)

    
if __name__ == "__main__":
    main()
