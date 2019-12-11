import scratch.create_2dconv_hantman_features as create_2dconv_hantman_features

# test data
argv = [
    '/groups/branson/home/kwaki/checkouts/QuackNN/scratch/create_2dconv_hantman_features.py',
    '--input_name', '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_test.hdf5',
    '--out_dir', '/nrs/branson/kwaki/data/20180710_2dconv_feat',
    '--load_network', '/nrs/branson/kwaki/outputs/20180709_2dconvforward/20180710-0.0005_0/networks/391127/network.pt',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0'
]

create_2dconv_hantman_features.main(argv)


# train data
argv = [
    '/groups/branson/home/kwaki/checkouts/QuackNN/scratch/create_2dconv_hantman_features.py',
    '--input_name', '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_train.hdf5',
    '--out_dir', '/nrs/branson/kwaki/data/20180710_2dconv_feat',
    '--load_network', '/nrs/branson/kwaki/outputs/20180709_2dconvforward/20180710-0.0005_0/networks/391127/network.pt',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0'
]

create_2dconv_hantman_features.main(argv)


# valid data
argv = [
    '/groups/branson/home/kwaki/checkouts/QuackNN/scratch/create_2dconv_hantman_features.py',
    '--input_name', '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_valid.hdf5',
    '--out_dir', '/nrs/branson/kwaki/data/20180710_2dconv_feat',
    '--load_network', '/nrs/branson/kwaki/outputs/20180709_2dconvforward/20180710-0.0005_0/networks/391127/network.pt',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0'
]

create_2dconv_hantman_features.main(argv)


# check the split data.
print("checking splits.")
import helpers.check_hantman_splits as check_hantman_splits
check_hantman_splits.main("/nrs/branson/kwaki/data/20180710_2dconv_feat/")
