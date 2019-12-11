"""Wrapper for feedforward testing."""
import hantman_3dconv


# arg_string = [
#     "hantman_3dconv.py",
#     "--train_file", "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_train.hdf5",
#     "--test_file", "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_test.hdf5",
#     "--valid_file", "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_valid.hdf5",
#     "--learning_rate", "0.0001",
#     "--hantman_mini_batch", "2",
#     "--total_epochs", "50",
#     "--display_dir", "/nrs/branson/kwaki/data/hantman_mp4",
#     "--video_dir", "/nrs/branson/kwaki/data/hantman_pruned",
#     "--cuda_device", "1",
#     "--reweight",
#     "--save_iterations", "10",
#     "--update_iterations", "10",
#     "--frames", "-5 -4 -3 -2 -1 0 1 2 3 4 5",
#     "--out_dir", "/nrs/branson/kwaki/outputs/20190105_test"
# ]
arg_string = [
    "hantman_3dconv.py",
    "--train_file", "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_debug_M134_train.hdf5",
    "--test_file", "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_debug_M134_test.hdf5",
    "--valid_file", "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_debug_M134_valid.hdf5",
    "--learning_rate", "0.0001",
    "--hantman_mini_batch", "2",
    "--total_epochs", "5",
    "--display_dir", "/nrs/branson/kwaki/data/hantman_mp4",
    "--video_dir", "/nrs/branson/kwaki/data/hantman_pruned",
    "--cuda_device", "1",
    "--reweight",
    "--save_iterations", "2",
    "--update_iterations", "2",
    "--frames", "-5 -4 -3 -2 -1 0 1 2 3 4 5",
    "--out_dir", "/nrs/branson/kwaki/outputs/20190105_test"
]



hantman_3dconv.main(arg_string)
