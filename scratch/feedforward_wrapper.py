"""Wrapper for feedforward testing."""
import feedforward


arg_string = [
    'feedforward.py',
    '--train_file', '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_train.hdf5',
    '--test_file', '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_test.hdf5',
    '--valid_file', '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_valid.hdf5',
    '--learning_rate', '0.0001',
    '--hantman_mini_batch', '10',
    '--total_epochs', '100',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '2',
    '--reweight',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/20180912_test'
]

feedforward.main(arg_string)
