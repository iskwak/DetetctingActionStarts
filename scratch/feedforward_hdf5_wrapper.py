"""Wrapper for feedforward testing."""
import feedforward_hdf5

DEBUG = True
if DEBUG is True:
    train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5'
    test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5'
else:
    train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_train.hdf5'
    test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5'

arg_string = [
    'feedforward.py',
    '--train_file', train_file,
    '--test_file', test_file,
    '--learning_rate', '0.01',
    '--hantman_mini_batch', '10',
    '--total_epochs', '100',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--save_iterations', '5',
    '--update_iterations', '5',
    '--out_dir', '/nrs/branson/kwaki/outputs/tests/feedforward',
    '--feat_keys', 'hoghof',
    '--cuda_device', '0',
    '--reweight',
    '--normalize',
    '--hantman_arch', 'bidirconcat',
    '--lstm_hidden_dim', '32',
    '--lstm_num_layers', '1',
    '--use_pool',
]


feedforward_hdf5.main(arg_string)
