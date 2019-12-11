import threaded_hungarian_mouse

DEBUG = True
if DEBUG is True:
    train_file = '/nrs/branson-v/kwaki/hantman/hantman_split_M134_valid.hdf5'
    test_file = '/nrs/branson-v/kwaki/hantman/hantman_split_M134_valid.hdf5'
    valid_file = '/nrs/branson-v/kwaki/hantman/hantman_split_M134_valid.hdf5'
else:
    train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_train.hdf5'
    test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_test.hdf5'
    valid_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_valid.hdf5'

arg_string = [
    'threaded_hungarian_mouse.py',
    '--train_file', train_file,
    '--test_file', test_file,
    '--valid_file', valid_file,
    '--learning_rate', '0.0001',
    '--mini_batch', '10',
    '--total_epochs', '25',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--max_workers', '1',
    '--cuda_device', '0',
    '--reweight',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/20190302_test',
    '--hantman_arch', 'bidirconcat',
    '--lstm_hidden_dim', '256',
    '--lstm_num_layers', '2',
    '--feat_keys', 'rgb_i3d_view1_fc,rgb_i3d_view2_fc',
    '--use_pool',
    '--loss', 'weighted_mse',
    '--normalize'
]

threaded_hungarian_mouse.main(arg_string)
