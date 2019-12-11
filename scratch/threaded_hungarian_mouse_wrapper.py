# pdb is stupid
import sys
sys.path.append('./')

# import threaded_hungarian_mouse
import load_hungarian_mouse

DEBUG = True
# if DEBUG is True:
#     train_file = '/media/drive2/data/20180729_base_hantman/hantman_valid.hdf5'
#     test_file = '/media/drive2/data/20180729_base_hantman/hantman_valid.hdf5'
#     valid_file = '/media/drive2/data/20180729_base_hantman/hantman_valid.hdf5'
# else:
#     train_file = '/media/drive2/data/20180729_base_hantman/hantman_train.hdf5'
#     test_file = '/media/drive2/data/20180729_base_hantman/hantman_test.hdf5'
#     valid_file = '/media/drive2/data/20180729_base_hantman/hantman_valid.hdf5'
# if DEBUG is True:
#     train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_valid.hdf5'
#     test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_valid.hdf5'
#     valid_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_valid.hdf5'
# else:
#     train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_train.hdf5'
#     test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_test.hdf5'
#     valid_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_valid.hdf5'
if DEBUG is True:
    train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5'
    test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5'
    valid_file = ''
else:
    train_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_train.hdf5'
    test_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_test.hdf5'
    valid_file = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_valid.hdf5'

arg_string = [
    # 'threaded_hungarian_mouse_test.py',
    'load_hungarian_mouse.py',
    '--train_file', train_file,
    '--test_file', test_file,
    # '--valid_file', valid_file,
    '--learning_rate', '0.00001',
    '--mini_batch', '10',
    '--total_epochs', '5',
    '--display_dir', '/nrs/branson/kwaki/data/hantman_mp4',
    '--video_dir', '/nrs/branson/kwaki/data/hantman_pruned',
    '--cuda_device', '0',
    '--reweight',
    '--normalize',
    '--save_iterations', '10',
    '--update_iterations', '10',
    '--out_dir', '/nrs/branson/kwaki/outputs/tests/20190329_test',
    '--hantman_arch', 'bidirconcat',
    '--lstm_hidden_dim', '256',
    '--lstm_num_layers', '2',
    # '--feat_keys', 'rgb_i3d_view1_fc,rgb_i3d_view2_fc',
    '--feat_keys', 'hoghof',
    '--nouse_pool',
    # '--loss', 'weighted_mse',
    '--loss', 'hungarian',
    # '--loss', 'wasserstein',
    '--hantman_perframe_weight', '0.9',
    '--anneal_type', 'line_step',
    '--perframe_stop', '0.5',
]

load_hungarian_mouse.main(arg_string)
